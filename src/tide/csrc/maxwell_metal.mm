/*
 * Maxwell wave equation propagator — Metal (Apple GPU) bridge
 *
 * Objective-C++ wrapper that compiles Metal compute shaders at runtime and
 * exposes them through the same extern "C" ABI used by the CUDA backend,
 * allowing the Python layer to dispatch to Apple GPU transparently.
 *
 * Build: compiled as .mm (Objective-C++) and linked against Metal.framework
 *        and Foundation.framework.
 *
 * The Metal shader source is loaded from maxwell.metal at runtime (found
 * relative to this source file or the shared library). The Metal framework
 * compiles it on first use — no offline Metal compiler (xcrun metal) required.
 *
 * Buffer strategy: we use newBufferWithBytes (copy-in) rather than
 * newBufferWithBytesNoCopy because PyTorch MPS tensor data_ptr() values
 * are not page-aligned. After GPU execution, mutable outputs are copied
 * back via memcpy. On Apple Silicon unified memory the copies are fast.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

#include "storage_utils.h"

// ---------------------------------------------------------------------------
// GridParams must match the struct in maxwell.metal exactly
// ---------------------------------------------------------------------------
struct GridParams {
  int64_t ny;
  int64_t nx;
  int64_t shot_numel;
  int64_t n_shots;
  int64_t n_sources_per_shot;
  int64_t n_receivers_per_shot;
  int64_t pml_y0;
  int64_t pml_y1;
  int64_t pml_x0;
  int64_t pml_x1;
  int64_t fd_pad;
  float rdy;
  float rdx;
  bool ca_batched;
  bool cb_batched;
  bool cq_batched;
};

// BackwardParams must match the struct in maxwell.metal exactly
struct BackwardParams {
  int64_t ny;
  int64_t nx;
  int64_t shot_numel;
  int64_t n_shots;
  int64_t n_sources_per_shot;
  int64_t n_receivers_per_shot;
  int64_t pml_y0;
  int64_t pml_y1;
  int64_t pml_x0;
  int64_t pml_x1;
  int64_t fd_pad;
  float rdy;
  float rdx;
  float dt;
  bool ca_batched;
  bool cb_batched;
  bool cq_batched;
  bool ca_requires_grad;
  bool cb_requires_grad;
  int64_t step_ratio;
  int64_t store_offset;
  bool store_ey;
  bool store_curl;
};

// ---------------------------------------------------------------------------
// Metal state — lazily initialised, held for the process lifetime
// ---------------------------------------------------------------------------
namespace {

id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLLibrary> g_library = nil;

id<MTLComputePipelineState> g_pso_forward_h = nil;
id<MTLComputePipelineState> g_pso_forward_e = nil;
id<MTLComputePipelineState> g_pso_add_sources = nil;
id<MTLComputePipelineState> g_pso_record_recv = nil;
id<MTLComputePipelineState> g_pso_forward_e_storage = nil;
id<MTLComputePipelineState> g_pso_backward_h = nil;
id<MTLComputePipelineState> g_pso_backward_e_grad = nil;
id<MTLComputePipelineState> g_pso_convert_grad = nil;

std::once_flag g_init_flag;

// Find the maxwell.metal shader source file
NSString *find_metal_source() {
  // Strategy 1: look next to the shared library using dladdr
  Dl_info info;
  if (dladdr((const void *)find_metal_source, &info) && info.dli_fname) {
    NSString *libPath = [NSString stringWithUTF8String:info.dli_fname];
    NSString *libDir = [libPath stringByDeletingLastPathComponent];

    // Check: <libdir>/csrc/maxwell.metal (development layout)
    NSString *candidate =
        [libDir stringByAppendingPathComponent:@"csrc/maxwell.metal"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:candidate])
      return candidate;
    // Check: <libdir>/maxwell.metal (installed layout)
    candidate = [libDir stringByAppendingPathComponent:@"maxwell.metal"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:candidate])
      return candidate;
  }
  // Strategy 2: look relative to __FILE__ (build tree)
  NSString *srcDir = [[NSString stringWithUTF8String:__FILE__]
      stringByDeletingLastPathComponent];
  NSString *candidate =
      [srcDir stringByAppendingPathComponent:@"maxwell.metal"];
  if ([[NSFileManager defaultManager] fileExistsAtPath:candidate])
    return candidate;

  return nil;
}

void metal_init() {
  @autoreleasepool {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
      fprintf(stderr, "[TIDE Metal] No Metal device found.\n");
      return;
    }
    g_queue = [g_device newCommandQueue];

    NSString *metalSourcePath = find_metal_source();
    if (!metalSourcePath) {
      fprintf(stderr,
              "[TIDE Metal] Cannot find maxwell.metal shader source.\n");
      return;
    }

    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:metalSourcePath
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) {
      fprintf(stderr, "[TIDE Metal] Failed to read %s: %s\n",
              [metalSourcePath UTF8String],
              error ? [[error localizedDescription] UTF8String] : "unknown");
      return;
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    // Use mathMode instead of deprecated fastMathEnabled on macOS 15+
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&                                 \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    opts.mathMode = MTLMathModeFast;
#else
    opts.fastMathEnabled = YES;
#endif

    g_library = [g_device newLibraryWithSource:source
                                       options:opts
                                         error:&error];
    if (!g_library) {
      fprintf(stderr, "[TIDE Metal] Failed to compile Metal shaders: %s\n",
              error ? [[error localizedDescription] UTF8String] : "unknown");
      return;
    }

    // Create pipeline states
    auto make_pso = [&](const char *name) -> id<MTLComputePipelineState> {
      NSString *nsName = [NSString stringWithUTF8String:name];
      id<MTLFunction> func = [g_library newFunctionWithName:nsName];
      if (!func) {
        fprintf(stderr, "[TIDE Metal] Function '%s' not found.\n", name);
        return nil;
      }
      NSError *psoError = nil;
      id<MTLComputePipelineState> pso =
          [g_device newComputePipelineStateWithFunction:func error:&psoError];
      if (!pso) {
        fprintf(stderr, "[TIDE Metal] Failed to create PSO for '%s': %s\n",
                name, [[psoError localizedDescription] UTF8String]);
      }
      return pso;
    };

    g_pso_forward_h = make_pso("forward_kernel_h");
    g_pso_forward_e = make_pso("forward_kernel_e");
    g_pso_add_sources = make_pso("add_sources_ey");
    g_pso_record_recv = make_pso("record_receivers_ey");
    g_pso_forward_e_storage = make_pso("forward_kernel_e_with_storage");
    g_pso_backward_h = make_pso("backward_kernel_lambda_h");
    g_pso_backward_e_grad = make_pso("backward_kernel_lambda_e_with_grad");
    g_pso_convert_grad = make_pso("convert_grad_kernel");
  }
}

inline bool ensure_metal() {
  std::call_once(g_init_flag, metal_init);
  return (g_device && g_queue && g_library && g_pso_forward_h &&
          g_pso_forward_e && g_pso_add_sources && g_pso_record_recv);
}

// Create an MTLBuffer by copying data in (safe for any alignment)
inline id<MTLBuffer> make_buffer(const void *ptr, size_t length) {
  if (!ptr || length == 0)
    return nil;
  return [g_device newBufferWithBytes:ptr
                               length:length
                              options:MTLResourceStorageModeShared];
}

// Create a zero-initialised mutable buffer
inline id<MTLBuffer> make_zero_buffer(size_t length) {
  if (length == 0)
    return nil;
  id<MTLBuffer> buf =
      [g_device newBufferWithLength:length
                            options:MTLResourceStorageModeShared];
  memset([buf contents], 0, length);
  return buf;
}

// Copy MTLBuffer contents back to a host pointer
inline void copy_back(id<MTLBuffer> buf, void *dst, size_t length) {
  if (buf && dst && length > 0) {
    memcpy(dst, [buf contents], length);
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Forward propagation implementation
// ---------------------------------------------------------------------------
static void maxwell_tm_float_forward_mps_impl(
    float const *const ca, float const *const cb, float const *const cq,
    float const *const f, // [nt, n_shots, n_sources_per_shot]
    float *const ey, float *const hx, float *const hz, float *const m_ey_x,
    float *const m_ey_z, float *const m_hx_z, float *const m_hz_x,
    float *const r, // [nt, n_shots, n_receivers_per_shot]
    float const *const ay, float const *const by, float const *const ayh,
    float const *const byh, float const *const ax, float const *const bx,
    float const *const axh, float const *const bxh, float const *const ky,
    float const *const kyh, float const *const kx, float const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    float const rdy_h, float const rdx_h, float const dt_h, int64_t const nt,
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device,
    int64_t const fd_pad) {
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;
  (void)device;

  if (!ensure_metal()) {
    fprintf(stderr, "[TIDE Metal] Metal backend not available.\n");
    return;
  }

  int64_t const shot_numel_h = ny_h * nx_h;

  // Fill GridParams
  GridParams params;
  params.ny = ny_h;
  params.nx = nx_h;
  params.shot_numel = shot_numel_h;
  params.n_shots = n_shots_h;
  params.n_sources_per_shot = n_sources_per_shot_h;
  params.n_receivers_per_shot = n_receivers_per_shot_h;
  params.pml_y0 = pml_y0_h;
  params.pml_y1 = pml_y1_h;
  params.pml_x0 = pml_x0_h;
  params.pml_x1 = pml_x1_h;
  params.fd_pad = fd_pad;
  params.rdy = rdy_h;
  params.rdx = rdx_h;
  params.ca_batched = ca_batched_h;
  params.cb_batched = cb_batched_h;
  params.cq_batched = cq_batched_h;

  @autoreleasepool {
    // ---- Sizes ----
    size_t const field_bytes =
        (size_t)n_shots_h * (size_t)shot_numel_h * sizeof(float);
    size_t const model_bytes =
        (ca_batched_h ? field_bytes : (size_t)shot_numel_h * sizeof(float));
    size_t const model_bytes_cb =
        (cb_batched_h ? field_bytes : (size_t)shot_numel_h * sizeof(float));
    size_t const model_bytes_cq =
        (cq_batched_h ? field_bytes : (size_t)shot_numel_h * sizeof(float));
    size_t const pml_y_bytes = (size_t)ny_h * sizeof(float);
    size_t const pml_x_bytes = (size_t)nx_h * sizeof(float);
    size_t const src_count = (size_t)n_shots_h * (size_t)n_sources_per_shot_h;
    size_t const rec_count = (size_t)n_shots_h * (size_t)n_receivers_per_shot_h;
    size_t const f_step_bytes = src_count * sizeof(float);
    size_t const r_step_bytes = rec_count * sizeof(float);
    size_t const f_total_bytes = (size_t)nt * f_step_bytes;
    size_t const r_total_bytes = (size_t)nt * r_step_bytes;

    // ---- Create GPU buffers (copy-in) ----
    // Read-only model parameters
    id<MTLBuffer> buf_ca = make_buffer(ca, model_bytes);
    id<MTLBuffer> buf_cb = make_buffer(cb, model_bytes_cb);
    id<MTLBuffer> buf_cq = make_buffer(cq, model_bytes_cq);

    // Mutable fields (copy in initial state, copy back after)
    id<MTLBuffer> buf_ey = make_buffer(ey, field_bytes);
    id<MTLBuffer> buf_hx = make_buffer(hx, field_bytes);
    id<MTLBuffer> buf_hz = make_buffer(hz, field_bytes);
    id<MTLBuffer> buf_m_ey_x = make_buffer(m_ey_x, field_bytes);
    id<MTLBuffer> buf_m_ey_z = make_buffer(m_ey_z, field_bytes);
    id<MTLBuffer> buf_m_hx_z = make_buffer(m_hx_z, field_bytes);
    id<MTLBuffer> buf_m_hz_x = make_buffer(m_hz_x, field_bytes);

    // PML profiles (read-only)
    id<MTLBuffer> buf_ay = make_buffer(ay, pml_y_bytes);
    id<MTLBuffer> buf_ayh = make_buffer(ayh, pml_y_bytes);
    id<MTLBuffer> buf_ax = make_buffer(ax, pml_x_bytes);
    id<MTLBuffer> buf_axh = make_buffer(axh, pml_x_bytes);
    id<MTLBuffer> buf_by = make_buffer(by, pml_y_bytes);
    id<MTLBuffer> buf_byh = make_buffer(byh, pml_y_bytes);
    id<MTLBuffer> buf_bx = make_buffer(bx, pml_x_bytes);
    id<MTLBuffer> buf_bxh = make_buffer(bxh, pml_x_bytes);
    id<MTLBuffer> buf_ky = make_buffer(ky, pml_y_bytes);
    id<MTLBuffer> buf_kyh = make_buffer(kyh, pml_y_bytes);
    id<MTLBuffer> buf_kx = make_buffer(kx, pml_x_bytes);
    id<MTLBuffer> buf_kxh = make_buffer(kxh, pml_x_bytes);

    // Source and receiver indices (read-only)
    id<MTLBuffer> buf_sources_i =
        (n_sources_per_shot_h > 0)
            ? make_buffer(sources_i, src_count * sizeof(int64_t))
            : nil;
    id<MTLBuffer> buf_receivers_i =
        (n_receivers_per_shot_h > 0)
            ? make_buffer(receivers_i, rec_count * sizeof(int64_t))
            : nil;

    // Full source amplitudes [nt * n_shots * n_sources_per_shot]
    id<MTLBuffer> buf_f = (n_sources_per_shot_h > 0 && f_total_bytes > 0)
                              ? make_buffer(f, f_total_bytes)
                              : nil;

    // Full receiver output [nt * n_shots * n_receivers_per_shot]
    id<MTLBuffer> buf_r = (n_receivers_per_shot_h > 0 && r_total_bytes > 0)
                              ? make_buffer(r, r_total_bytes)
                              : nil;

    // GridParams constant buffer
    id<MTLBuffer> buf_params =
        [g_device newBufferWithBytes:&params
                              length:sizeof(GridParams)
                             options:MTLResourceStorageModeShared];

    // ---- Compute grid dimensions ----
    int64_t grid_x = nx_h - 2 * fd_pad + 2;
    int64_t grid_y = ny_h - 2 * fd_pad + 2;
    if (grid_x <= 0)
      grid_x = 1;
    if (grid_y <= 0)
      grid_y = 1;

    MTLSize gridSize_field = MTLSizeMake((NSUInteger)grid_x, (NSUInteger)grid_y,
                                         (NSUInteger)n_shots_h);

    NSUInteger tw = g_pso_forward_h.threadExecutionWidth;
    NSUInteger th_max = g_pso_forward_h.maxTotalThreadsPerThreadgroup;
    NSUInteger tg_x = tw;
    NSUInteger tg_y = th_max / tw;
    if (tg_y < 1)
      tg_y = 1;
    MTLSize threadGroupSize_field = MTLSizeMake(tg_x, tg_y, 1);

    MTLSize gridSize_src =
        MTLSizeMake((NSUInteger)n_sources_per_shot_h, (NSUInteger)n_shots_h, 1);
    MTLSize threadGroupSize_src = MTLSizeMake(
        MIN((NSUInteger)n_sources_per_shot_h, (NSUInteger)64), 1, 1);
    if (threadGroupSize_src.width == 0)
      threadGroupSize_src.width = 1;

    MTLSize gridSize_rec = MTLSizeMake((NSUInteger)n_receivers_per_shot_h,
                                       (NSUInteger)n_shots_h, 1);
    MTLSize threadGroupSize_rec = MTLSizeMake(
        MIN((NSUInteger)n_receivers_per_shot_h, (NSUInteger)64), 1, 1);
    if (threadGroupSize_rec.width == 0)
      threadGroupSize_rec.width = 1;

    // ---- Time stepping loop ----
    int64_t const total_steps = nt;
    int64_t const batch_size = MIN(total_steps, (int64_t)256);

    for (int64_t t_start = 0; t_start < total_steps; t_start += batch_size) {
      int64_t t_end = MIN(t_start + batch_size, total_steps);

      id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];

      for (int64_t t_local = t_start; t_local < t_end; ++t_local) {
        int64_t t = start_t + t_local;

        // ---- forward_kernel_h ----
        {
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_forward_h];
          [enc setBuffer:buf_cq offset:0 atIndex:0];
          [enc setBuffer:buf_ey offset:0 atIndex:1];
          [enc setBuffer:buf_hx offset:0 atIndex:2];
          [enc setBuffer:buf_hz offset:0 atIndex:3];
          [enc setBuffer:buf_m_ey_x offset:0 atIndex:4];
          [enc setBuffer:buf_m_ey_z offset:0 atIndex:5];
          [enc setBuffer:buf_ay offset:0 atIndex:6];
          [enc setBuffer:buf_ayh offset:0 atIndex:7];
          [enc setBuffer:buf_ax offset:0 atIndex:8];
          [enc setBuffer:buf_axh offset:0 atIndex:9];
          [enc setBuffer:buf_by offset:0 atIndex:10];
          [enc setBuffer:buf_byh offset:0 atIndex:11];
          [enc setBuffer:buf_bx offset:0 atIndex:12];
          [enc setBuffer:buf_bxh offset:0 atIndex:13];
          [enc setBuffer:buf_ky offset:0 atIndex:14];
          [enc setBuffer:buf_kyh offset:0 atIndex:15];
          [enc setBuffer:buf_kx offset:0 atIndex:16];
          [enc setBuffer:buf_kxh offset:0 atIndex:17];
          [enc setBuffer:buf_params offset:0 atIndex:18];
          [enc dispatchThreads:gridSize_field
              threadsPerThreadgroup:threadGroupSize_field];
          [enc endEncoding];
        }

        // ---- forward_kernel_e ----
        {
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_forward_e];
          [enc setBuffer:buf_ca offset:0 atIndex:0];
          [enc setBuffer:buf_cb offset:0 atIndex:1];
          [enc setBuffer:buf_hx offset:0 atIndex:2];
          [enc setBuffer:buf_hz offset:0 atIndex:3];
          [enc setBuffer:buf_ey offset:0 atIndex:4];
          [enc setBuffer:buf_m_hx_z offset:0 atIndex:5];
          [enc setBuffer:buf_m_hz_x offset:0 atIndex:6];
          [enc setBuffer:buf_ay offset:0 atIndex:7];
          [enc setBuffer:buf_ayh offset:0 atIndex:8];
          [enc setBuffer:buf_ax offset:0 atIndex:9];
          [enc setBuffer:buf_axh offset:0 atIndex:10];
          [enc setBuffer:buf_by offset:0 atIndex:11];
          [enc setBuffer:buf_byh offset:0 atIndex:12];
          [enc setBuffer:buf_bx offset:0 atIndex:13];
          [enc setBuffer:buf_bxh offset:0 atIndex:14];
          [enc setBuffer:buf_ky offset:0 atIndex:15];
          [enc setBuffer:buf_kyh offset:0 atIndex:16];
          [enc setBuffer:buf_kx offset:0 atIndex:17];
          [enc setBuffer:buf_kxh offset:0 atIndex:18];
          [enc setBuffer:buf_params offset:0 atIndex:19];
          [enc dispatchThreads:gridSize_field
              threadsPerThreadgroup:threadGroupSize_field];
          [enc endEncoding];
        }

        // ---- add_sources_ey ----
        if (n_sources_per_shot_h > 0 && buf_sources_i && buf_f) {
          size_t f_offset = (size_t)t * f_step_bytes;

          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_add_sources];
          [enc setBuffer:buf_ey offset:0 atIndex:0];
          [enc setBuffer:buf_f offset:f_offset atIndex:1];
          [enc setBuffer:buf_sources_i offset:0 atIndex:2];
          [enc setBuffer:buf_params offset:0 atIndex:3];
          [enc dispatchThreads:gridSize_src
              threadsPerThreadgroup:threadGroupSize_src];
          [enc endEncoding];
        }

        // ---- record_receivers_ey ----
        if (n_receivers_per_shot_h > 0 && buf_receivers_i && buf_r) {
          size_t r_offset = (size_t)t * r_step_bytes;

          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_record_recv];
          [enc setBuffer:buf_r offset:r_offset atIndex:0];
          [enc setBuffer:buf_ey offset:0 atIndex:1];
          [enc setBuffer:buf_receivers_i offset:0 atIndex:2];
          [enc setBuffer:buf_params offset:0 atIndex:3];
          [enc dispatchThreads:gridSize_rec
              threadsPerThreadgroup:threadGroupSize_rec];
          [enc endEncoding];
        }
      }

      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // ---- Copy mutable results back ----
    copy_back(buf_ey, ey, field_bytes);
    copy_back(buf_hx, hx, field_bytes);
    copy_back(buf_hz, hz, field_bytes);
    copy_back(buf_m_ey_x, m_ey_x, field_bytes);
    copy_back(buf_m_ey_z, m_ey_z, field_bytes);
    copy_back(buf_m_hx_z, m_hx_z, field_bytes);
    copy_back(buf_m_hz_x, m_hz_x, field_bytes);
    if (buf_r && r) {
      copy_back(buf_r, r, r_total_bytes);
    }
  }
}

// ---------------------------------------------------------------------------
// Stamp out extern "C" entry points for each stencil order (2, 4, 6, 8)
// ---------------------------------------------------------------------------

#define DEFINE_METAL_FORWARD(STENCIL, FD_PAD_VAL)                              \
  extern "C" void maxwell_tm_##STENCIL##_float_forward_mps(                    \
      float const *ca, float const *cb, float const *cq, float const *f,       \
      float *ey, float *hx, float *hz, float *m_ey_x, float *m_ey_z,           \
      float *m_hx_z, float *m_hz_x, float *r, float const *ay,                 \
      float const *by, float const *ayh, float const *byh, float const *ax,    \
      float const *bx, float const *axh, float const *bxh, float const *ky,    \
      float const *kyh, float const *kx, float const *kxh,                     \
      int64_t const *sources_i, int64_t const *receivers_i, float rdy_h,       \
      float rdx_h, float dt_h, int64_t nt, int64_t n_shots_h, int64_t ny_h,    \
      int64_t nx_h, int64_t n_sources_per_shot_h,                              \
      int64_t n_receivers_per_shot_h, int64_t step_ratio_h, bool ca_batched_h, \
      bool cb_batched_h, bool cq_batched_h, int64_t start_t, int64_t pml_y0_h, \
      int64_t pml_x0_h, int64_t pml_y1_h, int64_t pml_x1_h, int64_t n_threads, \
      int64_t device) {                                                        \
    maxwell_tm_float_forward_mps_impl(                                         \
        ca, cb, cq, f, ey, hx, hz, m_ey_x, m_ey_z, m_hx_z, m_hz_x, r, ay, by,  \
        ayh, byh, ax, bx, axh, bxh, ky, kyh, kx, kxh, sources_i, receivers_i,  \
        rdy_h, rdx_h, dt_h, nt, n_shots_h, ny_h, nx_h, n_sources_per_shot_h,   \
        n_receivers_per_shot_h, step_ratio_h, ca_batched_h, cb_batched_h,      \
        cq_batched_h, start_t, pml_y0_h, pml_x0_h, pml_y1_h, pml_x1_h,         \
        n_threads, device, FD_PAD_VAL);                                        \
  }

DEFINE_METAL_FORWARD(2, 1)
DEFINE_METAL_FORWARD(4, 2)
DEFINE_METAL_FORWARD(6, 3)
DEFINE_METAL_FORWARD(8, 4)

// ---------------------------------------------------------------------------
// Query function — lets Python check if Metal backend was compiled
// ---------------------------------------------------------------------------
extern "C" int tide_metal_available(void) { return ensure_metal() ? 1 : 0; }

// ---------------------------------------------------------------------------
// Helper: fill common BackwardParams fields
// ---------------------------------------------------------------------------
static BackwardParams make_bparams(int64_t ny, int64_t nx, int64_t n_shots,
                                   int64_t n_src, int64_t n_rec, int64_t pml_y0,
                                   int64_t pml_y1, int64_t pml_x0,
                                   int64_t pml_x1, int64_t fd_pad, float rdy,
                                   float rdx, float dt, bool ca_bat,
                                   bool cb_bat, bool cq_bat, bool ca_rg,
                                   bool cb_rg, int64_t step_ratio) {
  BackwardParams bp;
  bp.ny = ny;
  bp.nx = nx;
  bp.shot_numel = ny * nx;
  bp.n_shots = n_shots;
  bp.n_sources_per_shot = n_src;
  bp.n_receivers_per_shot = n_rec;
  bp.pml_y0 = pml_y0;
  bp.pml_y1 = pml_y1;
  bp.pml_x0 = pml_x0;
  bp.pml_x1 = pml_x1;
  bp.fd_pad = fd_pad;
  bp.rdy = rdy;
  bp.rdx = rdx;
  bp.dt = dt;
  bp.ca_batched = ca_bat;
  bp.cb_batched = cb_bat;
  bp.cq_batched = cq_bat;
  bp.ca_requires_grad = ca_rg;
  bp.cb_requires_grad = cb_rg;
  bp.step_ratio = step_ratio;
  bp.store_offset = 0;
  bp.store_ey = false;
  bp.store_curl = false;
  return bp;
}

// Helper: encode forward_kernel_h (shared between forward_with_storage and
// forward)
static void encode_forward_h(
    id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf_cq,
    id<MTLBuffer> buf_ey, id<MTLBuffer> buf_hx, id<MTLBuffer> buf_hz,
    id<MTLBuffer> buf_m_ey_x, id<MTLBuffer> buf_m_ey_z, id<MTLBuffer> buf_ay,
    id<MTLBuffer> buf_ayh, id<MTLBuffer> buf_ax, id<MTLBuffer> buf_axh,
    id<MTLBuffer> buf_by, id<MTLBuffer> buf_byh, id<MTLBuffer> buf_bx,
    id<MTLBuffer> buf_bxh, id<MTLBuffer> buf_ky, id<MTLBuffer> buf_kyh,
    id<MTLBuffer> buf_kx, id<MTLBuffer> buf_kxh, id<MTLBuffer> buf_params,
    MTLSize gridSize, MTLSize tgSize) {
  [enc setComputePipelineState:g_pso_forward_h];
  [enc setBuffer:buf_cq offset:0 atIndex:0];
  [enc setBuffer:buf_ey offset:0 atIndex:1];
  [enc setBuffer:buf_hx offset:0 atIndex:2];
  [enc setBuffer:buf_hz offset:0 atIndex:3];
  [enc setBuffer:buf_m_ey_x offset:0 atIndex:4];
  [enc setBuffer:buf_m_ey_z offset:0 atIndex:5];
  [enc setBuffer:buf_ay offset:0 atIndex:6];
  [enc setBuffer:buf_ayh offset:0 atIndex:7];
  [enc setBuffer:buf_ax offset:0 atIndex:8];
  [enc setBuffer:buf_axh offset:0 atIndex:9];
  [enc setBuffer:buf_by offset:0 atIndex:10];
  [enc setBuffer:buf_byh offset:0 atIndex:11];
  [enc setBuffer:buf_bx offset:0 atIndex:12];
  [enc setBuffer:buf_bxh offset:0 atIndex:13];
  [enc setBuffer:buf_ky offset:0 atIndex:14];
  [enc setBuffer:buf_kyh offset:0 atIndex:15];
  [enc setBuffer:buf_kx offset:0 atIndex:16];
  [enc setBuffer:buf_kxh offset:0 atIndex:17];
  [enc setBuffer:buf_params offset:0 atIndex:18];
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
  [enc endEncoding];
}

// ---------------------------------------------------------------------------
// Forward with storage implementation
// ---------------------------------------------------------------------------
static void maxwell_tm_float_forward_with_storage_mps_impl(
    float const *ca, float const *cb, float const *cq, float const *f,
    float *ey, float *hx, float *hz, float *m_ey_x, float *m_ey_z,
    float *m_hx_z, float *m_hz_x, float *r, float *ey_store_1, void *ey_store_3,
    char const *const *ey_filenames, float *curl_store_1, void *curl_store_3,
    char const *const *curl_filenames, float const *ay, float const *by,
    float const *ayh, float const *byh, float const *ax, float const *bx,
    float const *axh, float const *bxh, float const *ky, float const *kyh,
    float const *kx, float const *kxh, int64_t const *sources_i,
    int64_t const *receivers_i, float rdy_h, float rdx_h, float dt_h,
    int64_t nt, int64_t n_shots, int64_t ny, int64_t nx, int64_t n_src,
    int64_t n_rec, int64_t step_ratio, int64_t storage_mode,
    int64_t shot_bytes_uncomp, bool ca_rg, bool cb_rg, bool ca_bat, bool cb_bat,
    bool cq_bat, int64_t start_t, int64_t pml_y0, int64_t pml_x0,
    int64_t pml_y1, int64_t pml_x1, int64_t n_threads, int64_t device,
    int64_t fd_pad) {
  (void)ey_store_3;
  (void)ey_filenames;
  (void)curl_store_3;
  (void)curl_filenames;
  (void)n_threads;
  (void)device;
  (void)dt_h;

  if (!ensure_metal()) {
    fprintf(stderr, "[TIDE Metal] Metal backend not available.\n");
    return;
  }

  int64_t const shot_numel = ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  int64_t const num_steps_stored = (nt + step_ratio - 1) / step_ratio;

  // GridParams for forward_kernel_h (uses GridParams struct)
  GridParams gp;
  gp.ny = ny;
  gp.nx = nx;
  gp.shot_numel = shot_numel;
  gp.n_shots = n_shots;
  gp.n_sources_per_shot = n_src;
  gp.n_receivers_per_shot = n_rec;
  gp.pml_y0 = pml_y0;
  gp.pml_y1 = pml_y1;
  gp.pml_x0 = pml_x0;
  gp.pml_x1 = pml_x1;
  gp.fd_pad = fd_pad;
  gp.rdy = rdy_h;
  gp.rdx = rdx_h;
  gp.ca_batched = ca_bat;
  gp.cb_batched = cb_bat;
  gp.cq_batched = cq_bat;

  @autoreleasepool {
    size_t field_bytes = (size_t)n_shots * (size_t)shot_numel * sizeof(float);
    size_t model_ca =
        (ca_bat ? field_bytes : (size_t)shot_numel * sizeof(float));
    size_t model_cb =
        (cb_bat ? field_bytes : (size_t)shot_numel * sizeof(float));
    size_t model_cq =
        (cq_bat ? field_bytes : (size_t)shot_numel * sizeof(float));
    size_t pml_y_b = (size_t)ny * sizeof(float);
    size_t pml_x_b = (size_t)nx * sizeof(float);
    size_t src_count = (size_t)n_shots * (size_t)n_src;
    size_t rec_count = (size_t)n_shots * (size_t)n_rec;
    size_t f_step = src_count * sizeof(float);
    size_t r_step = rec_count * sizeof(float);
    size_t f_total = (size_t)nt * f_step;
    size_t r_total = (size_t)nt * r_step;
    // Storage: only STORAGE_DEVICE mode.
    size_t ey_store_bytes =
        (ca_rg && storage_mode == STORAGE_DEVICE)
            ? (size_t)num_steps_stored * store_size * sizeof(float)
            : 0;
    size_t curl_store_bytes =
        (cb_rg && storage_mode == STORAGE_DEVICE)
            ? (size_t)num_steps_stored * store_size * sizeof(float)
            : 0;

    // Create GPU buffers
    id<MTLBuffer> buf_ca = make_buffer(ca, model_ca);
    id<MTLBuffer> buf_cb = make_buffer(cb, model_cb);
    id<MTLBuffer> buf_cq = make_buffer(cq, model_cq);
    id<MTLBuffer> buf_ey = make_buffer(ey, field_bytes);
    id<MTLBuffer> buf_hx = make_buffer(hx, field_bytes);
    id<MTLBuffer> buf_hz = make_buffer(hz, field_bytes);
    id<MTLBuffer> buf_m_ey_x = make_buffer(m_ey_x, field_bytes);
    id<MTLBuffer> buf_m_ey_z = make_buffer(m_ey_z, field_bytes);
    id<MTLBuffer> buf_m_hx_z = make_buffer(m_hx_z, field_bytes);
    id<MTLBuffer> buf_m_hz_x = make_buffer(m_hz_x, field_bytes);
    id<MTLBuffer> buf_ay = make_buffer(ay, pml_y_b);
    id<MTLBuffer> buf_ayh = make_buffer(ayh, pml_y_b);
    id<MTLBuffer> buf_ax = make_buffer(ax, pml_x_b);
    id<MTLBuffer> buf_axh = make_buffer(axh, pml_x_b);
    id<MTLBuffer> buf_by = make_buffer(by, pml_y_b);
    id<MTLBuffer> buf_byh = make_buffer(byh, pml_y_b);
    id<MTLBuffer> buf_bx = make_buffer(bx, pml_x_b);
    id<MTLBuffer> buf_bxh = make_buffer(bxh, pml_x_b);
    id<MTLBuffer> buf_ky = make_buffer(ky, pml_y_b);
    id<MTLBuffer> buf_kyh = make_buffer(kyh, pml_y_b);
    id<MTLBuffer> buf_kx = make_buffer(kx, pml_x_b);
    id<MTLBuffer> buf_kxh = make_buffer(kxh, pml_x_b);
    id<MTLBuffer> buf_src_i =
        (n_src > 0) ? make_buffer(sources_i, src_count * sizeof(int64_t)) : nil;
    id<MTLBuffer> buf_rec_i =
        (n_rec > 0) ? make_buffer(receivers_i, rec_count * sizeof(int64_t))
                    : nil;
    id<MTLBuffer> buf_f =
        (n_src > 0 && f_total > 0) ? make_buffer(f, f_total) : nil;
    id<MTLBuffer> buf_r =
        (n_rec > 0 && r_total > 0) ? make_buffer(r, r_total) : nil;
    // Storage buffers (on GPU)
    id<MTLBuffer> buf_ey_store =
        (ey_store_bytes > 0) ? make_zero_buffer(ey_store_bytes) : nil;
    id<MTLBuffer> buf_curl_store =
        (curl_store_bytes > 0) ? make_zero_buffer(curl_store_bytes) : nil;

    id<MTLBuffer> buf_gp =
        [g_device newBufferWithBytes:&gp
                              length:sizeof(GridParams)
                             options:MTLResourceStorageModeShared];

    int64_t grid_x = nx - 2 * fd_pad + 2;
    if (grid_x <= 0)
      grid_x = 1;
    int64_t grid_y = ny - 2 * fd_pad + 2;
    if (grid_y <= 0)
      grid_y = 1;
    MTLSize gridSize_field = MTLSizeMake((NSUInteger)grid_x, (NSUInteger)grid_y,
                                         (NSUInteger)n_shots);
    NSUInteger tw = g_pso_forward_h.threadExecutionWidth;
    NSUInteger th_max = g_pso_forward_h.maxTotalThreadsPerThreadgroup;
    MTLSize tgSize_field =
        MTLSizeMake(tw, th_max / tw > 0 ? th_max / tw : 1, 1);
    MTLSize gridSize_src =
        MTLSizeMake((NSUInteger)n_src, (NSUInteger)n_shots, 1);
    MTLSize tgSize_src =
        MTLSizeMake(MIN((NSUInteger)MAX(n_src, 1), (NSUInteger)64), 1, 1);
    MTLSize gridSize_rec =
        MTLSizeMake((NSUInteger)n_rec, (NSUInteger)n_shots, 1);
    MTLSize tgSize_rec =
        MTLSizeMake(MIN((NSUInteger)MAX(n_rec, 1), (NSUInteger)64), 1, 1);

    int64_t batch = MIN(nt, (int64_t)256);
    for (int64_t t_start = 0; t_start < nt; t_start += batch) {
      int64_t t_end = MIN(t_start + batch, nt);
      id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];

      for (int64_t t_local = t_start; t_local < t_end; ++t_local) {
        int64_t t = start_t + t_local;

        // forward_kernel_h
        {
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          encode_forward_h(enc, buf_cq, buf_ey, buf_hx, buf_hz, buf_m_ey_x,
                           buf_m_ey_z, buf_ay, buf_ayh, buf_ax, buf_axh, buf_by,
                           buf_byh, buf_bx, buf_bxh, buf_ky, buf_kyh, buf_kx,
                           buf_kxh, buf_gp, gridSize_field, tgSize_field);
        }

        // forward_kernel_e_with_storage
        {
          bool store_step = (t % step_ratio) == 0;
          bool s_ey = store_step && ca_rg;
          bool s_curl = store_step && cb_rg;
          int64_t step_idx = t / step_ratio;
          int64_t s_off =
              (storage_mode == STORAGE_DEVICE) ? step_idx * store_size : 0;

          BackwardParams bp =
              make_bparams(ny, nx, n_shots, n_src, n_rec, pml_y0, pml_y1,
                           pml_x0, pml_x1, fd_pad, rdy_h, rdx_h, dt_h, ca_bat,
                           cb_bat, cq_bat, ca_rg, cb_rg, step_ratio);
          bp.store_offset = s_off;
          bp.store_ey = s_ey;
          bp.store_curl = s_curl;

          id<MTLBuffer> buf_bp =
              [g_device newBufferWithBytes:&bp
                                    length:sizeof(BackwardParams)
                                   options:MTLResourceStorageModeShared];
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_forward_e_storage];
          [enc setBuffer:buf_ca offset:0 atIndex:0];
          [enc setBuffer:buf_cb offset:0 atIndex:1];
          [enc setBuffer:buf_hx offset:0 atIndex:2];
          [enc setBuffer:buf_hz offset:0 atIndex:3];
          [enc setBuffer:buf_ey offset:0 atIndex:4];
          [enc setBuffer:buf_m_hx_z offset:0 atIndex:5];
          [enc setBuffer:buf_m_hz_x offset:0 atIndex:6];
          [enc setBuffer:buf_ay offset:0 atIndex:7];
          [enc setBuffer:buf_ayh offset:0 atIndex:8];
          [enc setBuffer:buf_ax offset:0 atIndex:9];
          [enc setBuffer:buf_axh offset:0 atIndex:10];
          [enc setBuffer:buf_by offset:0 atIndex:11];
          [enc setBuffer:buf_byh offset:0 atIndex:12];
          [enc setBuffer:buf_bx offset:0 atIndex:13];
          [enc setBuffer:buf_bxh offset:0 atIndex:14];
          [enc setBuffer:buf_ky offset:0 atIndex:15];
          [enc setBuffer:buf_kyh offset:0 atIndex:16];
          [enc setBuffer:buf_kx offset:0 atIndex:17];
          [enc setBuffer:buf_kxh offset:0 atIndex:18];
          [enc setBuffer:(buf_ey_store ? buf_ey_store : buf_ey)
                  offset:0
                 atIndex:19];
          [enc setBuffer:(buf_curl_store ? buf_curl_store : buf_ey)
                  offset:0
                 atIndex:20];
          [enc setBuffer:buf_bp offset:0 atIndex:21];
          [enc dispatchThreads:gridSize_field
              threadsPerThreadgroup:tgSize_field];
          [enc endEncoding];
        }

        // add_sources_ey
        if (n_src > 0 && buf_src_i && buf_f) {
          size_t f_off = (size_t)t * f_step;
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_add_sources];
          [enc setBuffer:buf_ey offset:0 atIndex:0];
          [enc setBuffer:buf_f offset:f_off atIndex:1];
          [enc setBuffer:buf_src_i offset:0 atIndex:2];
          [enc setBuffer:buf_gp offset:0 atIndex:3];
          [enc dispatchThreads:gridSize_src threadsPerThreadgroup:tgSize_src];
          [enc endEncoding];
        }

        // record_receivers_ey
        if (n_rec > 0 && buf_rec_i && buf_r) {
          size_t r_off = (size_t)t * r_step;
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_record_recv];
          [enc setBuffer:buf_r offset:r_off atIndex:0];
          [enc setBuffer:buf_ey offset:0 atIndex:1];
          [enc setBuffer:buf_rec_i offset:0 atIndex:2];
          [enc setBuffer:buf_gp offset:0 atIndex:3];
          [enc dispatchThreads:gridSize_rec threadsPerThreadgroup:tgSize_rec];
          [enc endEncoding];
        }
      }
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // Copy results back
    copy_back(buf_ey, ey, field_bytes);
    copy_back(buf_hx, hx, field_bytes);
    copy_back(buf_hz, hz, field_bytes);
    copy_back(buf_m_ey_x, m_ey_x, field_bytes);
    copy_back(buf_m_ey_z, m_ey_z, field_bytes);
    copy_back(buf_m_hx_z, m_hx_z, field_bytes);
    copy_back(buf_m_hz_x, m_hz_x, field_bytes);
    if (buf_r && r)
      copy_back(buf_r, r, r_total);
    // Copy storage back to host
    if (buf_ey_store && ey_store_1)
      copy_back(buf_ey_store, ey_store_1, ey_store_bytes);
    if (buf_curl_store && curl_store_1)
      copy_back(buf_curl_store, curl_store_1, curl_store_bytes);
  }
}

// ---------------------------------------------------------------------------
// Backward propagation implementation
// ---------------------------------------------------------------------------
static void maxwell_tm_float_backward_mps_impl(
    float const *ca, float const *cb, float const *cq, float const *grad_r,
    float *lambda_ey, float *lambda_hx, float *lambda_hz, float *m_lambda_ey_x,
    float *m_lambda_ey_z, float *m_lambda_hx_z, float *m_lambda_hz_x,
    float *ey_store_1, void *ey_store_3, char const *const *ey_filenames,
    float *curl_store_1, void *curl_store_3, char const *const *curl_filenames,
    float *grad_f, float *grad_ca, float *grad_cb, float *grad_eps,
    float *grad_sigma, float *grad_ca_shot, float *grad_cb_shot,
    float const *ay, float const *by, float const *ayh, float const *byh,
    float const *ax, float const *bx, float const *axh, float const *bxh,
    float const *ky, float const *kyh, float const *kx, float const *kxh,
    int64_t const *sources_i, int64_t const *receivers_i, float rdy_h,
    float rdx_h, float dt_h, int64_t nt, int64_t n_shots, int64_t ny,
    int64_t nx, int64_t n_src, int64_t n_rec, int64_t step_ratio,
    int64_t storage_mode, int64_t shot_bytes_uncomp, bool ca_rg, bool cb_rg,
    bool ca_bat, bool cb_bat, bool cq_bat, int64_t start_t, int64_t pml_y0,
    int64_t pml_x0, int64_t pml_y1, int64_t pml_x1, int64_t n_threads,
    int64_t device, int64_t fd_pad) {
  (void)ey_store_3;
  (void)ey_filenames;
  (void)curl_store_3;
  (void)curl_filenames;
  (void)grad_ca_shot;
  (void)grad_cb_shot;
  (void)n_threads;
  (void)device;

  if (!ensure_metal()) {
    fprintf(stderr, "[TIDE Metal] Metal backend not available.\n");
    return;
  }

  int64_t const shot_numel = ny * nx;
  int64_t const store_size = n_shots * shot_numel;

  // GridParams for source/receiver kernels
  GridParams gp;
  gp.ny = ny;
  gp.nx = nx;
  gp.shot_numel = shot_numel;
  gp.n_shots = n_shots;
  gp.n_sources_per_shot = n_src;
  gp.n_receivers_per_shot = n_rec;
  gp.pml_y0 = pml_y0;
  gp.pml_y1 = pml_y1;
  gp.pml_x0 = pml_x0;
  gp.pml_x1 = pml_x1;
  gp.fd_pad = fd_pad;
  gp.rdy = rdy_h;
  gp.rdx = rdx_h;
  gp.ca_batched = ca_bat;
  gp.cb_batched = cb_bat;
  gp.cq_batched = cq_bat;

  @autoreleasepool {
    size_t field_bytes = (size_t)n_shots * (size_t)shot_numel * sizeof(float);
    size_t model_ca =
        (ca_bat ? field_bytes : (size_t)shot_numel * sizeof(float));
    size_t model_cb =
        (cb_bat ? field_bytes : (size_t)shot_numel * sizeof(float));
    size_t model_cq =
        (cq_bat ? field_bytes : (size_t)shot_numel * sizeof(float));
    size_t pml_y_b = (size_t)ny * sizeof(float);
    size_t pml_x_b = (size_t)nx * sizeof(float);
    size_t src_count = (size_t)n_shots * (size_t)n_src;
    size_t rec_count = (size_t)n_shots * (size_t)n_rec;
    size_t grad_f_step = src_count * sizeof(float);
    size_t grad_r_step = rec_count * sizeof(float);
    size_t grad_f_total = (size_t)nt * grad_f_step;
    size_t grad_r_total = (size_t)nt * grad_r_step;
    size_t grad_model_bytes =
        (ca_bat ? field_bytes : (size_t)shot_numel * sizeof(float));
    int64_t num_steps_stored = (nt + step_ratio - 1) / step_ratio;
    size_t ey_store_bytes =
        (ca_rg && storage_mode == STORAGE_DEVICE)
            ? (size_t)num_steps_stored * store_size * sizeof(float)
            : 0;
    size_t curl_store_bytes =
        (cb_rg && storage_mode == STORAGE_DEVICE)
            ? (size_t)num_steps_stored * store_size * sizeof(float)
            : 0;

    // Create GPU buffers
    id<MTLBuffer> buf_ca = make_buffer(ca, model_ca);
    id<MTLBuffer> buf_cb = make_buffer(cb, model_cb);
    id<MTLBuffer> buf_cq = make_buffer(cq, model_cq);
    id<MTLBuffer> buf_lambda_ey = make_buffer(lambda_ey, field_bytes);
    id<MTLBuffer> buf_lambda_hx = make_buffer(lambda_hx, field_bytes);
    id<MTLBuffer> buf_lambda_hz = make_buffer(lambda_hz, field_bytes);
    id<MTLBuffer> buf_m_ley_x = make_buffer(m_lambda_ey_x, field_bytes);
    id<MTLBuffer> buf_m_ley_z = make_buffer(m_lambda_ey_z, field_bytes);
    id<MTLBuffer> buf_m_lhx_z = make_buffer(m_lambda_hx_z, field_bytes);
    id<MTLBuffer> buf_m_lhz_x = make_buffer(m_lambda_hz_x, field_bytes);
    id<MTLBuffer> buf_ay = make_buffer(ay, pml_y_b);
    id<MTLBuffer> buf_ayh = make_buffer(ayh, pml_y_b);
    id<MTLBuffer> buf_ax = make_buffer(ax, pml_x_b);
    id<MTLBuffer> buf_axh = make_buffer(axh, pml_x_b);
    id<MTLBuffer> buf_by = make_buffer(by, pml_y_b);
    id<MTLBuffer> buf_byh = make_buffer(byh, pml_y_b);
    id<MTLBuffer> buf_bx = make_buffer(bx, pml_x_b);
    id<MTLBuffer> buf_bxh = make_buffer(bxh, pml_x_b);
    id<MTLBuffer> buf_ky = make_buffer(ky, pml_y_b);
    id<MTLBuffer> buf_kyh = make_buffer(kyh, pml_y_b);
    id<MTLBuffer> buf_kx = make_buffer(kx, pml_x_b);
    id<MTLBuffer> buf_kxh = make_buffer(kxh, pml_x_b);
    id<MTLBuffer> buf_src_i =
        (n_src > 0) ? make_buffer(sources_i, src_count * sizeof(int64_t)) : nil;
    id<MTLBuffer> buf_rec_i =
        (n_rec > 0) ? make_buffer(receivers_i, rec_count * sizeof(int64_t))
                    : nil;
    id<MTLBuffer> buf_grad_r = (n_rec > 0 && grad_r_total > 0)
                                   ? make_buffer(grad_r, grad_r_total)
                                   : nil;
    id<MTLBuffer> buf_grad_f = (n_src > 0 && grad_f_total > 0)
                                   ? make_buffer(grad_f, grad_f_total)
                                   : nil;
    id<MTLBuffer> buf_grad_ca =
        (ca_rg && grad_ca) ? make_buffer(grad_ca, grad_model_bytes) : nil;
    id<MTLBuffer> buf_grad_cb =
        (cb_rg && grad_cb) ? make_buffer(grad_cb, grad_model_bytes) : nil;
    id<MTLBuffer> buf_ey_store =
        (ey_store_bytes > 0) ? make_buffer(ey_store_1, ey_store_bytes) : nil;
    id<MTLBuffer> buf_curl_store =
        (curl_store_bytes > 0) ? make_buffer(curl_store_1, curl_store_bytes)
                               : nil;

    id<MTLBuffer> buf_gp =
        [g_device newBufferWithBytes:&gp
                              length:sizeof(GridParams)
                             options:MTLResourceStorageModeShared];

    int64_t grid_x = nx - 2 * fd_pad + 2;
    if (grid_x <= 0)
      grid_x = 1;
    int64_t grid_y = ny - 2 * fd_pad + 2;
    if (grid_y <= 0)
      grid_y = 1;
    MTLSize gridSize_field = MTLSizeMake((NSUInteger)grid_x, (NSUInteger)grid_y,
                                         (NSUInteger)n_shots);
    NSUInteger tw = g_pso_backward_h.threadExecutionWidth;
    NSUInteger th_max = g_pso_backward_h.maxTotalThreadsPerThreadgroup;
    MTLSize tgSize_field =
        MTLSizeMake(tw, th_max / tw > 0 ? th_max / tw : 1, 1);
    MTLSize gridSize_src =
        MTLSizeMake((NSUInteger)n_src, (NSUInteger)n_shots, 1);
    MTLSize tgSize_src =
        MTLSizeMake(MIN((NSUInteger)MAX(n_src, 1), (NSUInteger)64), 1, 1);
    MTLSize gridSize_rec =
        MTLSizeMake((NSUInteger)n_rec, (NSUInteger)n_shots, 1);
    MTLSize tgSize_rec =
        MTLSizeMake(MIN((NSUInteger)MAX(n_rec, 1), (NSUInteger)64), 1, 1);

    // Time-reversed loop
    int64_t batch = MIN(nt, (int64_t)64);
    for (int64_t t_start_rev = 0; t_start_rev < nt; t_start_rev += batch) {
      int64_t t_end_rev = MIN(t_start_rev + batch, nt);
      id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];

      for (int64_t t_idx = t_start_rev; t_idx < t_end_rev; ++t_idx) {
        int64_t t = start_t - 1 - t_idx;
        int64_t store_idx = t / step_ratio;
        bool do_grad = (t % step_ratio) == 0;
        int64_t s_off =
            (storage_mode == STORAGE_DEVICE) ? store_idx * store_size : 0;

        // Inject adjoint residuals (adjoint of receiver recording)
        if (n_rec > 0 && buf_rec_i && buf_grad_r) {
          size_t gr_off = (size_t)t * grad_r_step;
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_add_sources];
          [enc setBuffer:buf_lambda_ey offset:0 atIndex:0];
          [enc setBuffer:buf_grad_r offset:gr_off atIndex:1];
          [enc setBuffer:buf_rec_i offset:0 atIndex:2];
          // Need GridParams with n_sources = n_rec for this kernel
          GridParams gp_rec = gp;
          gp_rec.n_sources_per_shot = n_rec;
          id<MTLBuffer> buf_gp_rec =
              [g_device newBufferWithBytes:&gp_rec
                                    length:sizeof(GridParams)
                                   options:MTLResourceStorageModeShared];
          [enc setBuffer:buf_gp_rec offset:0 atIndex:3];
          [enc dispatchThreads:gridSize_rec threadsPerThreadgroup:tgSize_rec];
          [enc endEncoding];
        }

        // Record adjoint source gradient (adjoint of source injection)
        if (n_src > 0 && buf_src_i && buf_grad_f) {
          size_t gf_off = (size_t)t * grad_f_step;
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_record_recv];
          [enc setBuffer:buf_grad_f offset:gf_off atIndex:0];
          [enc setBuffer:buf_lambda_ey offset:0 atIndex:1];
          [enc setBuffer:buf_src_i offset:0 atIndex:2];
          GridParams gp_src = gp;
          gp_src.n_receivers_per_shot = n_src;
          id<MTLBuffer> buf_gp_src =
              [g_device newBufferWithBytes:&gp_src
                                    length:sizeof(GridParams)
                                   options:MTLResourceStorageModeShared];
          [enc setBuffer:buf_gp_src offset:0 atIndex:3];
          [enc dispatchThreads:gridSize_src threadsPerThreadgroup:tgSize_src];
          [enc endEncoding];
        }

        // backward_kernel_lambda_e_with_grad
        {
          BackwardParams bp =
              make_bparams(ny, nx, n_shots, n_src, n_rec, pml_y0, pml_y1,
                           pml_x0, pml_x1, fd_pad, rdy_h, rdx_h, dt_h, ca_bat,
                           cb_bat, cq_bat, ca_rg, cb_rg, step_ratio);
          bp.store_offset = s_off;
          bp.store_ey = do_grad && ca_rg;
          bp.store_curl = do_grad && cb_rg;
          id<MTLBuffer> buf_bp =
              [g_device newBufferWithBytes:&bp
                                    length:sizeof(BackwardParams)
                                   options:MTLResourceStorageModeShared];
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_backward_e_grad];
          [enc setBuffer:buf_ca offset:0 atIndex:0];
          [enc setBuffer:buf_cq offset:0 atIndex:1];
          [enc setBuffer:buf_lambda_hx offset:0 atIndex:2];
          [enc setBuffer:buf_lambda_hz offset:0 atIndex:3];
          [enc setBuffer:buf_lambda_ey offset:0 atIndex:4];
          [enc setBuffer:buf_m_lhx_z offset:0 atIndex:5];
          [enc setBuffer:buf_m_lhz_x offset:0 atIndex:6];
          [enc setBuffer:(buf_ey_store ? buf_ey_store : buf_lambda_ey)
                  offset:0
                 atIndex:7];
          [enc setBuffer:(buf_curl_store ? buf_curl_store : buf_lambda_ey)
                  offset:0
                 atIndex:8];
          [enc setBuffer:(buf_grad_ca ? buf_grad_ca : buf_lambda_ey)
                  offset:0
                 atIndex:9];
          [enc setBuffer:(buf_grad_cb ? buf_grad_cb : buf_lambda_ey)
                  offset:0
                 atIndex:10];
          [enc setBuffer:buf_ay offset:0 atIndex:11];
          [enc setBuffer:buf_ayh offset:0 atIndex:12];
          [enc setBuffer:buf_ax offset:0 atIndex:13];
          [enc setBuffer:buf_axh offset:0 atIndex:14];
          [enc setBuffer:buf_by offset:0 atIndex:15];
          [enc setBuffer:buf_byh offset:0 atIndex:16];
          [enc setBuffer:buf_bx offset:0 atIndex:17];
          [enc setBuffer:buf_bxh offset:0 atIndex:18];
          [enc setBuffer:buf_ky offset:0 atIndex:19];
          [enc setBuffer:buf_kyh offset:0 atIndex:20];
          [enc setBuffer:buf_kx offset:0 atIndex:21];
          [enc setBuffer:buf_kxh offset:0 atIndex:22];
          [enc setBuffer:buf_bp offset:0 atIndex:23];
          [enc dispatchThreads:gridSize_field
              threadsPerThreadgroup:tgSize_field];
          [enc endEncoding];
        }

        // backward_kernel_lambda_h
        {
          BackwardParams bp =
              make_bparams(ny, nx, n_shots, n_src, n_rec, pml_y0, pml_y1,
                           pml_x0, pml_x1, fd_pad, rdy_h, rdx_h, dt_h, ca_bat,
                           cb_bat, cq_bat, ca_rg, cb_rg, step_ratio);
          id<MTLBuffer> buf_bp =
              [g_device newBufferWithBytes:&bp
                                    length:sizeof(BackwardParams)
                                   options:MTLResourceStorageModeShared];
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:g_pso_backward_h];
          [enc setBuffer:buf_cb offset:0 atIndex:0];
          [enc setBuffer:buf_lambda_ey offset:0 atIndex:1];
          [enc setBuffer:buf_lambda_hx offset:0 atIndex:2];
          [enc setBuffer:buf_lambda_hz offset:0 atIndex:3];
          [enc setBuffer:buf_m_ley_x offset:0 atIndex:4];
          [enc setBuffer:buf_m_ley_z offset:0 atIndex:5];
          [enc setBuffer:buf_ay offset:0 atIndex:6];
          [enc setBuffer:buf_ayh offset:0 atIndex:7];
          [enc setBuffer:buf_ax offset:0 atIndex:8];
          [enc setBuffer:buf_axh offset:0 atIndex:9];
          [enc setBuffer:buf_by offset:0 atIndex:10];
          [enc setBuffer:buf_byh offset:0 atIndex:11];
          [enc setBuffer:buf_bx offset:0 atIndex:12];
          [enc setBuffer:buf_bxh offset:0 atIndex:13];
          [enc setBuffer:buf_ky offset:0 atIndex:14];
          [enc setBuffer:buf_kyh offset:0 atIndex:15];
          [enc setBuffer:buf_kx offset:0 atIndex:16];
          [enc setBuffer:buf_kxh offset:0 atIndex:17];
          [enc setBuffer:buf_bp offset:0 atIndex:18];
          [enc dispatchThreads:gridSize_field
              threadsPerThreadgroup:tgSize_field];
          [enc endEncoding];
        }
      }
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // convert_grad_ca_cb_to_eps_sigma on GPU
    if ((ca_rg || cb_rg) && (grad_eps || grad_sigma)) {
      id<MTLBuffer> buf_grad_eps =
          grad_eps ? make_zero_buffer(grad_model_bytes) : nil;
      id<MTLBuffer> buf_grad_sigma =
          grad_sigma ? make_zero_buffer(grad_model_bytes) : nil;
      BackwardParams bp = make_bparams(
          ny, nx, n_shots, n_src, n_rec, pml_y0, pml_y1, pml_x0, pml_x1, fd_pad,
          rdy_h, rdx_h, dt_h, ca_bat, cb_bat, cq_bat, ca_rg, cb_rg, step_ratio);
      id<MTLBuffer> buf_bp =
          [g_device newBufferWithBytes:&bp
                                length:sizeof(BackwardParams)
                               options:MTLResourceStorageModeShared];
      int64_t out_shots = ca_bat ? n_shots : 1;
      MTLSize gridSize_grad =
          MTLSizeMake((NSUInteger)nx, (NSUInteger)ny, (NSUInteger)out_shots);
      id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:g_pso_convert_grad];
      [enc setBuffer:buf_ca offset:0 atIndex:0];
      [enc setBuffer:buf_cb offset:0 atIndex:1];
      [enc setBuffer:(buf_grad_ca ? buf_grad_ca : buf_ca) offset:0 atIndex:2];
      [enc setBuffer:(buf_grad_cb ? buf_grad_cb : buf_ca) offset:0 atIndex:3];
      [enc setBuffer:(buf_grad_eps ? buf_grad_eps : buf_ca) offset:0 atIndex:4];
      [enc setBuffer:(buf_grad_sigma ? buf_grad_sigma : buf_ca)
              offset:0
             atIndex:5];
      [enc setBuffer:buf_bp offset:0 atIndex:6];
      [enc dispatchThreads:gridSize_grad threadsPerThreadgroup:tgSize_field];
      [enc endEncoding];
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
      if (buf_grad_eps && grad_eps)
        copy_back(buf_grad_eps, grad_eps, grad_model_bytes);
      if (buf_grad_sigma && grad_sigma)
        copy_back(buf_grad_sigma, grad_sigma, grad_model_bytes);
    }

    // Copy results back
    copy_back(buf_lambda_ey, lambda_ey, field_bytes);
    copy_back(buf_lambda_hx, lambda_hx, field_bytes);
    copy_back(buf_lambda_hz, lambda_hz, field_bytes);
    copy_back(buf_m_ley_x, m_lambda_ey_x, field_bytes);
    copy_back(buf_m_ley_z, m_lambda_ey_z, field_bytes);
    copy_back(buf_m_lhx_z, m_lambda_hx_z, field_bytes);
    copy_back(buf_m_lhz_x, m_lambda_hz_x, field_bytes);
    if (buf_grad_ca && grad_ca)
      copy_back(buf_grad_ca, grad_ca, grad_model_bytes);
    if (buf_grad_cb && grad_cb)
      copy_back(buf_grad_cb, grad_cb, grad_model_bytes);
    if (buf_grad_f && grad_f)
      copy_back(buf_grad_f, grad_f, grad_f_total);
  }
}

// ---------------------------------------------------------------------------
// Extern C entry points for all stencil orders
// ---------------------------------------------------------------------------

#define DEFINE_METAL_FORWARD_STORAGE(STENCIL, FD_PAD_VAL)                      \
  extern "C" void maxwell_tm_##STENCIL##_float_forward_with_storage_mps(       \
      float const *ca, float const *cb, float const *cq, float const *f,       \
      float *ey, float *hx, float *hz, float *m_ey_x, float *m_ey_z,           \
      float *m_hx_z, float *m_hz_x, float *r, float *ey_store_1,               \
      void *ey_store_3, char const *const *ey_fn, float *curl_store_1,         \
      void *curl_store_3, char const *const *curl_fn, float const *ay,         \
      float const *by, float const *ayh, float const *byh, float const *ax,    \
      float const *bx, float const *axh, float const *bxh, float const *ky,    \
      float const *kyh, float const *kx, float const *kxh, int64_t const *si,  \
      int64_t const *ri, float rdy, float rdx, float dt, int64_t nt,           \
      int64_t ns, int64_t ny, int64_t nx, int64_t nsrc, int64_t nrec,          \
      int64_t sr, int64_t sm, int64_t sbu, bool carg, bool cbrg, bool cab,     \
      bool cbb, bool cqb, int64_t st, int64_t py0, int64_t px0, int64_t py1,   \
      int64_t px1, int64_t nth, int64_t dev) {                                 \
    maxwell_tm_float_forward_with_storage_mps_impl(                            \
        ca, cb, cq, f, ey, hx, hz, m_ey_x, m_ey_z, m_hx_z, m_hz_x, r,          \
        ey_store_1, ey_store_3, ey_fn, curl_store_1, curl_store_3, curl_fn,    \
        ay, by, ayh, byh, ax, bx, axh, bxh, ky, kyh, kx, kxh, si, ri, rdy,     \
        rdx, dt, nt, ns, ny, nx, nsrc, nrec, sr, sm, sbu, carg, cbrg, cab,     \
        cbb, cqb, st, py0, px0, py1, px1, nth, dev, FD_PAD_VAL);               \
  }

#define DEFINE_METAL_BACKWARD(STENCIL, FD_PAD_VAL)                             \
  extern "C" void maxwell_tm_##STENCIL##_float_backward_mps(                   \
      float const *ca, float const *cb, float const *cq, float const *gr,      \
      float *ley, float *lhx, float *lhz, float *mlex, float *mlez,            \
      float *mlhxz, float *mlhzx, float *eys1, void *eys3,                     \
      char const *const *eyfn, float *cs1, void *cs3, char const *const *cfn,  \
      float *gf, float *gca, float *gcb, float *ge, float *gs, float *gcas,    \
      float *gcbs, float const *ay, float const *by, float const *ayh,         \
      float const *byh, float const *ax, float const *bx, float const *axh,    \
      float const *bxh, float const *ky, float const *kyh, float const *kx,    \
      float const *kxh, int64_t const *si, int64_t const *ri, float rdy,       \
      float rdx, float dt, int64_t nt, int64_t ns, int64_t ny, int64_t nx,     \
      int64_t nsrc, int64_t nrec, int64_t sr, int64_t sm, int64_t sbu,         \
      bool carg, bool cbrg, bool cab, bool cbb, bool cqb, int64_t st,          \
      int64_t py0, int64_t px0, int64_t py1, int64_t px1, int64_t nth,         \
      int64_t dev) {                                                           \
    maxwell_tm_float_backward_mps_impl(                                        \
        ca, cb, cq, gr, ley, lhx, lhz, mlex, mlez, mlhxz, mlhzx, eys1, eys3,   \
        eyfn, cs1, cs3, cfn, gf, gca, gcb, ge, gs, gcas, gcbs, ay, by, ayh,    \
        byh, ax, bx, axh, bxh, ky, kyh, kx, kxh, si, ri, rdy, rdx, dt, nt, ns, \
        ny, nx, nsrc, nrec, sr, sm, sbu, carg, cbrg, cab, cbb, cqb, st, py0,   \
        px0, py1, px1, nth, dev, FD_PAD_VAL);                                  \
  }

DEFINE_METAL_FORWARD_STORAGE(2, 1)
DEFINE_METAL_FORWARD_STORAGE(4, 2)
DEFINE_METAL_FORWARD_STORAGE(6, 3)
DEFINE_METAL_FORWARD_STORAGE(8, 4)

DEFINE_METAL_BACKWARD(2, 1)
DEFINE_METAL_BACKWARD(4, 2)
DEFINE_METAL_BACKWARD(6, 3)
DEFINE_METAL_BACKWARD(8, 4)
