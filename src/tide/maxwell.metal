/*
 * Maxwell wave equation propagator (Metal implementation)
 *
 * This file contains the Metal compute shader implementation of the 2D TM
 * Maxwell equations propagator for Apple GPU (MPS) acceleration.
 *
 * Forward-only kernels:
 *   - forward_kernel_h: Update H fields (Hx, Hz) with CPML
 *   - forward_kernel_e: Update E field (Ey) with CPML
 *   - add_sources_ey: Inject source into Ey field
 *   - record_receivers_ey: Record Ey at receiver locations
 */

#include <metal_stdlib>
using namespace metal;

// Grid parameters passed as a constant buffer
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

// 2D indexing
#define ND_INDEX(base, dy, dx) ((base) + (dy) * params.nx + (dx))


// ============================================================================
// Staggered-grid finite-difference stencil operators
// ============================================================================

// --- DIFFY1: d/dy at integer grid points ---
inline float diffy1_2(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (f[i] - f[i - nx]) * rdy;
}
inline float diffy1_4(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (9.0f/8.0f * (f[i] - f[i - nx])
          - 1.0f/24.0f * (f[i + nx] - f[i - 2*nx])) * rdy;
}
inline float diffy1_6(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (75.0f/64.0f * (f[i] - f[i - nx])
          - 25.0f/384.0f * (f[i + nx] - f[i - 2*nx])
          + 3.0f/640.0f * (f[i + 2*nx] - f[i - 3*nx])) * rdy;
}
inline float diffy1_8(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (1225.0f/1024.0f * (f[i] - f[i - nx])
          - 245.0f/3072.0f * (f[i + nx] - f[i - 2*nx])
          + 49.0f/5120.0f * (f[i + 2*nx] - f[i - 3*nx])
          - 5.0f/7168.0f * (f[i + 3*nx] - f[i - 4*nx])) * rdy;
}

// --- DIFFX1: d/dx at integer grid points ---
inline float diffx1_2(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (f[i] - f[i - 1]) * rdx;
}
inline float diffx1_4(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (9.0f/8.0f * (f[i] - f[i - 1])
          - 1.0f/24.0f * (f[i + 1] - f[i - 2])) * rdx;
}
inline float diffx1_6(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (75.0f/64.0f * (f[i] - f[i - 1])
          - 25.0f/384.0f * (f[i + 1] - f[i - 2])
          + 3.0f/640.0f * (f[i + 2] - f[i - 3])) * rdx;
}
inline float diffx1_8(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (1225.0f/1024.0f * (f[i] - f[i - 1])
          - 245.0f/3072.0f * (f[i + 1] - f[i - 2])
          + 49.0f/5120.0f * (f[i + 2] - f[i - 3])
          - 5.0f/7168.0f * (f[i + 3] - f[i - 4])) * rdx;
}

// --- DIFFYH1: d/dy at half grid points ---
inline float diffyh1_2(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (f[i + nx] - f[i]) * rdy;
}
inline float diffyh1_4(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (9.0f/8.0f * (f[i + nx] - f[i])
          - 1.0f/24.0f * (f[i + 2*nx] - f[i - nx])) * rdy;
}
inline float diffyh1_6(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (75.0f/64.0f * (f[i + nx] - f[i])
          - 25.0f/384.0f * (f[i + 2*nx] - f[i - nx])
          + 3.0f/640.0f * (f[i + 3*nx] - f[i - 2*nx])) * rdy;
}
inline float diffyh1_8(device const float* f, int64_t i, int64_t nx, float rdy) {
    return (1225.0f/1024.0f * (f[i + nx] - f[i])
          - 245.0f/3072.0f * (f[i + 2*nx] - f[i - nx])
          + 49.0f/5120.0f * (f[i + 3*nx] - f[i - 2*nx])
          - 5.0f/7168.0f * (f[i + 4*nx] - f[i - 3*nx])) * rdy;
}

// --- DIFFXH1: d/dx at half grid points ---
inline float diffxh1_2(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (f[i + 1] - f[i]) * rdx;
}
inline float diffxh1_4(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (9.0f/8.0f * (f[i + 1] - f[i])
          - 1.0f/24.0f * (f[i + 2] - f[i - 1])) * rdx;
}
inline float diffxh1_6(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (75.0f/64.0f * (f[i + 1] - f[i])
          - 25.0f/384.0f * (f[i + 2] - f[i - 1])
          + 3.0f/640.0f * (f[i + 3] - f[i - 2])) * rdx;
}
inline float diffxh1_8(device const float* f, int64_t i, int64_t nx, float rdx) {
    return (1225.0f/1024.0f * (f[i + 1] - f[i])
          - 245.0f/3072.0f * (f[i + 2] - f[i - 1])
          + 49.0f/5120.0f * (f[i + 3] - f[i - 2])
          - 5.0f/7168.0f * (f[i + 4] - f[i - 3])) * rdx;
}

// Generic dispatchers based on fd_pad (fd_pad = stencil / 2)
inline float diffy1(device const float* f, int64_t i, int64_t nx, float rdy, int64_t fd_pad) {
    switch (fd_pad) {
        case 1: return diffy1_2(f, i, nx, rdy);
        case 2: return diffy1_4(f, i, nx, rdy);
        case 3: return diffy1_6(f, i, nx, rdy);
        default: return diffy1_8(f, i, nx, rdy);
    }
}
inline float diffx1(device const float* f, int64_t i, int64_t nx, float rdx, int64_t fd_pad) {
    switch (fd_pad) {
        case 1: return diffx1_2(f, i, nx, rdx);
        case 2: return diffx1_4(f, i, nx, rdx);
        case 3: return diffx1_6(f, i, nx, rdx);
        default: return diffx1_8(f, i, nx, rdx);
    }
}
inline float diffyh1(device const float* f, int64_t i, int64_t nx, float rdy, int64_t fd_pad) {
    switch (fd_pad) {
        case 1: return diffyh1_2(f, i, nx, rdy);
        case 2: return diffyh1_4(f, i, nx, rdy);
        case 3: return diffyh1_6(f, i, nx, rdy);
        default: return diffyh1_8(f, i, nx, rdy);
    }
}
inline float diffxh1(device const float* f, int64_t i, int64_t nx, float rdx, int64_t fd_pad) {
    switch (fd_pad) {
        case 1: return diffxh1_2(f, i, nx, rdx);
        case 2: return diffxh1_4(f, i, nx, rdx);
        case 3: return diffxh1_6(f, i, nx, rdx);
        default: return diffxh1_8(f, i, nx, rdx);
    }
}


// ============================================================================
// Kernel: Update H fields (Hx and Hz) with CPML
// ============================================================================

kernel void forward_kernel_h(
    device const float*  cq        [[buffer(0)]],
    device const float*  ey        [[buffer(1)]],
    device       float*  hx        [[buffer(2)]],
    device       float*  hz        [[buffer(3)]],
    device       float*  m_ey_x    [[buffer(4)]],
    device       float*  m_ey_z    [[buffer(5)]],
    device const float*  ay        [[buffer(6)]],
    device const float*  ayh       [[buffer(7)]],
    device const float*  ax        [[buffer(8)]],
    device const float*  axh       [[buffer(9)]],
    device const float*  by        [[buffer(10)]],
    device const float*  byh       [[buffer(11)]],
    device const float*  bx        [[buffer(12)]],
    device const float*  bxh       [[buffer(13)]],
    device const float*  ky        [[buffer(14)]],
    device const float*  kyh       [[buffer(15)]],
    device const float*  kx        [[buffer(16)]],
    device const float*  kxh       [[buffer(17)]],
    constant GridParams& params    [[buffer(18)]],
    uint3 gid [[thread_position_in_grid]])
{
    int64_t x = (int64_t)gid.x + params.fd_pad;
    int64_t y = (int64_t)gid.y + params.fd_pad;
    int64_t shot_idx = (int64_t)gid.z;

    if (shot_idx >= params.n_shots) return;
    if (y >= params.ny - params.fd_pad + 1) return;
    if (x >= params.nx - params.fd_pad + 1) return;

    int64_t const j = y * params.nx + x;
    int64_t const i = shot_idx * params.shot_numel + j;

    float const cq_val = params.cq_batched ? cq[i] : cq[j];

    // PML half-grid boundaries
    int64_t const pml_y0h = params.pml_y0;
    int64_t const pml_y1h = max(params.pml_y0, params.pml_y1 - 1);
    int64_t const pml_x0h = params.pml_x0;
    int64_t const pml_x1h = max(params.pml_x0, params.pml_x1 - 1);

    // Update Hx: Hx -= cq * dEy/dz
    if (y < params.ny - params.fd_pad) {
        bool pml_y_flag = y < pml_y0h || y >= pml_y1h;

        float dey_dz = diffyh1(ey, i, params.nx, params.rdy, params.fd_pad);

        if (pml_y_flag) {
            m_ey_z[i] = byh[y] * m_ey_z[i] + ayh[y] * dey_dz;
            dey_dz = dey_dz / kyh[y] + m_ey_z[i];
        }

        hx[i] -= cq_val * dey_dz;
    }

    // Update Hz: Hz += cq * dEy/dx
    if (x < params.nx - params.fd_pad) {
        bool pml_x_flag = x < pml_x0h || x >= pml_x1h;

        float dey_dx = diffxh1(ey, i, params.nx, params.rdx, params.fd_pad);

        if (pml_x_flag) {
            m_ey_x[i] = bxh[x] * m_ey_x[i] + axh[x] * dey_dx;
            dey_dx = dey_dx / kxh[x] + m_ey_x[i];
        }

        hz[i] += cq_val * dey_dx;
    }
}


// ============================================================================
// Kernel: Update E field (Ey) with CPML
// ============================================================================

kernel void forward_kernel_e(
    device const float*  ca        [[buffer(0)]],
    device const float*  cb        [[buffer(1)]],
    device const float*  hx        [[buffer(2)]],
    device const float*  hz        [[buffer(3)]],
    device       float*  ey        [[buffer(4)]],
    device       float*  m_hx_z    [[buffer(5)]],
    device       float*  m_hz_x    [[buffer(6)]],
    device const float*  ay        [[buffer(7)]],
    device const float*  ayh       [[buffer(8)]],
    device const float*  ax        [[buffer(9)]],
    device const float*  axh       [[buffer(10)]],
    device const float*  by        [[buffer(11)]],
    device const float*  byh       [[buffer(12)]],
    device const float*  bx        [[buffer(13)]],
    device const float*  bxh       [[buffer(14)]],
    device const float*  ky        [[buffer(15)]],
    device const float*  kyh       [[buffer(16)]],
    device const float*  kx        [[buffer(17)]],
    device const float*  kxh       [[buffer(18)]],
    constant GridParams& params    [[buffer(19)]],
    uint3 gid [[thread_position_in_grid]])
{
    int64_t x = (int64_t)gid.x + params.fd_pad;
    int64_t y = (int64_t)gid.y + params.fd_pad;
    int64_t shot_idx = (int64_t)gid.z;

    if (shot_idx >= params.n_shots) return;
    if (y >= params.ny - params.fd_pad + 1) return;
    if (x >= params.nx - params.fd_pad + 1) return;

    int64_t const j = y * params.nx + x;
    int64_t const i = shot_idx * params.shot_numel + j;

    float const ca_val = params.ca_batched ? ca[i] : ca[j];
    float const cb_val = params.cb_batched ? cb[i] : cb[j];

    bool pml_y = y < params.pml_y0 || y >= params.pml_y1;
    bool pml_x = x < params.pml_x0 || x >= params.pml_x1;

    float dhz_dx = diffx1(hz, i, params.nx, params.rdx, params.fd_pad);
    float dhx_dz = diffy1(hx, i, params.nx, params.rdy, params.fd_pad);

    if (pml_x) {
        m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
        dhz_dx = dhz_dx / kx[x] + m_hz_x[i];
    }

    if (pml_y) {
        m_hx_z[i] = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
        dhx_dz = dhx_dz / ky[y] + m_hx_z[i];
    }

    ey[i] = ca_val * ey[i] + cb_val * (dhz_dx - dhx_dz);
}


// ============================================================================
// Kernel: Add source to Ey field
// ============================================================================

kernel void add_sources_ey(
    device       float*   ey        [[buffer(0)]],
    device const float*   f         [[buffer(1)]],
    device const int64_t* sources_i [[buffer(2)]],
    constant GridParams&  params    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int64_t source_idx = (int64_t)gid.x;
    int64_t shot_idx   = (int64_t)gid.y;

    if (source_idx >= params.n_sources_per_shot) return;
    if (shot_idx >= params.n_shots) return;

    int64_t k = shot_idx * params.n_sources_per_shot + source_idx;
    int64_t src = sources_i[k];
    if (src >= 0) {
        ey[shot_idx * params.shot_numel + src] += f[k];
    }
}


// ============================================================================
// Kernel: Record Ey at receiver locations
// ============================================================================

kernel void record_receivers_ey(
    device       float*   r          [[buffer(0)]],
    device const float*   ey         [[buffer(1)]],
    device const int64_t* receivers_i [[buffer(2)]],
    constant GridParams&  params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int64_t receiver_idx = (int64_t)gid.x;
    int64_t shot_idx     = (int64_t)gid.y;

    if (receiver_idx >= params.n_receivers_per_shot) return;
    if (shot_idx >= params.n_shots) return;

    int64_t k = shot_idx * params.n_receivers_per_shot + receiver_idx;
    int64_t rec = receivers_i[k];
    if (rec >= 0) {
        r[k] = ey[shot_idx * params.shot_numel + rec];
    }
}


// ============================================================================
// Backward pass kernels
// ============================================================================

// Additional parameters for backward/storage kernels
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
    int64_t store_offset;    // offset into storage buffer for current timestep
    bool store_ey;           // whether to store Ey this timestep
    bool store_curl;         // whether to store curl_H this timestep
};


// ============================================================================
// Kernel: Forward E with storage (stores Ey and curl_H for backward pass)
// ============================================================================

kernel void forward_kernel_e_with_storage(
    device const float*  ca          [[buffer(0)]],
    device const float*  cb          [[buffer(1)]],
    device const float*  hx          [[buffer(2)]],
    device const float*  hz          [[buffer(3)]],
    device       float*  ey          [[buffer(4)]],
    device       float*  m_hx_z      [[buffer(5)]],
    device       float*  m_hz_x      [[buffer(6)]],
    device const float*  ay          [[buffer(7)]],
    device const float*  ayh         [[buffer(8)]],
    device const float*  ax          [[buffer(9)]],
    device const float*  axh         [[buffer(10)]],
    device const float*  by          [[buffer(11)]],
    device const float*  byh         [[buffer(12)]],
    device const float*  bx          [[buffer(13)]],
    device const float*  bxh         [[buffer(14)]],
    device const float*  ky          [[buffer(15)]],
    device const float*  kyh         [[buffer(16)]],
    device const float*  kx          [[buffer(17)]],
    device const float*  kxh         [[buffer(18)]],
    device       float*  ey_store    [[buffer(19)]],
    device       float*  curl_store  [[buffer(20)]],
    constant BackwardParams& params  [[buffer(21)]],
    uint3 gid [[thread_position_in_grid]])
{
    int64_t x = (int64_t)gid.x + params.fd_pad;
    int64_t y = (int64_t)gid.y + params.fd_pad;
    int64_t shot_idx = (int64_t)gid.z;

    if (shot_idx >= params.n_shots) return;
    if (y >= params.ny - params.fd_pad + 1) return;
    if (x >= params.nx - params.fd_pad + 1) return;

    int64_t const j = y * params.nx + x;
    int64_t const i = shot_idx * params.shot_numel + j;
    int64_t const store_idx = params.store_offset + shot_idx * params.shot_numel + j;

    float const ca_val = params.ca_batched ? ca[i] : ca[j];
    float const cb_val = params.cb_batched ? cb[i] : cb[j];

    bool pml_y = y < params.pml_y0 || y >= params.pml_y1;
    bool pml_x = x < params.pml_x0 || x >= params.pml_x1;

    float dhz_dx = diffx1(hz, i, params.nx, params.rdx, params.fd_pad);
    float dhx_dz = diffy1(hx, i, params.nx, params.rdy, params.fd_pad);

    if (pml_x) {
        m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
        dhz_dx = dhz_dx / kx[x] + m_hz_x[i];
    }

    if (pml_y) {
        m_hx_z[i] = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
        dhx_dz = dhx_dz / ky[y] + m_hx_z[i];
    }

    float curl_h = dhz_dx - dhx_dz;

    // Store Ey before update (needed for grad_ca)
    if (params.store_ey) {
        ey_store[store_idx] = ey[i];
    }
    // Store curl_H (needed for grad_cb)
    if (params.store_curl) {
        curl_store[store_idx] = curl_h;
    }

    ey[i] = ca_val * ey[i] + cb_val * curl_h;
}


// ============================================================================
// Kernel: Backward adjoint H fields (lambda_Hx, lambda_Hz)
// ============================================================================

kernel void backward_kernel_lambda_h(
    device const float*  cb            [[buffer(0)]],
    device const float*  lambda_ey     [[buffer(1)]],
    device       float*  lambda_hx     [[buffer(2)]],
    device       float*  lambda_hz     [[buffer(3)]],
    device       float*  m_lambda_ey_x [[buffer(4)]],
    device       float*  m_lambda_ey_z [[buffer(5)]],
    device const float*  ay            [[buffer(6)]],
    device const float*  ayh           [[buffer(7)]],
    device const float*  ax            [[buffer(8)]],
    device const float*  axh           [[buffer(9)]],
    device const float*  by            [[buffer(10)]],
    device const float*  byh           [[buffer(11)]],
    device const float*  bx            [[buffer(12)]],
    device const float*  bxh           [[buffer(13)]],
    device const float*  ky            [[buffer(14)]],
    device const float*  kyh           [[buffer(15)]],
    device const float*  kx            [[buffer(16)]],
    device const float*  kxh           [[buffer(17)]],
    constant BackwardParams& params    [[buffer(18)]],
    uint3 gid [[thread_position_in_grid]])
{
    int64_t x = (int64_t)gid.x + params.fd_pad;
    int64_t y = (int64_t)gid.y + params.fd_pad;
    int64_t shot_idx = (int64_t)gid.z;

    if (shot_idx >= params.n_shots) return;
    if (y >= params.ny - params.fd_pad + 1) return;
    if (x >= params.nx - params.fd_pad + 1) return;

    int64_t const j = y * params.nx + x;
    int64_t const i = shot_idx * params.shot_numel + j;

    float const cb_val = params.cb_batched ? cb[i] : cb[j];

    // PML half-grid boundaries
    int64_t const pml_y0h = params.pml_y0;
    int64_t const pml_y1h = max(params.pml_y0, params.pml_y1 - (int64_t)1);
    int64_t const pml_x0h = params.pml_x0;
    int64_t const pml_x1h = max(params.pml_x0, params.pml_x1 - (int64_t)1);

    // Update lambda_Hx: lambda_Hx -= cb * d(lambda_Ey)/dz
    if (y < params.ny - params.fd_pad) {
        bool pml_y_flag = y < pml_y0h || y >= pml_y1h;

        float d_lambda_ey_dz = diffyh1(lambda_ey, i, params.nx, params.rdy, params.fd_pad);

        if (pml_y_flag) {
            m_lambda_ey_z[i] = byh[y] * m_lambda_ey_z[i] + ayh[y] * d_lambda_ey_dz;
            d_lambda_ey_dz = d_lambda_ey_dz / kyh[y] + m_lambda_ey_z[i];
        }

        lambda_hx[i] -= cb_val * d_lambda_ey_dz;
    }

    // Update lambda_Hz: lambda_Hz += cb * d(lambda_Ey)/dx
    if (x < params.nx - params.fd_pad) {
        bool pml_x_flag = x < pml_x0h || x >= pml_x1h;

        float d_lambda_ey_dx = diffxh1(lambda_ey, i, params.nx, params.rdx, params.fd_pad);

        if (pml_x_flag) {
            m_lambda_ey_x[i] = bxh[x] * m_lambda_ey_x[i] + axh[x] * d_lambda_ey_dx;
            d_lambda_ey_dx = d_lambda_ey_dx / kxh[x] + m_lambda_ey_x[i];
        }

        lambda_hz[i] += cb_val * d_lambda_ey_dx;
    }
}


// ============================================================================
// Kernel: Backward adjoint E field (lambda_Ey) + gradient accumulation
// ============================================================================

kernel void backward_kernel_lambda_e_with_grad(
    device const float*  ca              [[buffer(0)]],
    device const float*  cq              [[buffer(1)]],
    device const float*  lambda_hx       [[buffer(2)]],
    device const float*  lambda_hz       [[buffer(3)]],
    device       float*  lambda_ey       [[buffer(4)]],
    device       float*  m_lambda_hx_z   [[buffer(5)]],
    device       float*  m_lambda_hz_x   [[buffer(6)]],
    device const float*  ey_store        [[buffer(7)]],
    device const float*  curl_h_store    [[buffer(8)]],
    device       float*  grad_ca         [[buffer(9)]],
    device       float*  grad_cb         [[buffer(10)]],
    device const float*  ay              [[buffer(11)]],
    device const float*  ayh             [[buffer(12)]],
    device const float*  ax              [[buffer(13)]],
    device const float*  axh             [[buffer(14)]],
    device const float*  by              [[buffer(15)]],
    device const float*  byh             [[buffer(16)]],
    device const float*  bx              [[buffer(17)]],
    device const float*  bxh             [[buffer(18)]],
    device const float*  ky              [[buffer(19)]],
    device const float*  kyh             [[buffer(20)]],
    device const float*  kx              [[buffer(21)]],
    device const float*  kxh             [[buffer(22)]],
    constant BackwardParams& params      [[buffer(23)]],
    uint3 gid [[thread_position_in_grid]])
{
    int64_t x = (int64_t)gid.x + params.fd_pad;
    int64_t y = (int64_t)gid.y + params.fd_pad;
    int64_t shot_idx = (int64_t)gid.z;

    if (shot_idx >= params.n_shots) return;
    if (y >= params.ny - params.fd_pad + 1) return;
    if (x >= params.nx - params.fd_pad + 1) return;

    int64_t const j = y * params.nx + x;
    int64_t const i = shot_idx * params.shot_numel + j;
    int64_t const store_idx = params.store_offset + shot_idx * params.shot_numel + j;

    float const ca_val = params.ca_batched ? ca[i] : ca[j];
    float const cq_val = params.cq_batched ? cq[i] : cq[j];

    bool pml_y = y < params.pml_y0 || y >= params.pml_y1;
    bool pml_x = x < params.pml_x0 || x >= params.pml_x1;

    // Compute d(lambda_Hz)/dx and d(lambda_Hx)/dz at integer grid points
    float d_lambda_hz_dx = diffx1(lambda_hz, i, params.nx, params.rdx, params.fd_pad);
    float d_lambda_hx_dz = diffy1(lambda_hx, i, params.nx, params.rdy, params.fd_pad);

    // Apply adjoint CPML
    if (pml_x) {
        m_lambda_hz_x[i] = bx[x] * m_lambda_hz_x[i] + ax[x] * d_lambda_hz_dx;
        d_lambda_hz_dx = d_lambda_hz_dx / kx[x] + m_lambda_hz_x[i];
    }
    if (pml_y) {
        m_lambda_hx_z[i] = by[y] * m_lambda_hx_z[i] + ay[y] * d_lambda_hx_dz;
        d_lambda_hx_dz = d_lambda_hx_dz / ky[y] + m_lambda_hx_z[i];
    }

    float curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;

    // Store current lambda_Ey before update (this is lambda_Ey^{n+1})
    float lambda_ey_curr = lambda_ey[i];

    // Update lambda_Ey: lambda_Ey^n = ca * lambda_Ey^{n+1} + cq * curl_lambda_H
    lambda_ey[i] = ca_val * lambda_ey_curr + cq_val * curl_lambda_h;

    // Accumulate gradients only in interior (non-PML) region
    if (!pml_y && !pml_x) {
        float step_ratio_f = float(params.step_ratio);

        // grad_ca += lambda_Ey^{n+1} * E_y^n * step_ratio
        if (params.ca_requires_grad && params.store_ey) {
            float ey_n = ey_store[store_idx];
            // For non-batched: accumulate to grid location j
            // For batched: accumulate to shot-specific location i
            if (params.ca_batched) {
                grad_ca[i] += lambda_ey_curr * ey_n * step_ratio_f;
            } else {
                grad_ca[j] += lambda_ey_curr * ey_n * step_ratio_f;
            }
        }

        // grad_cb += lambda_Ey^{n+1} * curl_H^n * step_ratio
        if (params.cb_requires_grad && params.store_curl) {
            float curl_h_n = curl_h_store[store_idx];
            if (params.ca_batched) {
                grad_cb[i] += lambda_ey_curr * curl_h_n * step_ratio_f;
            } else {
                grad_cb[j] += lambda_ey_curr * curl_h_n * step_ratio_f;
            }
        }
    }
}


// ============================================================================
// Kernel: Convert grad_ca/grad_cb to grad_epsilon/grad_sigma
// ============================================================================

kernel void convert_grad_kernel(
    device const float*  ca          [[buffer(0)]],
    device const float*  cb          [[buffer(1)]],
    device const float*  grad_ca     [[buffer(2)]],
    device const float*  grad_cb     [[buffer(3)]],
    device       float*  grad_eps    [[buffer(4)]],
    device       float*  grad_sigma  [[buffer(5)]],
    constant BackwardParams& params  [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    int64_t x = (int64_t)gid.x;
    int64_t y = (int64_t)gid.y;
    int64_t shot_idx = (int64_t)gid.z;

    int64_t const out_shots = params.ca_batched ? params.n_shots : (int64_t)1;
    if (shot_idx >= out_shots) return;
    if (y >= params.ny) return;
    if (x >= params.nx) return;

    int64_t const j = y * params.nx + x;
    int64_t const shot_offset = shot_idx * params.shot_numel;
    int64_t const out_idx = params.ca_batched ? (shot_offset + j) : j;
    int64_t const ca_idx = params.ca_batched ? (shot_offset + j) : j;
    int64_t const cb_idx = params.cb_batched ? (shot_offset + j) : j;

    float const ca_val = ca[ca_idx];
    float const cb_val = cb[cb_idx];

    float const grad_ca_val = params.ca_requires_grad ? grad_ca[out_idx] : 0.0f;
    float const grad_cb_val = params.cb_requires_grad ? grad_cb[out_idx] : 0.0f;

    float const inv_dt = 1.0f / params.dt;
    float const cb_sq = cb_val * cb_val;
    float const dca_de = (1.0f - ca_val) * cb_val * inv_dt;
    float const dcb_de = -cb_sq * inv_dt;
    float const dca_ds = -0.5f * (1.0f + ca_val) * cb_val;
    float const dcb_ds = -0.5f * cb_sq;

    // EP0 = 8.85418782e-12f
    float const EP0 = 8.85418782e-12f;

    if (grad_eps != nullptr) {
        grad_eps[out_idx] = (grad_ca_val * dca_de + grad_cb_val * dcb_de) * EP0;
    }
    if (grad_sigma != nullptr) {
        grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
    }
}

