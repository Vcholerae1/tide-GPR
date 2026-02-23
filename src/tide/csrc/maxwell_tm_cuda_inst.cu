#define TIDE_FD_PAD (tide::StencilTraits<TIDE_STENCIL>::FD_PAD)
namespace FUNC(Inst) {
#define LAMBDA_HX(dy, dx) lambda_hx[ND_INDEX(i, dy, dx)]
#define LAMBDA_HZ(dy, dx) lambda_hz[ND_INDEX(i, dy, dx)]
// Forward kernel: Update H fields (Hx and Hz)
__global__ __launch_bounds__(256) void forward_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ey, TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz, TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const m_ey_z, TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (shot_idx >= n_shots)
    return;
  if (y >= ny - TIDE_FD_PAD + 1)
    return;
  if (x >= nx - TIDE_FD_PAD + 1)
    return;

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,   ax,    axh,        by,     byh,    bx,
      bxh,     ky,    kyh,   kx,         kxh,    rdy,    rdx,
      n_shots, ny,    nx,    shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  false, false, cq_batched // ca/cb batched unused
  };

  forward_kernel_h_core<TIDE_DTYPE, TIDE_STENCIL>(
      params, cq, ey, hx, hz, m_ey_x, m_ey_z, y, x, shot_idx);
}

// Forward kernel: Update E field (Ey) - standard version
__global__ __launch_bounds__(256) void forward_kernel_e(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (shot_idx >= n_shots)
    return;
  if (y >= ny - TIDE_FD_PAD + 1)
    return;
  if (x >= nx - TIDE_FD_PAD + 1)
    return;

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false // cq batched unused
  };

  forward_kernel_e_core<TIDE_DTYPE, TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, y, x, shot_idx);
}

// Forward kernel: Update E field (Ey) with storage for gradient computation
__global__ __launch_bounds__(256) void forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const ey_store,     // Can be NULL
    TIDE_DTYPE *__restrict const curl_h_store, // Can be NULL
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh, bool const ca_requires_grad,
    bool const cb_requires_grad) {

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (shot_idx >= n_shots)
    return;
  if (y >= ny - TIDE_FD_PAD + 1)
    return;
  if (x >= nx - TIDE_FD_PAD + 1)
    return;

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false // cq batched unused
  };

  forward_kernel_e_with_storage_core<TIDE_DTYPE, TIDE_DTYPE, TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_store, curl_h_store,
      ca_requires_grad, cb_requires_grad, y, x, shot_idx);
}

// Forward kernel: Update E field (Ey) with BF16 storage for gradient
// computation Stores Ey and curl_H in __nv_bfloat16 to reduce snapshot
// bandwidth/size.
__global__ __launch_bounds__(256) void forward_kernel_e_with_storage_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    __nv_bfloat16 *__restrict const ey_store,     // Can be NULL
    __nv_bfloat16 *__restrict const curl_h_store, // Can be NULL
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh, bool const ca_requires_grad,
    bool const cb_requires_grad) {

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (shot_idx >= n_shots)
    return;
  if (y >= ny - TIDE_FD_PAD + 1)
    return;
  if (x >= nx - TIDE_FD_PAD + 1)
    return;

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false // cq batched unused
  };

  forward_kernel_e_with_storage_core<TIDE_DTYPE, __nv_bfloat16, TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_store, curl_h_store,
      ca_requires_grad, cb_requires_grad, y, x, shot_idx);
}

// Backward kernel: Update adjoint λ_H fields
__global__ void
backward_kernel_lambda_h(TIDE_DTYPE const *__restrict const cb,
                         TIDE_DTYPE const *__restrict const lambda_ey,
                         TIDE_DTYPE *__restrict const lambda_hx,
                         TIDE_DTYPE *__restrict const lambda_hz,
                         TIDE_DTYPE *__restrict const m_lambda_ey_x,
                         TIDE_DTYPE *__restrict const m_lambda_ey_z,
                         TIDE_DTYPE const *__restrict const ay,
                         TIDE_DTYPE const *__restrict const ayh,
                         TIDE_DTYPE const *__restrict const ax,
                         TIDE_DTYPE const *__restrict const axh,
                         TIDE_DTYPE const *__restrict const by,
                         TIDE_DTYPE const *__restrict const byh,
                         TIDE_DTYPE const *__restrict const bx,
                         TIDE_DTYPE const *__restrict const bxh,
                         TIDE_DTYPE const *__restrict const ky,
                         TIDE_DTYPE const *__restrict const kyh,
                         TIDE_DTYPE const *__restrict const kx,
                         TIDE_DTYPE const *__restrict const kxh) {

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (shot_idx >= n_shots)
    return;
  if (y >= ny - TIDE_FD_PAD + 1)
    return;
  if (x >= nx - TIDE_FD_PAD + 1)
    return;

  GridParams<TIDE_DTYPE> params =
      {
          ay,      ayh,   ax,         axh,        by,     byh,    bx,
          bxh,     ky,    kyh,        kx,         kxh,    rdy,    rdx,
          n_shots, ny,    nx,         shot_numel, pml_y0, pml_y1, pml_x0,
          pml_x1,  false, cb_batched, false // ca/cq batched unused
      };

  backward_kernel_lambda_h_core<TIDE_DTYPE, TIDE_STENCIL>(
      params, cb, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x, m_lambda_ey_z,
      y, x, shot_idx);
}

// Backward kernel: Update adjoint λ_Ey field with per-shot gradient
// accumulation Uses pml_y0/pml_y1/pml_x0/pml_x1 for both adjoint propagation
// and gradient masking NO atomicAdd - each shot writes to its own memory region
__global__ void backward_kernel_lambda_e_with_grad(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const curl_h_store,
    TIDE_DTYPE
        *__restrict const grad_ca_shot, // [n_shots, ny, nx] - per-shot gradient
    TIDE_DTYPE
        *__restrict const grad_cb_shot, // [n_shots, ny, nx] - per-shot gradient
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val) {

  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (y < ny - TIDE_FD_PAD + 1 && x < nx - TIDE_FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cq_shot_i = cq_batched ? cq[i] : cq[j];

    // Determine PML region (pml_y/pml_x = true means in PML region)
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    // Compute D_x^{hT}[λ_Hz] at integer grid points
    // EXACT ADJOINT: use transpose of DIFFX1 -> which is DIFFXH1
    TIDE_DTYPE d_lambda_hz_dx = DIFFXH1(LAMBDA_HZ);
    // Compute D_z^{hT}[λ_Hx] at integer grid points
    // EXACT ADJOINT: use transpose of DIFFY1 -> which is DIFFYH1
    TIDE_DTYPE d_lambda_hx_dz = DIFFYH1(LAMBDA_HX);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    // Apply adjoint CPML for d(λ_Hz)/dx (only in PML region)
    if (pml_x) {
      m_lambda_hz_x[i] = bx_val * m_lambda_hz_x[i] + ax_val * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / kx_val + m_lambda_hz_x[i];
    }

    // Apply adjoint CPML for d(λ_Hx)/dz (only in PML region)
    if (pml_y) {
      m_lambda_hx_z[i] = by_val * m_lambda_hx_z[i] + ay_val * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / ky_val + m_lambda_hx_z[i];
    }

    // curl_λH = d(λ_Hz)/dx - d(λ_Hx)/dz
    TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;

    // Store current λ_Ey before update (this is λ_Ey^{n+1})
    TIDE_DTYPE lambda_ey_curr = lambda_ey[i];

    // Update λ_Ey: λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * curl_λH
    lambda_ey[i] = ca_shot_i * lambda_ey_curr + cq_shot_i * curl_lambda_h;

    // Accumulate per-shot gradients only in interior region (!pml_y && !pml_x)
    if (!pml_y && !pml_x) {
      // grad_ca_shot[shot_idx, y, x] += λ_Ey^{n+1} * E_y^n
      // Convert from BF16 back to FP32 for computation
      if (ca_requires_grad && ey_store != nullptr) {
        TIDE_DTYPE ey_n = ey_store[i];
        grad_ca_shot[i] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio_val;
      }

      // grad_cb_shot[shot_idx, y, x] += λ_Ey^{n+1} * curl_H^n
      if (cb_requires_grad && curl_h_store != nullptr) {
        TIDE_DTYPE curl_h_n = curl_h_store[i];
        grad_cb_shot[i] +=
            lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio_val;
      }
    }
  }
}

// Backward kernel: Update adjoint λ_Ey field with BF16 snapshot loads.
__global__ void backward_kernel_lambda_e_with_grad_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    __nv_bfloat16 const *__restrict const ey_store,
    __nv_bfloat16 const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val) {

  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (y < ny - TIDE_FD_PAD + 1 && x < nx - TIDE_FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_shot_i = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cq_shot_i = cq_batched ? cq[i] : cq[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    // EXACT ADJOINT: use transposed difference operators
    TIDE_DTYPE d_lambda_hz_dx = DIFFXH1(LAMBDA_HZ);
    TIDE_DTYPE d_lambda_hx_dz = DIFFYH1(LAMBDA_HX);

    // Pre-load PML coefficients into registers (optimization 1.2)
    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_lambda_hz_x[i] = bx_val * m_lambda_hz_x[i] + ax_val * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / kx_val + m_lambda_hz_x[i];
    }

    if (pml_y) {
      m_lambda_hx_z[i] = by_val * m_lambda_hx_z[i] + ay_val * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / ky_val + m_lambda_hx_z[i];
    }

    TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;

    TIDE_DTYPE lambda_ey_curr = lambda_ey[i];
    lambda_ey[i] = ca_shot_i * lambda_ey_curr + cq_shot_i * curl_lambda_h;

    if (!pml_y && !pml_x) {
      if (ca_requires_grad && ey_store != nullptr) {
        TIDE_DTYPE ey_n = (TIDE_DTYPE)__bfloat162float(ey_store[i]);
        grad_ca_shot[i] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio_val;
      }
      if (cb_requires_grad && curl_h_store != nullptr) {
        TIDE_DTYPE curl_h_n = (TIDE_DTYPE)__bfloat162float(curl_h_store[i]);
        grad_cb_shot[i] +=
            lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio_val;
      }
    }
  }
}

// Backward kernel: Update adjoint λ_Ey field (no gradient accumulation).
__global__ void
backward_kernel_lambda_e(TIDE_DTYPE const *__restrict const ca,
                         TIDE_DTYPE const *__restrict const cq,
                         TIDE_DTYPE const *__restrict const lambda_hx,
                         TIDE_DTYPE const *__restrict const lambda_hz,
                         TIDE_DTYPE *__restrict const lambda_ey,
                         TIDE_DTYPE *__restrict const m_lambda_hx_z,
                         TIDE_DTYPE *__restrict const m_lambda_hz_x,
                         TIDE_DTYPE const *__restrict const ay,
                         TIDE_DTYPE const *__restrict const ayh,
                         TIDE_DTYPE const *__restrict const ax,
                         TIDE_DTYPE const *__restrict const axh,
                         TIDE_DTYPE const *__restrict const by,
                         TIDE_DTYPE const *__restrict const byh,
                         TIDE_DTYPE const *__restrict const bx,
                         TIDE_DTYPE const *__restrict const bxh,
                         TIDE_DTYPE const *__restrict const ky,
                         TIDE_DTYPE const *__restrict const kyh,
                         TIDE_DTYPE const *__restrict const kx,
                         TIDE_DTYPE const *__restrict const kxh) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (shot_idx >= n_shots)
    return;
  if (y >= ny - TIDE_FD_PAD + 1)
    return;
  if (x >= nx - TIDE_FD_PAD + 1)
    return;

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,    axh,        by,     byh,    bx,
      bxh,     ky,         kyh,   kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,    shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, false, cq_batched // cb batched unused
  };

  backward_kernel_lambda_e_with_grad_core<TIDE_DTYPE, TIDE_DTYPE, TIDE_STENCIL>(
      params, ca, cq, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
      m_lambda_hz_x, (TIDE_DTYPE const *)nullptr, (TIDE_DTYPE const *)nullptr,
      (TIDE_DTYPE *)nullptr, (TIDE_DTYPE *)nullptr, false, false, 0, y, x,
      shot_idx);
}

// Exact CPML adjoint helpers: prepare transformed fluxes, then apply transposed
// spatial operators in a separate kernel to preserve discrete consistency.
__global__ void backward_kernel_lambda_h_prepare_exact(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
    TIDE_DTYPE *__restrict const work_x, TIDE_DTYPE *__restrict const work_z,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (y < ny - TIDE_FD_PAD + 1 && x < nx - TIDE_FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const pml_y0h = pml_y0;
    int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
    int64_t const pml_x0h = pml_x0;
    int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);
    bool const pml_y = y < pml_y0h || y >= pml_y1h;
    bool const pml_x = x < pml_x0h || x >= pml_x1h;

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;
    TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];
    TIDE_DTYPE const g = cb_val * lambda_ey[i];
    if (pml_x) {
      TIDE_DTYPE const tmp_x = m_lambda_ey_x[i] + g;
      work_x[i] = g / __ldg(&kx[x]) + __ldg(&ax[x]) * tmp_x;
      m_lambda_ey_x[i] = __ldg(&bx[x]) * tmp_x;
    } else {
      work_x[i] = g;
    }
    if (pml_y) {
      TIDE_DTYPE const tmp_z = m_lambda_ey_z[i] + g;
      work_z[i] = g / __ldg(&ky[y]) + __ldg(&ay[y]) * tmp_z;
      m_lambda_ey_z[i] = __ldg(&by[y]) * tmp_z;
    } else {
      work_z[i] = g;
    }
  }
}

__global__ void
backward_kernel_lambda_h_apply_exact(TIDE_DTYPE const *__restrict const work_x,
                                     TIDE_DTYPE const *__restrict const work_z,
                                     TIDE_DTYPE *__restrict const lambda_hx,
                                     TIDE_DTYPE *__restrict const lambda_hz) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (y < ny - TIDE_FD_PAD + 1 && x < nx - TIDE_FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;
#define ONE(dy, dx) ((TIDE_DTYPE)1)
#define WORK_X_L(dy, dx) work_x[ND_INDEX(i, dy, dx)]
#define WORK_Z_L(dy, dx) work_z[ND_INDEX(i, dy, dx)]
    if (y < ny - TIDE_FD_PAD) {
      lambda_hx[i] -= DIFFY1_ADJ(ONE, WORK_Z_L);
    }
    if (x < nx - TIDE_FD_PAD) {
      lambda_hz[i] += DIFFX1_ADJ(ONE, WORK_X_L);
    }
#undef ONE
#undef WORK_X_L
#undef WORK_Z_L
  }
}

__global__ void backward_kernel_lambda_e_prepare_exact(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const work_x, TIDE_DTYPE *__restrict const work_z,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (y < ny - TIDE_FD_PAD + 1 && x < nx - TIDE_FD_PAD + 1 && shot_idx < n_shots) {
    bool const pml_y = y < pml_y0 || y >= pml_y1;
    bool const pml_x = x < pml_x0 || x >= pml_x1;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;
    TIDE_DTYPE const cq_val = cq_batched ? cq[i] : cq[j];
    TIDE_DTYPE const g_x = cq_val * lambda_hz[i];
    TIDE_DTYPE const g_z = -cq_val * lambda_hx[i];
    if (pml_x) {
      TIDE_DTYPE const tmp_x = m_lambda_hz_x[i] + g_x;
      work_x[i] = g_x / __ldg(&kxh[x]) + __ldg(&axh[x]) * tmp_x;
      m_lambda_hz_x[i] = __ldg(&bxh[x]) * tmp_x;
    } else {
      work_x[i] = g_x;
    }
    if (pml_y) {
      TIDE_DTYPE const tmp_z = m_lambda_hx_z[i] + g_z;
      work_z[i] = g_z / __ldg(&kyh[y]) + __ldg(&ayh[y]) * tmp_z;
      m_lambda_hx_z[i] = __ldg(&byh[y]) * tmp_z;
    } else {
      work_z[i] = g_z;
    }
  }
}

__global__ void backward_kernel_lambda_e_apply_exact(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const work_x,
    TIDE_DTYPE const *__restrict const work_z,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (y < ny - TIDE_FD_PAD + 1 && x < nx - TIDE_FD_PAD + 1 && shot_idx < n_shots) {
    bool const pml_y = y < pml_y0 || y >= pml_y1;
    bool const pml_x = x < pml_x0 || x >= pml_x1;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;
    TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
#define ONE(dy, dx) ((TIDE_DTYPE)1)
#define WORK_X_L(dy, dx) work_x[ND_INDEX(i, dy, dx)]
#define WORK_Z_L(dy, dx) work_z[ND_INDEX(i, dy, dx)]
    TIDE_DTYPE const curl_lambda_h =
        DIFFXH1_ADJ(ONE, WORK_X_L) + DIFFYH1_ADJ(ONE, WORK_Z_L);
#undef WORK_X_L
#undef WORK_Z_L
#undef ONE
    TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
    lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
    if (!pml_y && !pml_x && ca_requires_grad && ey_store != nullptr) {
      grad_ca_shot[i] +=
          lambda_ey_curr * ey_store[i] * (TIDE_DTYPE)step_ratio_val;
    }
    if (!pml_y && !pml_x && cb_requires_grad && curl_h_store != nullptr) {
      grad_cb_shot[i] +=
          lambda_ey_curr * curl_h_store[i] * (TIDE_DTYPE)step_ratio_val;
    }
  }
}

__global__ void backward_kernel_lambda_e_apply_exact_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const work_x,
    TIDE_DTYPE const *__restrict const work_z,
    TIDE_DTYPE *__restrict const lambda_ey,
    __nv_bfloat16 const *__restrict const ey_store,
    __nv_bfloat16 const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (y < ny - TIDE_FD_PAD + 1 && x < nx - TIDE_FD_PAD + 1 && shot_idx < n_shots) {
    bool const pml_y = y < pml_y0 || y >= pml_y1;
    bool const pml_x = x < pml_x0 || x >= pml_x1;
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;
    TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
#define ONE(dy, dx) ((TIDE_DTYPE)1)
#define WORK_X_L(dy, dx) work_x[ND_INDEX(i, dy, dx)]
#define WORK_Z_L(dy, dx) work_z[ND_INDEX(i, dy, dx)]
    TIDE_DTYPE const curl_lambda_h =
        DIFFXH1_ADJ(ONE, WORK_X_L) + DIFFYH1_ADJ(ONE, WORK_Z_L);
#undef WORK_X_L
#undef WORK_Z_L
#undef ONE
    TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
    lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
    if (!pml_y && !pml_x && ca_requires_grad && ey_store != nullptr) {
      TIDE_DTYPE const ey_n = (TIDE_DTYPE)__bfloat162float(ey_store[i]);
      grad_ca_shot[i] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio_val;
    }
    if (!pml_y && !pml_x && cb_requires_grad && curl_h_store != nullptr) {
      TIDE_DTYPE const curl_h_n = (TIDE_DTYPE)__bfloat162float(curl_h_store[i]);
      grad_cb_shot[i] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio_val;
    }
  }
}

// Combine per-shot gradients into final gradient (sum across shots)
__global__ void combine_grad(TIDE_DTYPE *__restrict const grad,
                             TIDE_DTYPE const *__restrict const grad_shot) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + TIDE_FD_PAD;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + TIDE_FD_PAD;
  if (y < ny - TIDE_FD_PAD && x < nx - TIDE_FD_PAD) {
    int64_t j = y * nx + x;
    int64_t const stride = shot_numel;
    TIDE_DTYPE sum = 0;
#pragma unroll 4
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      sum += grad_shot[shot_idx * stride + j];
    }
    grad[j] += sum;
  }
}

} // namespace

// Forward propagation function
extern "C" void FUNC(forward)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f, TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx, TIDE_DTYPE *const hz, TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z, TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x, TIDE_DTYPE *const r, TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by, TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh, TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh, TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh, TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, scalar_t const rdy_h,
    scalar_t const rdx_h, scalar_t const dt_h, int64_t const nt,
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;

  int64_t const shot_numel_h = ny_h * nx_h;

  // Copy constants to device with caching to avoid redundant copies
  static scalar_t cached_rdy = 0, cached_rdx = 0;
  static int64_t cached_n_shots = -1, cached_ny = -1, cached_nx = -1;
  static int64_t cached_shot_numel = -1, cached_n_sources_per_shot = -1,
                 cached_n_receivers_per_shot = -1;
  static int64_t cached_pml_y0 = -1, cached_pml_y1 = -1;
  static int64_t cached_pml_x0 = -1, cached_pml_x1 = -1;
  static bool cached_ca_batched = false, cached_cb_batched = false,
              cached_cq_batched = false;
  static int64_t cached_device = -1;
  static bool first_call = true;

  if (first_call || cached_device != device || cached_rdy != rdy_h ||
      cached_rdx != rdx_h || cached_n_shots != n_shots_h || cached_ny != ny_h ||
      cached_nx != nx_h || cached_shot_numel != shot_numel_h ||
      cached_n_sources_per_shot != n_sources_per_shot_h ||
      cached_n_receivers_per_shot != n_receivers_per_shot_h ||
      cached_pml_y0 != pml_y0_h || cached_pml_y1 != pml_y1_h ||
      cached_pml_x0 != pml_x0_h || cached_pml_x1 != pml_x1_h ||
      cached_ca_batched != ca_batched_h || cached_cb_batched != cb_batched_h ||
      cached_cq_batched != cq_batched_h) {

    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(scalar_t));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(scalar_t));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                       sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                       sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

    cached_rdy = rdy_h;
    cached_rdx = rdx_h;
    cached_n_shots = n_shots_h;
    cached_ny = ny_h;
    cached_nx = nx_h;
    cached_shot_numel = shot_numel_h;
    cached_n_sources_per_shot = n_sources_per_shot_h;
    cached_n_receivers_per_shot = n_receivers_per_shot_h;
    cached_pml_y0 = pml_y0_h;
    cached_pml_y1 = pml_y1_h;
    cached_pml_x0 = pml_x0_h;
    cached_pml_x1 = pml_x1_h;
    cached_ca_batched = ca_batched_h;
    cached_cb_batched = cb_batched_h;
    cached_cq_batched = cq_batched_h;
    cached_device = device;
    first_call = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * TIDE_FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * TIDE_FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);
#if TIDE_FD_PAD > 1
  size_t const shmem_h_bytes = (size_t)(dimBlock.x + 2 * TIDE_FD_PAD) *
                               (size_t)(dimBlock.y + 2 * TIDE_FD_PAD) *
                               sizeof(TIDE_DTYPE);
  size_t const shmem_e_bytes = 0;
#else
  size_t const shmem_h_bytes = 0;
  size_t const shmem_e_bytes = 0;
#endif

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources((n_sources_per_shot_h + dimBlock_sources.x - 1) /
                           dimBlock_sources.x,
                       n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers((n_receivers_per_shot_h + dimBlock_receivers.x - 1) /
                             dimBlock_receivers.x,
                         n_shots_h, 1);

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<dimGrid, dimBlock, shmem_h_bytes>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);
    forward_kernel_e<<<dimGrid, dimBlock, shmem_e_bytes>>>(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  gpuErrchk(cudaPeekAtLastError());
}

extern "C" void FUNC(forward_with_storage)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f, TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx, TIDE_DTYPE *const hz, TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z, TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x, TIDE_DTYPE *const r, void *const ey_store_1,
    void *const ey_store_3, char const *const *const ey_filenames,
    void *const curl_store_1, void *const curl_store_3,
    char const *const *const curl_filenames, TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by, TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh, TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh, TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh, TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, scalar_t const rdy_h,
    scalar_t const rdx_h, scalar_t const dt_h, int64_t const nt,
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h, int64_t const storage_mode_h,
    int64_t const shot_bytes_uncomp_h, bool const ca_requires_grad,
    bool const cb_requires_grad, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device) {

  cudaSetDevice(device);
  (void)n_threads;

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (shot_bytes_uncomp_h == shot_numel_h * 2);
  cudaStream_t copy_stream = nullptr;
  cudaEvent_t store_ready;
  cudaEvent_t copy_done[NUM_BUFFERS];
  bool copy_in_flight[NUM_BUFFERS];
  for (int i = 0; i < NUM_BUFFERS; i++)
    copy_in_flight[i] = false;

  if (storage_mode_h == STORAGE_CPU) {
    gpuErrchk(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    gpuErrchk(cudaEventCreateWithFlags(&store_ready, cudaEventDisableTiming));
    for (int i = 0; i < NUM_BUFFERS; i++) {
      gpuErrchk(
          cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming));
    }
  }

  // Copy constants to device with caching to avoid redundant copies
  static scalar_t cached_rdy2 = 0, cached_rdx2 = 0;
  static int64_t cached_n_shots2 = -1, cached_ny2 = -1, cached_nx2 = -1;
  static int64_t cached_shot_numel2 = -1, cached_n_sources_per_shot2 = -1,
                 cached_n_receivers_per_shot2 = -1;
  static int64_t cached_pml_y02 = -1, cached_pml_y12 = -1;
  static int64_t cached_pml_x02 = -1, cached_pml_x12 = -1;
  static bool cached_ca_batched2 = false, cached_cb_batched2 = false,
              cached_cq_batched2 = false;
  static int64_t cached_device2 = -1;
  static bool first_call2 = true;

  if (first_call2 || cached_device2 != device || cached_rdy2 != rdy_h ||
      cached_rdx2 != rdx_h || cached_n_shots2 != n_shots_h ||
      cached_ny2 != ny_h || cached_nx2 != nx_h ||
      cached_shot_numel2 != shot_numel_h ||
      cached_n_sources_per_shot2 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot2 != n_receivers_per_shot_h ||
      cached_pml_y02 != pml_y0_h || cached_pml_y12 != pml_y1_h ||
      cached_pml_x02 != pml_x0_h || cached_pml_x12 != pml_x1_h ||
      cached_ca_batched2 != ca_batched_h ||
      cached_cb_batched2 != cb_batched_h ||
      cached_cq_batched2 != cq_batched_h) {

    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(scalar_t));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(scalar_t));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                       sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                       sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

    cached_rdy2 = rdy_h;
    cached_rdx2 = rdx_h;
    cached_n_shots2 = n_shots_h;
    cached_ny2 = ny_h;
    cached_nx2 = nx_h;
    cached_shot_numel2 = shot_numel_h;
    cached_n_sources_per_shot2 = n_sources_per_shot_h;
    cached_n_receivers_per_shot2 = n_receivers_per_shot_h;
    cached_pml_y02 = pml_y0_h;
    cached_pml_y12 = pml_y1_h;
    cached_pml_x02 = pml_x0_h;
    cached_pml_x12 = pml_x1_h;
    cached_ca_batched2 = ca_batched_h;
    cached_cb_batched2 = cb_batched_h;
    cached_cq_batched2 = cq_batched_h;
    cached_device2 = device;
    first_call2 = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * TIDE_FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * TIDE_FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);
#if TIDE_FD_PAD > 1
  size_t const shmem_h_bytes = (size_t)(dimBlock.x + 2 * TIDE_FD_PAD) *
                               (size_t)(dimBlock.y + 2 * TIDE_FD_PAD) *
                               sizeof(TIDE_DTYPE);
  size_t const shmem_e_bytes = 2 * shmem_h_bytes;
#else
  size_t const shmem_h_bytes = 0;
  size_t const shmem_e_bytes = 0;
#endif

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources((n_sources_per_shot_h + dimBlock_sources.x - 1) /
                           dimBlock_sources.x,
                       n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers((n_receivers_per_shot_h + dimBlock_receivers.x - 1) /
                             dimBlock_receivers.x,
                         n_shots_h, 1);

  FILE *fp_ey = nullptr;
  FILE *fp_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad)
      fp_ey = fopen(ey_filenames[0], "wb");
    if (cb_requires_grad)
      fp_curl = fopen(curl_filenames[0], "wb");
  }

  auto store1_offset_bytes = [&](int64_t step_idx) -> size_t {
    if (storage_mode_h == STORAGE_DEVICE) {
      return (size_t)step_idx * bytes_per_step_store;
    }
    if (storage_mode_h == STORAGE_CPU) {
      return (size_t)(step_idx % NUM_BUFFERS) * bytes_per_step_store;
    }
    return 0;
  };

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<dimGrid, dimBlock, shmem_h_bytes>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);

    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    bool const want_store = store_ey || store_curl;
    if (want_store) {
      int64_t const step_idx = t / step_ratio_h;
      int const store_buf =
          (storage_mode_h == STORAGE_CPU) ? (int)(step_idx % NUM_BUFFERS) : 0;
      if (storage_mode_h == STORAGE_CPU && copy_in_flight[store_buf]) {
        gpuErrchk(cudaStreamWaitEvent(0, copy_done[store_buf], 0));
        copy_in_flight[store_buf] = false;
      }
      size_t const store1_offset = store1_offset_bytes(step_idx);

      void *__restrict const ey_store_1_t =
          (uint8_t *)ey_store_1 + store1_offset;
      void *__restrict const ey_store_3_t =
          (uint8_t *)ey_store_3 + (storage_mode_h == STORAGE_CPU
                                       ? (size_t)step_idx * bytes_per_step_store
                                       : 0);

      void *__restrict const curl_store_1_t =
          (uint8_t *)curl_store_1 + store1_offset;
      void *__restrict const curl_store_3_t =
          (uint8_t *)curl_store_3 +
          (storage_mode_h == STORAGE_CPU
               ? (size_t)step_idx * bytes_per_step_store
               : 0);

      if (storage_bf16_h) {
        forward_kernel_e_with_storage_bf16<<<dimGrid, dimBlock,
                                             shmem_e_bytes>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
            store_curl ? (__nv_bfloat16 *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      } else {
        forward_kernel_e_with_storage<<<dimGrid, dimBlock, shmem_e_bytes>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
            store_curl ? (TIDE_DTYPE *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      }

      if (storage_mode_h == STORAGE_CPU) {
        gpuErrchk(cudaEventRecord(store_ready, 0));
        gpuErrchk(cudaStreamWaitEvent(copy_stream, store_ready, 0));
        if (store_ey) {
          gpuErrchk(cudaMemcpyAsync(ey_store_3_t, ey_store_1_t,
                                    bytes_per_step_store,
                                    cudaMemcpyDeviceToHost, copy_stream));
        }
        if (store_curl) {
          gpuErrchk(cudaMemcpyAsync(curl_store_3_t, curl_store_1_t,
                                    bytes_per_step_store,
                                    cudaMemcpyDeviceToHost, copy_stream));
        }
        gpuErrchk(cudaEventRecord(copy_done[store_buf], copy_stream));
        copy_in_flight[store_buf] = true;
      } else {
        if (store_ey) {
          storage_save_snapshot_gpu(
              ey_store_1_t, ey_store_3_t, fp_ey, storage_mode_h, step_idx,
              (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
        }
        if (store_curl) {
          storage_save_snapshot_gpu(
              curl_store_1_t, curl_store_3_t, fp_curl, storage_mode_h, step_idx,
              (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
        }
      }
    } else {
      forward_kernel_e<<<dimGrid, dimBlock, shmem_e_bytes>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<dimGrid_sources, dimBlock_sources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  if (storage_mode_h == STORAGE_CPU) {
    gpuErrchk(cudaStreamSynchronize(copy_stream));
    for (int i = 0; i < NUM_BUFFERS; i++) {
      gpuErrchk(cudaEventDestroy(copy_done[i]));
    }
    gpuErrchk(cudaEventDestroy(store_ready));
    gpuErrchk(cudaStreamDestroy(copy_stream));
  }

  if (fp_ey != nullptr)
    fclose(fp_ey);
  if (fp_curl != nullptr)
    fclose(fp_curl);

  gpuErrchk(cudaPeekAtLastError());
}

extern "C" void FUNC(backward)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const lambda_ey, TIDE_DTYPE *const lambda_hx,
    TIDE_DTYPE *const lambda_hz, TIDE_DTYPE *const m_lambda_ey_x,
    TIDE_DTYPE *const m_lambda_ey_z, TIDE_DTYPE *const m_lambda_hx_z,
    TIDE_DTYPE *const m_lambda_hz_x, void *const ey_store_1,
    void *const ey_store_3, char const *const *const ey_filenames,
    void *const curl_store_1, void *const curl_store_3,
    char const *const *const curl_filenames, TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca, TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE
        *const grad_ca_shot, // [n_shots, ny, nx] - per-shot gradient workspace
    TIDE_DTYPE
        *const grad_cb_shot, // [n_shots, ny, nx] - per-shot gradient workspace
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    scalar_t const rdy_h, scalar_t const rdx_h, scalar_t const dt_h,
    int64_t const nt, int64_t const n_shots_h, int64_t const ny_h,
    int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    int64_t const storage_mode_h, int64_t const shot_bytes_uncomp_h,
    bool const ca_requires_grad, bool const cb_requires_grad,
    bool const ca_batched_h, bool const cb_batched_h, bool const cq_batched_h,
    int64_t const start_t, int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h, int64_t const n_threads,
    int64_t const device) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)n_threads;

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (shot_bytes_uncomp_h == shot_numel_h * 2);
  cudaStream_t copy_stream = nullptr;
  cudaEvent_t copy_done[NUM_BUFFERS];
  bool copy_in_flight[NUM_BUFFERS];
  for (int i = 0; i < NUM_BUFFERS; i++)
    copy_in_flight[i] = false;

  if (storage_mode_h == STORAGE_CPU) {
    gpuErrchk(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    for (int i = 0; i < NUM_BUFFERS; i++) {
      gpuErrchk(
          cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming));
    }
  }

  // Copy constants to device with caching to avoid redundant copies
  static scalar_t cached_rdy3 = 0, cached_rdx3 = 0;
  static int64_t cached_n_shots3 = -1, cached_ny3 = -1, cached_nx3 = -1;
  static int64_t cached_shot_numel3 = -1, cached_n_sources_per_shot3 = -1,
                 cached_n_receivers_per_shot3 = -1;
  static int64_t cached_pml_y03 = -1, cached_pml_y13 = -1;
  static int64_t cached_pml_x03 = -1, cached_pml_x13 = -1;
  static bool cached_ca_batched3 = false, cached_cb_batched3 = false,
              cached_cq_batched3 = false;
  static int64_t cached_device3 = -1;
  static bool first_call3 = true;

  if (first_call3 || cached_device3 != device || cached_rdy3 != rdy_h ||
      cached_rdx3 != rdx_h || cached_n_shots3 != n_shots_h ||
      cached_ny3 != ny_h || cached_nx3 != nx_h ||
      cached_shot_numel3 != shot_numel_h ||
      cached_n_sources_per_shot3 != n_sources_per_shot_h ||
      cached_n_receivers_per_shot3 != n_receivers_per_shot_h ||
      cached_pml_y03 != pml_y0_h || cached_pml_y13 != pml_y1_h ||
      cached_pml_x03 != pml_x0_h || cached_pml_x13 != pml_x1_h ||
      cached_ca_batched3 != ca_batched_h ||
      cached_cb_batched3 != cb_batched_h ||
      cached_cq_batched3 != cq_batched_h) {

    cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(scalar_t));
    cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(scalar_t));
    cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
    cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
    cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
    cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                       sizeof(int64_t));
    cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                       sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
    cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
    cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
    cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

    cached_rdy3 = rdy_h;
    cached_rdx3 = rdx_h;
    cached_n_shots3 = n_shots_h;
    cached_ny3 = ny_h;
    cached_nx3 = nx_h;
    cached_shot_numel3 = shot_numel_h;
    cached_n_sources_per_shot3 = n_sources_per_shot_h;
    cached_n_receivers_per_shot3 = n_receivers_per_shot_h;
    cached_pml_y03 = pml_y0_h;
    cached_pml_y13 = pml_y1_h;
    cached_pml_x03 = pml_x0_h;
    cached_pml_x13 = pml_x1_h;
    cached_ca_batched3 = ca_batched_h;
    cached_cb_batched3 = cb_batched_h;
    cached_cq_batched3 = cq_batched_h;
    cached_device3 = device;
    first_call3 = false;
  }

  dim3 dimBlock(32, 8, 1);
  int64_t gridx = (nx_h - 2 * TIDE_FD_PAD + 2 + dimBlock.x - 1) / dimBlock.x;
  int64_t gridy = (ny_h - 2 * TIDE_FD_PAD + 2 + dimBlock.y - 1) / dimBlock.y;
  int64_t gridz = n_shots_h;
  dim3 dimGrid(gridx, gridy, gridz);

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources((n_sources_per_shot_h + dimBlock_sources.x - 1) /
                           dimBlock_sources.x,
                       n_shots_h, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers((n_receivers_per_shot_h + dimBlock_receivers.x - 1) /
                             dimBlock_receivers.x,
                         n_shots_h, 1);

  FILE *fp_ey = nullptr;
  FILE *fp_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad)
      fp_ey = fopen(ey_filenames[0], "rb");
    if (cb_requires_grad)
      fp_curl = fopen(curl_filenames[0], "rb");
  }

  size_t const adj_work_bytes =
      (size_t)n_shots_h * (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  TIDE_DTYPE *lambda_work_x = nullptr;
  TIDE_DTYPE *lambda_work_z = nullptr;
  gpuErrchk(cudaMalloc((void **)&lambda_work_x, adj_work_bytes));
  gpuErrchk(cudaMalloc((void **)&lambda_work_z, adj_work_bytes));

  auto store1_offset_bytes = [&](int64_t store_idx) -> size_t {
    if (storage_mode_h == STORAGE_DEVICE) {
      return (size_t)store_idx * bytes_per_step_store;
    }
    if (storage_mode_h == STORAGE_CPU) {
      return (size_t)(store_idx % NUM_BUFFERS) * bytes_per_step_store;
    }
    return 0;
  };

  auto store3_offset_bytes = [&](int64_t store_idx) -> size_t {
    return (storage_mode_h == STORAGE_CPU)
               ? (size_t)store_idx * bytes_per_step_store
               : 0;
  };

  auto prefetch_snapshots = [&](int64_t store_idx, bool want_ey,
                                bool want_curl) {
    if (storage_mode_h != STORAGE_CPU || (!want_ey && !want_curl)) {
      return;
    }
    int const store_buf = (int)(store_idx % NUM_BUFFERS);
    if (copy_in_flight[store_buf]) {
      gpuErrchk(cudaStreamWaitEvent(copy_stream, copy_done[store_buf], 0));
    }
    size_t const store1_offset = store1_offset_bytes(store_idx);
    size_t const store3_offset = store3_offset_bytes(store_idx);
    void *ey_store_1_t = (uint8_t *)ey_store_1 + store1_offset;
    void *curl_store_1_t = (uint8_t *)curl_store_1 + store1_offset;
    void *ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;
    void *curl_store_3_t = (uint8_t *)curl_store_3 + store3_offset;
    if (want_ey) {
      gpuErrchk(cudaMemcpyAsync(ey_store_1_t, ey_store_3_t,
                                bytes_per_step_store, cudaMemcpyHostToDevice,
                                copy_stream));
    }
    if (want_curl) {
      gpuErrchk(cudaMemcpyAsync(curl_store_1_t, curl_store_3_t,
                                bytes_per_step_store, cudaMemcpyHostToDevice,
                                copy_stream));
    }
    gpuErrchk(cudaEventRecord(copy_done[store_buf], copy_stream));
    copy_in_flight[store_buf] = true;
  };

  int64_t const t_min = start_t - nt;
  if (storage_mode_h == STORAGE_CPU && (ca_requires_grad || cb_requires_grad)) {
    int64_t t_prefetch = start_t - 1;
    int64_t const mod = t_prefetch % step_ratio_h;
    if (mod != 0)
      t_prefetch -= mod;
    if (t_prefetch >= t_min) {
      prefetch_snapshots(t_prefetch / step_ratio_h, ca_requires_grad,
                         cb_requires_grad);
    }
  }

  // Time reversed loop
  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    // Inject adjoint source (receiver residual) at receiver locations
    // Use add_adjoint_sources_ey which checks n_receivers_per_shot
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

    // Record adjoint field at source locations for source gradient
    // Use record_adjoint_at_sources which checks n_sources_per_shot
    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources<<<dimGrid_sources, dimBlock_sources>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, lambda_ey, sources_i);
    }

    int64_t const store_idx = t / step_ratio_h;
    bool const do_grad = (t % step_ratio_h) == 0;
    bool const grad_ey = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;

    size_t const store1_offset = store1_offset_bytes(store_idx);
    size_t const store3_offset = store3_offset_bytes(store_idx);

    void *__restrict const ey_store_1_t = (uint8_t *)ey_store_1 + store1_offset;
    void *__restrict const ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;

    void *__restrict const curl_store_1_t =
        (uint8_t *)curl_store_1 + store1_offset;
    void *__restrict const curl_store_3_t =
        (uint8_t *)curl_store_3 + store3_offset;

    if (storage_mode_h == STORAGE_CPU && (grad_ey || grad_curl)) {
      int const store_buf = (int)(store_idx % NUM_BUFFERS);
      if (!copy_in_flight[store_buf]) {
        prefetch_snapshots(store_idx, grad_ey, grad_curl);
      }
      gpuErrchk(cudaStreamWaitEvent(0, copy_done[store_buf], 0));
      copy_in_flight[store_buf] = false;
    } else if (storage_mode_h == STORAGE_DISK) {
      if (grad_ey) {
        storage_load_snapshot_gpu(
            (void *)ey_store_1_t, (void *)ey_store_3_t, fp_ey, storage_mode_h,
            store_idx, (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      }
      if (grad_curl) {
        storage_load_snapshot_gpu(
            (void *)curl_store_1_t, (void *)curl_store_3_t, fp_curl,
            storage_mode_h, store_idx, (size_t)shot_bytes_uncomp_h,
            (size_t)n_shots_h);
      }
    }

    gpuErrchk(cudaMemset(lambda_work_x, 0, adj_work_bytes));
    gpuErrchk(cudaMemset(lambda_work_z, 0, adj_work_bytes));
    backward_kernel_lambda_h_prepare_exact<<<dimGrid, dimBlock>>>(
        cb, lambda_ey, m_lambda_ey_x, m_lambda_ey_z, lambda_work_x,
        lambda_work_z, ay, ax, by, bx, ky, kx);
    backward_kernel_lambda_h_apply_exact<<<dimGrid, dimBlock>>>(
        lambda_work_x, lambda_work_z, lambda_hx, lambda_hz);

    gpuErrchk(cudaMemset(lambda_work_x, 0, adj_work_bytes));
    gpuErrchk(cudaMemset(lambda_work_z, 0, adj_work_bytes));
    backward_kernel_lambda_e_prepare_exact<<<dimGrid, dimBlock>>>(
        cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x, lambda_work_x,
        lambda_work_z, ayh, axh, byh, bxh, kyh, kxh);

    if (grad_ey || grad_curl) {
      if (storage_bf16_h) {
        backward_kernel_lambda_e_apply_exact_bf16<<<dimGrid, dimBlock>>>(
            ca, lambda_work_x, lambda_work_z, lambda_ey,
            grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
            grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
            grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
      } else {
        backward_kernel_lambda_e_apply_exact<<<dimGrid, dimBlock>>>(
            ca, lambda_work_x, lambda_work_z, lambda_ey,
            grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
            grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
            grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
      }
    } else {
      backward_kernel_lambda_e_apply_exact<<<dimGrid, dimBlock>>>(
          ca, lambda_work_x, lambda_work_z, lambda_ey, nullptr, nullptr,
          grad_ca_shot, grad_cb_shot, false, false, 1);
    }

    if (storage_mode_h == STORAGE_CPU && do_grad &&
        (ca_requires_grad || cb_requires_grad)) {
      int64_t const next_t = t - step_ratio_h;
      if (next_t >= t_min) {
        prefetch_snapshots(store_idx - 1, ca_requires_grad, cb_requires_grad);
      }
    }
  }

  if (storage_mode_h == STORAGE_CPU) {
    gpuErrchk(cudaStreamSynchronize(copy_stream));
    for (int i = 0; i < NUM_BUFFERS; i++) {
      gpuErrchk(cudaEventDestroy(copy_done[i]));
    }
    gpuErrchk(cudaStreamDestroy(copy_stream));
  }

  if (fp_ey != nullptr)
    fclose(fp_ey);
  if (fp_curl != nullptr)
    fclose(fp_curl);

  // Combine per-shot gradients (only if not batched - batched case keeps
  // per-shot grads)
  dim3 dimBlock_combine(32, 32, 1);
  dim3 dimGrid_combine(
      (nx_h - 2 * TIDE_FD_PAD + dimBlock_combine.x - 1) / dimBlock_combine.x,
      (ny_h - 2 * TIDE_FD_PAD + dimBlock_combine.y - 1) / dimBlock_combine.y, 1);

  if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_ca, grad_ca_shot);
  }
  if (cb_requires_grad && !cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_cb, grad_cb_shot);
  }

  gpuErrchk(cudaFree(lambda_work_x));
  gpuErrchk(cudaFree(lambda_work_z));

  gpuErrchk(cudaPeekAtLastError());
}

} // namespace FUNC(Inst)
#undef TIDE_FD_PAD
#undef LAMBDA_HX
#undef LAMBDA_HZ
