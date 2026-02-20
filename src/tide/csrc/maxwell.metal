/*
 * Maxwell wave equation propagator (Metal implementation)
 *
 * 2D TM Maxwell equations propagator for Apple GPU acceleration.
 */

#include <metal_stdlib>
using namespace metal;

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

// ============================================================================
// Staggered-grid finite-difference stencil operators
// ============================================================================

inline float diffy1_2(device const float* f, int i, int nx, float rdy) {
    return (f[i] - f[i - nx]) * rdy;
}
inline float diffy1_4(device const float* f, int i, int nx, float rdy) {
    return (9.0f/8.0f * (f[i] - f[i - nx])
          - 1.0f/24.0f * (f[i + nx] - f[i - 2*nx])) * rdy;
}
inline float diffy1_6(device const float* f, int i, int nx, float rdy) {
    return (75.0f/64.0f * (f[i] - f[i - nx])
          - 25.0f/384.0f * (f[i + nx] - f[i - 2*nx])
          + 3.0f/640.0f * (f[i + 2*nx] - f[i - 3*nx])) * rdy;
}
inline float diffy1_8(device const float* f, int i, int nx, float rdy) {
    return (1225.0f/1024.0f * (f[i] - f[i - nx])
          - 245.0f/3072.0f * (f[i + nx] - f[i - 2*nx])
          + 49.0f/5120.0f * (f[i + 2*nx] - f[i - 3*nx])
          - 5.0f/7168.0f * (f[i + 3*nx] - f[i - 4*nx])) * rdy;
}

inline float diffx1_2(device const float* f, int i, int nx, float rdx) {
    return (f[i] - f[i - 1]) * rdx;
}
inline float diffx1_4(device const float* f, int i, int nx, float rdx) {
    return (9.0f/8.0f * (f[i] - f[i - 1])
          - 1.0f/24.0f * (f[i + 1] - f[i - 2])) * rdx;
}
inline float diffx1_6(device const float* f, int i, int nx, float rdx) {
    return (75.0f/64.0f * (f[i] - f[i - 1])
          - 25.0f/384.0f * (f[i + 1] - f[i - 2])
          + 3.0f/640.0f * (f[i + 2] - f[i - 3])) * rdx;
}
inline float diffx1_8(device const float* f, int i, int nx, float rdx) {
    return (1225.0f/1024.0f * (f[i] - f[i - 1])
          - 245.0f/3072.0f * (f[i + 1] - f[i - 2])
          + 49.0f/5120.0f * (f[i + 2] - f[i - 3])
          - 5.0f/7168.0f * (f[i + 3] - f[i - 4])) * rdx;
}

inline float diffyh1_2(device const float* f, int i, int nx, float rdy) {
    return (f[i + nx] - f[i]) * rdy;
}
inline float diffyh1_4(device const float* f, int i, int nx, float rdy) {
    return (9.0f/8.0f * (f[i + nx] - f[i])
          - 1.0f/24.0f * (f[i + 2*nx] - f[i - nx])) * rdy;
}
inline float diffyh1_6(device const float* f, int i, int nx, float rdy) {
    return (75.0f/64.0f * (f[i + nx] - f[i])
          - 25.0f/384.0f * (f[i + 2*nx] - f[i - nx])
          + 3.0f/640.0f * (f[i + 3*nx] - f[i - 2*nx])) * rdy;
}
inline float diffyh1_8(device const float* f, int i, int nx, float rdy) {
    return (1225.0f/1024.0f * (f[i + nx] - f[i])
          - 245.0f/3072.0f * (f[i + 2*nx] - f[i - nx])
          + 49.0f/5120.0f * (f[i + 3*nx] - f[i - 2*nx])
          - 5.0f/7168.0f * (f[i + 4*nx] - f[i - 3*nx])) * rdy;
}

inline float diffxh1_2(device const float* f, int i, int nx, float rdx) {
    return (f[i + 1] - f[i]) * rdx;
}
inline float diffxh1_4(device const float* f, int i, int nx, float rdx) {
    return (9.0f/8.0f * (f[i + 1] - f[i])
          - 1.0f/24.0f * (f[i + 2] - f[i - 1])) * rdx;
}
inline float diffxh1_6(device const float* f, int i, int nx, float rdx) {
    return (75.0f/64.0f * (f[i + 1] - f[i])
          - 25.0f/384.0f * (f[i + 2] - f[i - 1])
          + 3.0f/640.0f * (f[i + 3] - f[i - 2])) * rdx;
}
inline float diffxh1_8(device const float* f, int i, int nx, float rdx) {
    return (1225.0f/1024.0f * (f[i + 1] - f[i])
          - 245.0f/3072.0f * (f[i + 2] - f[i - 1])
          + 49.0f/5120.0f * (f[i + 3] - f[i - 2])
          - 5.0f/7168.0f * (f[i + 4] - f[i - 3])) * rdx;
}

template<int FD_PAD>
inline float diffy1_fd(device const float* f, int i, int nx, float rdy);
template<int FD_PAD>
inline float diffx1_fd(device const float* f, int i, int nx, float rdx);
template<int FD_PAD>
inline float diffyh1_fd(device const float* f, int i, int nx, float rdy);
template<int FD_PAD>
inline float diffxh1_fd(device const float* f, int i, int nx, float rdx);

template<>
inline float diffy1_fd<1>(device const float* f, int i, int nx, float rdy) { return diffy1_2(f, i, nx, rdy); }
template<>
inline float diffy1_fd<2>(device const float* f, int i, int nx, float rdy) { return diffy1_4(f, i, nx, rdy); }
template<>
inline float diffy1_fd<3>(device const float* f, int i, int nx, float rdy) { return diffy1_6(f, i, nx, rdy); }
template<>
inline float diffy1_fd<4>(device const float* f, int i, int nx, float rdy) { return diffy1_8(f, i, nx, rdy); }

template<>
inline float diffx1_fd<1>(device const float* f, int i, int nx, float rdx) { return diffx1_2(f, i, nx, rdx); }
template<>
inline float diffx1_fd<2>(device const float* f, int i, int nx, float rdx) { return diffx1_4(f, i, nx, rdx); }
template<>
inline float diffx1_fd<3>(device const float* f, int i, int nx, float rdx) { return diffx1_6(f, i, nx, rdx); }
template<>
inline float diffx1_fd<4>(device const float* f, int i, int nx, float rdx) { return diffx1_8(f, i, nx, rdx); }

template<>
inline float diffyh1_fd<1>(device const float* f, int i, int nx, float rdy) { return diffyh1_2(f, i, nx, rdy); }
template<>
inline float diffyh1_fd<2>(device const float* f, int i, int nx, float rdy) { return diffyh1_4(f, i, nx, rdy); }
template<>
inline float diffyh1_fd<3>(device const float* f, int i, int nx, float rdy) { return diffyh1_6(f, i, nx, rdy); }
template<>
inline float diffyh1_fd<4>(device const float* f, int i, int nx, float rdy) { return diffyh1_8(f, i, nx, rdy); }

template<>
inline float diffxh1_fd<1>(device const float* f, int i, int nx, float rdx) { return diffxh1_2(f, i, nx, rdx); }
template<>
inline float diffxh1_fd<2>(device const float* f, int i, int nx, float rdx) { return diffxh1_4(f, i, nx, rdx); }
template<>
inline float diffxh1_fd<3>(device const float* f, int i, int nx, float rdx) { return diffxh1_6(f, i, nx, rdx); }
template<>
inline float diffxh1_fd<4>(device const float* f, int i, int nx, float rdx) { return diffxh1_8(f, i, nx, rdx); }

// ============================================================================
// Shared kernel implementations (templated by FD pad)
// ============================================================================

template<int FD_PAD>
inline void forward_kernel_h_impl(
    device const float* cq,
    device const float* ey,
    device float* hx,
    device float* hz,
    device float* m_ey_x,
    device float* m_ey_z,
    device const float* ay,
    device const float* ayh,
    device const float* ax,
    device const float* axh,
    device const float* by,
    device const float* byh,
    device const float* bx,
    device const float* bxh,
    device const float* ky,
    device const float* kyh,
    device const float* kx,
    device const float* kxh,
    constant GridParams& params,
    uint3 gid) {
    int const ny = (int)params.ny;
    int const nx = (int)params.nx;
    int const shot_numel = (int)params.shot_numel;
    int const n_shots = (int)params.n_shots;

    int const x = (int)gid.x + FD_PAD;
    int const y = (int)gid.y + FD_PAD;
    int const shot_idx = (int)gid.z;

    if (shot_idx >= n_shots) return;
    if (y >= ny - FD_PAD + 1) return;
    if (x >= nx - FD_PAD + 1) return;

    int const j = y * nx + x;
    int64_t const i = (int64_t)shot_idx * (int64_t)shot_numel + (int64_t)j;
    float const cq_val = params.cq_batched ? cq[i] : cq[j];

    int const pml_y0h = (int)params.pml_y0;
    int const pml_y1h = max((int)params.pml_y0, (int)params.pml_y1 - 1);
    int const pml_x0h = (int)params.pml_x0;
    int const pml_x1h = max((int)params.pml_x0, (int)params.pml_x1 - 1);

    if (y < ny - FD_PAD) {
        bool const pml_y_flag = y < pml_y0h || y >= pml_y1h;
        float dey_dz = diffyh1_fd<FD_PAD>(ey, (int)i, nx, params.rdy);
        if (pml_y_flag) {
            m_ey_z[i] = byh[y] * m_ey_z[i] + ayh[y] * dey_dz;
            dey_dz = dey_dz / kyh[y] + m_ey_z[i];
        }
        hx[i] -= cq_val * dey_dz;
    }

    if (x < nx - FD_PAD) {
        bool const pml_x_flag = x < pml_x0h || x >= pml_x1h;
        float dey_dx = diffxh1_fd<FD_PAD>(ey, (int)i, nx, params.rdx);
        if (pml_x_flag) {
            m_ey_x[i] = bxh[x] * m_ey_x[i] + axh[x] * dey_dx;
            dey_dx = dey_dx / kxh[x] + m_ey_x[i];
        }
        hz[i] += cq_val * dey_dx;
    }
}

template<int FD_PAD>
inline void forward_kernel_e_impl(
    device const float* ca,
    device const float* cb,
    device const float* hx,
    device const float* hz,
    device float* ey,
    device float* m_hx_z,
    device float* m_hz_x,
    device const float* ay,
    device const float* ayh,
    device const float* ax,
    device const float* axh,
    device const float* by,
    device const float* byh,
    device const float* bx,
    device const float* bxh,
    device const float* ky,
    device const float* kyh,
    device const float* kx,
    device const float* kxh,
    constant GridParams& params,
    uint3 gid) {
    int const ny = (int)params.ny;
    int const nx = (int)params.nx;
    int const shot_numel = (int)params.shot_numel;
    int const n_shots = (int)params.n_shots;

    int const x = (int)gid.x + FD_PAD;
    int const y = (int)gid.y + FD_PAD;
    int const shot_idx = (int)gid.z;

    if (shot_idx >= n_shots) return;
    if (y >= ny - FD_PAD + 1) return;
    if (x >= nx - FD_PAD + 1) return;

    int const j = y * nx + x;
    int64_t const i = (int64_t)shot_idx * (int64_t)shot_numel + (int64_t)j;

    float const ca_val = params.ca_batched ? ca[i] : ca[j];
    float const cb_val = params.cb_batched ? cb[i] : cb[j];

    bool const pml_y = y < (int)params.pml_y0 || y >= (int)params.pml_y1;
    bool const pml_x = x < (int)params.pml_x0 || x >= (int)params.pml_x1;

    float dhz_dx = diffx1_fd<FD_PAD>(hz, (int)i, nx, params.rdx);
    float dhx_dz = diffy1_fd<FD_PAD>(hx, (int)i, nx, params.rdy);

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

template<int FD_PAD>
inline void forward_kernel_e_with_storage_impl(
    device const float* ca,
    device const float* cb,
    device const float* hx,
    device const float* hz,
    device float* ey,
    device float* m_hx_z,
    device float* m_hz_x,
    device const float* ay,
    device const float* ayh,
    device const float* ax,
    device const float* axh,
    device const float* by,
    device const float* byh,
    device const float* bx,
    device const float* bxh,
    device const float* ky,
    device const float* kyh,
    device const float* kx,
    device const float* kxh,
    device float* ey_store,
    device float* curl_store,
    constant BackwardParams& params,
    uint3 gid) {
    int const ny = (int)params.ny;
    int const nx = (int)params.nx;
    int const shot_numel = (int)params.shot_numel;
    int const n_shots = (int)params.n_shots;

    int const x = (int)gid.x + FD_PAD;
    int const y = (int)gid.y + FD_PAD;
    int const shot_idx = (int)gid.z;

    if (shot_idx >= n_shots) return;
    if (y >= ny - FD_PAD + 1) return;
    if (x >= nx - FD_PAD + 1) return;

    int const j = y * nx + x;
    int64_t const i = (int64_t)shot_idx * (int64_t)shot_numel + (int64_t)j;
    int64_t const store_idx = params.store_offset + (int64_t)shot_idx * (int64_t)shot_numel + (int64_t)j;

    float const ca_val = params.ca_batched ? ca[i] : ca[j];
    float const cb_val = params.cb_batched ? cb[i] : cb[j];

    bool const pml_y = y < (int)params.pml_y0 || y >= (int)params.pml_y1;
    bool const pml_x = x < (int)params.pml_x0 || x >= (int)params.pml_x1;

    float dhz_dx = diffx1_fd<FD_PAD>(hz, (int)i, nx, params.rdx);
    float dhx_dz = diffy1_fd<FD_PAD>(hx, (int)i, nx, params.rdy);

    if (pml_x) {
        m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
        dhz_dx = dhz_dx / kx[x] + m_hz_x[i];
    }

    if (pml_y) {
        m_hx_z[i] = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
        dhx_dz = dhx_dz / ky[y] + m_hx_z[i];
    }

    float const curl_h = dhz_dx - dhx_dz;
    if (params.store_ey) {
        ey_store[store_idx] = ey[i];
    }
    if (params.store_curl) {
        curl_store[store_idx] = curl_h;
    }

    ey[i] = ca_val * ey[i] + cb_val * curl_h;
}

template<int FD_PAD>
inline void backward_kernel_lambda_h_impl(
    device const float* cb,
    device const float* lambda_ey,
    device float* lambda_hx,
    device float* lambda_hz,
    device float* m_lambda_ey_x,
    device float* m_lambda_ey_z,
    device const float* ay,
    device const float* ayh,
    device const float* ax,
    device const float* axh,
    device const float* by,
    device const float* byh,
    device const float* bx,
    device const float* bxh,
    device const float* ky,
    device const float* kyh,
    device const float* kx,
    device const float* kxh,
    constant BackwardParams& params,
    uint3 gid) {
    int const ny = (int)params.ny;
    int const nx = (int)params.nx;
    int const shot_numel = (int)params.shot_numel;
    int const n_shots = (int)params.n_shots;

    int const x = (int)gid.x + FD_PAD;
    int const y = (int)gid.y + FD_PAD;
    int const shot_idx = (int)gid.z;

    if (shot_idx >= n_shots) return;
    if (y >= ny - FD_PAD + 1) return;
    if (x >= nx - FD_PAD + 1) return;

    int const j = y * nx + x;
    int64_t const i = (int64_t)shot_idx * (int64_t)shot_numel + (int64_t)j;

    float const cb_val = params.cb_batched ? cb[i] : cb[j];

    int const pml_y0h = (int)params.pml_y0;
    int const pml_y1h = max((int)params.pml_y0, (int)params.pml_y1 - 1);
    int const pml_x0h = (int)params.pml_x0;
    int const pml_x1h = max((int)params.pml_x0, (int)params.pml_x1 - 1);

    if (y < ny - FD_PAD) {
        bool const pml_y_flag = y < pml_y0h || y >= pml_y1h;

        float d_lambda_ey_dz = diffyh1_fd<FD_PAD>(lambda_ey, (int)i, nx, params.rdy);

        if (pml_y_flag) {
            m_lambda_ey_z[i] = byh[y] * m_lambda_ey_z[i] + ayh[y] * d_lambda_ey_dz;
            d_lambda_ey_dz = d_lambda_ey_dz / kyh[y] + m_lambda_ey_z[i];
        }

        lambda_hx[i] -= cb_val * d_lambda_ey_dz;
    }

    if (x < nx - FD_PAD) {
        bool const pml_x_flag = x < pml_x0h || x >= pml_x1h;

        float d_lambda_ey_dx = diffxh1_fd<FD_PAD>(lambda_ey, (int)i, nx, params.rdx);

        if (pml_x_flag) {
            m_lambda_ey_x[i] = bxh[x] * m_lambda_ey_x[i] + axh[x] * d_lambda_ey_dx;
            d_lambda_ey_dx = d_lambda_ey_dx / kxh[x] + m_lambda_ey_x[i];
        }

        lambda_hz[i] += cb_val * d_lambda_ey_dx;
    }
}

template<int FD_PAD>
inline void backward_kernel_lambda_e_with_grad_impl(
    device const float* ca,
    device const float* cq,
    device const float* lambda_hx,
    device const float* lambda_hz,
    device float* lambda_ey,
    device float* m_lambda_hx_z,
    device float* m_lambda_hz_x,
    device const float* ey_store,
    device const float* curl_h_store,
    device float* grad_ca,
    device float* grad_cb,
    device const float* ay,
    device const float* ayh,
    device const float* ax,
    device const float* axh,
    device const float* by,
    device const float* byh,
    device const float* bx,
    device const float* bxh,
    device const float* ky,
    device const float* kyh,
    device const float* kx,
    device const float* kxh,
    constant BackwardParams& params,
    uint3 gid) {
    int const ny = (int)params.ny;
    int const nx = (int)params.nx;
    int const shot_numel = (int)params.shot_numel;
    int const n_shots = (int)params.n_shots;

    int const x = (int)gid.x + FD_PAD;
    int const y = (int)gid.y + FD_PAD;
    int const shot_idx = (int)gid.z;

    if (shot_idx >= n_shots) return;
    if (y >= ny - FD_PAD + 1) return;
    if (x >= nx - FD_PAD + 1) return;

    int const j = y * nx + x;
    int64_t const i = (int64_t)shot_idx * (int64_t)shot_numel + (int64_t)j;
    int64_t const store_idx = params.store_offset + (int64_t)shot_idx * (int64_t)shot_numel + (int64_t)j;

    float const ca_val = params.ca_batched ? ca[i] : ca[j];
    float const cq_val = params.cq_batched ? cq[i] : cq[j];

    bool const pml_y = y < (int)params.pml_y0 || y >= (int)params.pml_y1;
    bool const pml_x = x < (int)params.pml_x0 || x >= (int)params.pml_x1;

    float d_lambda_hz_dx = diffx1_fd<FD_PAD>(lambda_hz, (int)i, nx, params.rdx);
    float d_lambda_hx_dz = diffy1_fd<FD_PAD>(lambda_hx, (int)i, nx, params.rdy);

    if (pml_x) {
        m_lambda_hz_x[i] = bx[x] * m_lambda_hz_x[i] + ax[x] * d_lambda_hz_dx;
        d_lambda_hz_dx = d_lambda_hz_dx / kx[x] + m_lambda_hz_x[i];
    }
    if (pml_y) {
        m_lambda_hx_z[i] = by[y] * m_lambda_hx_z[i] + ay[y] * d_lambda_hx_dz;
        d_lambda_hx_dz = d_lambda_hx_dz / ky[y] + m_lambda_hx_z[i];
    }

    float const curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;
    float const lambda_ey_curr = lambda_ey[i];

    lambda_ey[i] = ca_val * lambda_ey_curr + cq_val * curl_lambda_h;

    if (!pml_y && !pml_x) {
        float const step_ratio_f = float(params.step_ratio);

        if (params.ca_requires_grad && params.store_ey) {
            float const ey_n = ey_store[store_idx];
            if (params.ca_batched) {
                grad_ca[i] += lambda_ey_curr * ey_n * step_ratio_f;
            } else {
                grad_ca[j] += lambda_ey_curr * ey_n * step_ratio_f;
            }
        }

        if (params.cb_requires_grad && params.store_curl) {
            float const curl_h_n = curl_h_store[store_idx];
            if (params.ca_batched) {
                grad_cb[i] += lambda_ey_curr * curl_h_n * step_ratio_f;
            } else {
                grad_cb[j] += lambda_ey_curr * curl_h_n * step_ratio_f;
            }
        }
    }
}

#define DEFINE_STENCIL_KERNELS(SUFFIX, FD_PAD) \
kernel void forward_kernel_h_##SUFFIX( \
    device const float* cq [[buffer(0)]], \
    device const float* ey [[buffer(1)]], \
    device float* hx [[buffer(2)]], \
    device float* hz [[buffer(3)]], \
    device float* m_ey_x [[buffer(4)]], \
    device float* m_ey_z [[buffer(5)]], \
    device const float* ay [[buffer(6)]], \
    device const float* ayh [[buffer(7)]], \
    device const float* ax [[buffer(8)]], \
    device const float* axh [[buffer(9)]], \
    device const float* by [[buffer(10)]], \
    device const float* byh [[buffer(11)]], \
    device const float* bx [[buffer(12)]], \
    device const float* bxh [[buffer(13)]], \
    device const float* ky [[buffer(14)]], \
    device const float* kyh [[buffer(15)]], \
    device const float* kx [[buffer(16)]], \
    device const float* kxh [[buffer(17)]], \
    constant GridParams& params [[buffer(18)]], \
    uint3 gid [[thread_position_in_grid]]) { \
    forward_kernel_h_impl<FD_PAD>(cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, params, gid); \
} \
kernel void forward_kernel_e_##SUFFIX( \
    device const float* ca [[buffer(0)]], \
    device const float* cb [[buffer(1)]], \
    device const float* hx [[buffer(2)]], \
    device const float* hz [[buffer(3)]], \
    device float* ey [[buffer(4)]], \
    device float* m_hx_z [[buffer(5)]], \
    device float* m_hz_x [[buffer(6)]], \
    device const float* ay [[buffer(7)]], \
    device const float* ayh [[buffer(8)]], \
    device const float* ax [[buffer(9)]], \
    device const float* axh [[buffer(10)]], \
    device const float* by [[buffer(11)]], \
    device const float* byh [[buffer(12)]], \
    device const float* bx [[buffer(13)]], \
    device const float* bxh [[buffer(14)]], \
    device const float* ky [[buffer(15)]], \
    device const float* kyh [[buffer(16)]], \
    device const float* kx [[buffer(17)]], \
    device const float* kxh [[buffer(18)]], \
    constant GridParams& params [[buffer(19)]], \
    uint3 gid [[thread_position_in_grid]]) { \
    forward_kernel_e_impl<FD_PAD>(ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, params, gid); \
} \
kernel void forward_kernel_e_with_storage_##SUFFIX( \
    device const float* ca [[buffer(0)]], \
    device const float* cb [[buffer(1)]], \
    device const float* hx [[buffer(2)]], \
    device const float* hz [[buffer(3)]], \
    device float* ey [[buffer(4)]], \
    device float* m_hx_z [[buffer(5)]], \
    device float* m_hz_x [[buffer(6)]], \
    device const float* ay [[buffer(7)]], \
    device const float* ayh [[buffer(8)]], \
    device const float* ax [[buffer(9)]], \
    device const float* axh [[buffer(10)]], \
    device const float* by [[buffer(11)]], \
    device const float* byh [[buffer(12)]], \
    device const float* bx [[buffer(13)]], \
    device const float* bxh [[buffer(14)]], \
    device const float* ky [[buffer(15)]], \
    device const float* kyh [[buffer(16)]], \
    device const float* kx [[buffer(17)]], \
    device const float* kxh [[buffer(18)]], \
    device float* ey_store [[buffer(19)]], \
    device float* curl_store [[buffer(20)]], \
    constant BackwardParams& params [[buffer(21)]], \
    uint3 gid [[thread_position_in_grid]]) { \
    forward_kernel_e_with_storage_impl<FD_PAD>(ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, ey_store, curl_store, params, gid); \
} \
kernel void backward_kernel_lambda_h_##SUFFIX( \
    device const float* cb [[buffer(0)]], \
    device const float* lambda_ey [[buffer(1)]], \
    device float* lambda_hx [[buffer(2)]], \
    device float* lambda_hz [[buffer(3)]], \
    device float* m_lambda_ey_x [[buffer(4)]], \
    device float* m_lambda_ey_z [[buffer(5)]], \
    device const float* ay [[buffer(6)]], \
    device const float* ayh [[buffer(7)]], \
    device const float* ax [[buffer(8)]], \
    device const float* axh [[buffer(9)]], \
    device const float* by [[buffer(10)]], \
    device const float* byh [[buffer(11)]], \
    device const float* bx [[buffer(12)]], \
    device const float* bxh [[buffer(13)]], \
    device const float* ky [[buffer(14)]], \
    device const float* kyh [[buffer(15)]], \
    device const float* kx [[buffer(16)]], \
    device const float* kxh [[buffer(17)]], \
    constant BackwardParams& params [[buffer(18)]], \
    uint3 gid [[thread_position_in_grid]]) { \
    backward_kernel_lambda_h_impl<FD_PAD>(cb, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x, m_lambda_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, params, gid); \
} \
kernel void backward_kernel_lambda_e_with_grad_##SUFFIX( \
    device const float* ca [[buffer(0)]], \
    device const float* cq [[buffer(1)]], \
    device const float* lambda_hx [[buffer(2)]], \
    device const float* lambda_hz [[buffer(3)]], \
    device float* lambda_ey [[buffer(4)]], \
    device float* m_lambda_hx_z [[buffer(5)]], \
    device float* m_lambda_hz_x [[buffer(6)]], \
    device const float* ey_store [[buffer(7)]], \
    device const float* curl_h_store [[buffer(8)]], \
    device float* grad_ca [[buffer(9)]], \
    device float* grad_cb [[buffer(10)]], \
    device const float* ay [[buffer(11)]], \
    device const float* ayh [[buffer(12)]], \
    device const float* ax [[buffer(13)]], \
    device const float* axh [[buffer(14)]], \
    device const float* by [[buffer(15)]], \
    device const float* byh [[buffer(16)]], \
    device const float* bx [[buffer(17)]], \
    device const float* bxh [[buffer(18)]], \
    device const float* ky [[buffer(19)]], \
    device const float* kyh [[buffer(20)]], \
    device const float* kx [[buffer(21)]], \
    device const float* kxh [[buffer(22)]], \
    constant BackwardParams& params [[buffer(23)]], \
    uint3 gid [[thread_position_in_grid]]) { \
    backward_kernel_lambda_e_with_grad_impl<FD_PAD>(ca, cq, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z, m_lambda_hz_x, ey_store, curl_h_store, grad_ca, grad_cb, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, params, gid); \
}

DEFINE_STENCIL_KERNELS(2, 1)
DEFINE_STENCIL_KERNELS(4, 2)
DEFINE_STENCIL_KERNELS(6, 3)
DEFINE_STENCIL_KERNELS(8, 4)

// ============================================================================
// Source and receiver kernels
// ============================================================================

kernel void add_sources_ey(
    device float* ey [[buffer(0)]],
    device const float* f [[buffer(1)]],
    device const int64_t* sources_i [[buffer(2)]],
    constant GridParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    int const source_idx = (int)gid.x;
    int const shot_idx = (int)gid.y;

    if (source_idx >= (int)params.n_sources_per_shot) return;
    if (shot_idx >= (int)params.n_shots) return;

    int64_t const k = (int64_t)shot_idx * params.n_sources_per_shot + (int64_t)source_idx;
    int64_t const src = sources_i[k];
    if (src >= 0) {
        ey[(int64_t)shot_idx * params.shot_numel + src] += f[k];
    }
}

kernel void record_receivers_ey(
    device float* r [[buffer(0)]],
    device const float* ey [[buffer(1)]],
    device const int64_t* receivers_i [[buffer(2)]],
    constant GridParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    int const receiver_idx = (int)gid.x;
    int const shot_idx = (int)gid.y;

    if (receiver_idx >= (int)params.n_receivers_per_shot) return;
    if (shot_idx >= (int)params.n_shots) return;

    int64_t const k = (int64_t)shot_idx * params.n_receivers_per_shot + (int64_t)receiver_idx;
    int64_t const rec = receivers_i[k];
    if (rec >= 0) {
        r[k] = ey[(int64_t)shot_idx * params.shot_numel + rec];
    }
}

// ============================================================================
// Convert grad_ca/grad_cb to grad_epsilon/grad_sigma
// ============================================================================

kernel void convert_grad_kernel(
    device const float* ca [[buffer(0)]],
    device const float* cb [[buffer(1)]],
    device const float* grad_ca [[buffer(2)]],
    device const float* grad_cb [[buffer(3)]],
    device float* grad_eps [[buffer(4)]],
    device float* grad_sigma [[buffer(5)]],
    constant BackwardParams& params [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    int const x = (int)gid.x;
    int const y = (int)gid.y;
    int const shot_idx = (int)gid.z;

    int const out_shots = params.ca_batched ? (int)params.n_shots : 1;
    if (shot_idx >= out_shots) return;
    if (y >= (int)params.ny) return;
    if (x >= (int)params.nx) return;

    int const nx = (int)params.nx;
    int const j = y * nx + x;
    int64_t const shot_offset = (int64_t)shot_idx * params.shot_numel;
    int64_t const out_idx = params.ca_batched ? (shot_offset + j) : (int64_t)j;
    int64_t const ca_idx = params.ca_batched ? (shot_offset + j) : (int64_t)j;
    int64_t const cb_idx = params.cb_batched ? (shot_offset + j) : (int64_t)j;

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

    float const EP0 = 8.85418782e-12f;

    if (grad_eps != nullptr) {
        grad_eps[out_idx] = (grad_ca_val * dca_de + grad_cb_val * dcb_de) * EP0;
    }
    if (grad_sigma != nullptr) {
        grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
    }
}
