namespace cg = cooperative_groups;

struct PersistentForwardParams3D {
  TIDE_DTYPE const *ca, *cb, *cq, *f;
  TIDE_DTYPE *ex, *ey, *ez, *hx, *hy, *hz;
  TIDE_DTYPE *m_hy_z, *m_hz_y, *m_hz_x, *m_hx_z, *m_hx_y, *m_hy_x;
  TIDE_DTYPE *m_ey_z, *m_ez_y, *m_ez_x, *m_ex_z, *m_ex_y, *m_ey_x;
  TIDE_DTYPE *r;
  TIDE_DTYPE const *az, *bz, *azh, *bzh, *ay, *by, *ayh, *byh;
  TIDE_DTYPE const *ax, *bx, *axh, *bxh, *kz, *kzh, *ky, *kyh, *kx, *kxh;
  int64_t const *sources_i, *receivers_i;
  int64_t start_t, nt;
  int source_component, receiver_component;
};

__device__ __forceinline__ LinearCellIndex3D persistent_index_3d(
    int64_t const i) {
  LinearCellIndex3D idx{};
  int64_t const local = i % shot_numel;
  idx.shot_idx = static_cast<int>(i / shot_numel);
  idx.j = static_cast<int>(local);
  idx.nz_i = static_cast<int>(nz);
  idx.ny_i = static_cast<int>(ny);
  idx.nx_i = static_cast<int>(nx);
  idx.z = static_cast<int>(local / (ny * nx));
  int64_t const rem = local - static_cast<int64_t>(idx.z) * ny * nx;
  idx.y = static_cast<int>(rem / nx);
  idx.x = static_cast<int>(rem - static_cast<int64_t>(idx.y) * nx);
  return idx;
}

__device__ __forceinline__ void persistent_update_h_3d(
    PersistentForwardParams3D const &p, int64_t const i,
    LinearCellIndex3D const &idx) {
  if (!is_active_cell_3d(idx)) return;
  int const j = idx.j, z = idx.z, y = idx.y, x = idx.x;
  TIDE_DTYPE const *ex_ptr=p.ex,*ey_ptr=p.ey,*ez_ptr=p.ez;
  TIDE_DTYPE *hx_ptr=p.hx,*hy_ptr=p.hy,*hz_ptr=p.hz;
#define ex ex_ptr
#define ey ey_ptr
#define ez ez_ptr
#define hx hx_ptr
#define hy hy_ptr
#define hz hz_ptr
#define EX_L(dz, dy, dx) EX(dz, dy, dx)
#define EY_L(dz, dy, dx) EY(dz, dy, dx)
#define EZ_L(dz, dy, dx) EZ(dz, dy, dx)
  TIDE_DTYPE const cq_val = cq_batched ? p.cq[i] : p.cq[j];
  int const z0 = static_cast<int>(pml_z0);
  int const z1 = tide_max(z0, static_cast<int>(pml_z1) - 1);
  int const y0 = static_cast<int>(pml_y0);
  int const y1 = tide_max(y0, static_cast<int>(pml_y1) - 1);
  int const x0 = static_cast<int>(pml_x0);
  int const x1 = tide_max(x0, static_cast<int>(pml_x1) - 1);
  TIDE_DTYPE a=0,b=0,c=0,d=0,e=0,f=0;
  if (z < idx.nz_i-FD_PAD) { a=DIFFZH1(EY_L); if(z<z0||z>=z1){p.m_ey_z[i]=p.bzh[z]*p.m_ey_z[i]+p.azh[z]*a;a=a/p.kzh[z]+p.m_ey_z[i];}}
  if (y < idx.ny_i-FD_PAD) { b=DIFFYH1(EZ_L); if(y<y0||y>=y1){p.m_ez_y[i]=p.byh[y]*p.m_ez_y[i]+p.ayh[y]*b;b=b/p.kyh[y]+p.m_ez_y[i];}}
  if (x < idx.nx_i-FD_PAD) { c=DIFFXH1(EZ_L); if(x<x0||x>=x1){p.m_ez_x[i]=p.bxh[x]*p.m_ez_x[i]+p.axh[x]*c;c=c/p.kxh[x]+p.m_ez_x[i];}}
  if (z < idx.nz_i-FD_PAD) { d=DIFFZH1(EX_L); if(z<z0||z>=z1){p.m_ex_z[i]=p.bzh[z]*p.m_ex_z[i]+p.azh[z]*d;d=d/p.kzh[z]+p.m_ex_z[i];}}
  if (y < idx.ny_i-FD_PAD) { e=DIFFYH1(EX_L); if(y<y0||y>=y1){p.m_ex_y[i]=p.byh[y]*p.m_ex_y[i]+p.ayh[y]*e;e=e/p.kyh[y]+p.m_ex_y[i];}}
  if (x < idx.nx_i-FD_PAD) { f=DIFFXH1(EY_L); if(x<x0||x>=x1){p.m_ey_x[i]=p.bxh[x]*p.m_ey_x[i]+p.axh[x]*f;f=f/p.kxh[x]+p.m_ey_x[i];}}
  hx[i]-=cq_val*(a-b); hy[i]-=cq_val*(c-d); hz[i]-=cq_val*(e-f);
#undef EX_L
#undef EY_L
#undef EZ_L
#undef ex
#undef ey
#undef ez
#undef hx
#undef hy
#undef hz
}

__device__ __forceinline__ void persistent_update_e_3d(
    PersistentForwardParams3D const &p, int64_t const i,
    LinearCellIndex3D const &idx) {
  if (!is_active_cell_3d(idx)) return;
  int const j=idx.j,z=idx.z,y=idx.y,x=idx.x;
  TIDE_DTYPE *ex_ptr=p.ex,*ey_ptr=p.ey,*ez_ptr=p.ez;
  TIDE_DTYPE const *hx_ptr=p.hx,*hy_ptr=p.hy,*hz_ptr=p.hz;
#define ex ex_ptr
#define ey ey_ptr
#define ez ez_ptr
#define hx hx_ptr
#define hy hy_ptr
#define hz hz_ptr
#define HX_L(dz, dy, dx) HX(dz, dy, dx)
#define HY_L(dz, dy, dx) HY(dz, dy, dx)
#define HZ_L(dz, dy, dx) HZ(dz, dy, dx)
  TIDE_DTYPE const av=ca_batched?p.ca[i]:p.ca[j], bv=cb_batched?p.cb[i]:p.cb[j];
  TIDE_DTYPE a=DIFFZ1(HY_L),b=DIFFY1(HZ_L),c=DIFFX1(HZ_L);
  TIDE_DTYPE d=DIFFZ1(HX_L),e=DIFFY1(HX_L),f=DIFFX1(HY_L);
  if(z<pml_z0||z>=pml_z1){p.m_hy_z[i]=p.bz[z]*p.m_hy_z[i]+p.az[z]*a;a=a/p.kz[z]+p.m_hy_z[i];p.m_hx_z[i]=p.bz[z]*p.m_hx_z[i]+p.az[z]*d;d=d/p.kz[z]+p.m_hx_z[i];}
  if(y<pml_y0||y>=pml_y1){p.m_hz_y[i]=p.by[y]*p.m_hz_y[i]+p.ay[y]*b;b=b/p.ky[y]+p.m_hz_y[i];p.m_hx_y[i]=p.by[y]*p.m_hx_y[i]+p.ay[y]*e;e=e/p.ky[y]+p.m_hx_y[i];}
  if(x<pml_x0||x>=pml_x1){p.m_hz_x[i]=p.bx[x]*p.m_hz_x[i]+p.ax[x]*c;c=c/p.kx[x]+p.m_hz_x[i];p.m_hy_x[i]=p.bx[x]*p.m_hy_x[i]+p.ax[x]*f;f=f/p.kx[x]+p.m_hy_x[i];}
  ex[i]=av*ex[i]+bv*(a-b);ey[i]=av*ey[i]+bv*(c-d);ez[i]=av*ez[i]+bv*(e-f);
#undef HX_L
#undef HY_L
#undef HZ_L
#undef ex
#undef ey
#undef ez
#undef hx
#undef hy
#undef hz
}

__global__ void persistent_forward_kernel_3d(PersistentForwardParams3D p) {
  cg::grid_group grid=cg::this_grid();
  int64_t const tid=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  int64_t const stride=(int64_t)gridDim.x*blockDim.x;
  int64_t const cells=n_shots*shot_numel;
  int64_t const ns=n_shots*n_sources_per_shot;
  int64_t const nr=n_shots*n_receivers_per_shot;
  for(int64_t step=0;step<p.nt;++step){
    for(int64_t i=tid;i<cells;i+=stride) persistent_update_h_3d(p,i,persistent_index_3d(i));
    grid.sync();
    for(int64_t i=tid;i<cells;i+=stride) persistent_update_e_3d(p,i,persistent_index_3d(i));
    grid.sync();
    TIDE_DTYPE *sf=p.source_component==0?p.ex:(p.source_component==2?p.ez:p.ey);
    if(p.f&&p.sources_i) for(int64_t q=tid;q<ns;q+=stride){int64_t s=p.sources_i[q];if(s>=0)sf[(q/n_sources_per_shot)*shot_numel+s]+=p.f[(p.start_t+step)*ns+q];}
    grid.sync();
    TIDE_DTYPE const *rf=p.receiver_component==0?p.ex:(p.receiver_component==2?p.ez:p.ey);
    if(p.r&&p.receivers_i) for(int64_t q=tid;q<nr;q+=stride){int64_t s=p.receivers_i[q];p.r[(p.start_t+step)*nr+q]=s>=0?rf[(q/n_receivers_per_shot)*shot_numel+s]:(TIDE_DTYPE)0;}
  }
}
