
#ifndef STAGGERED_GRID_H
#define STAGGERED_GRID_H

// Define TIDE_STAGGERED_GRID_3D=1 before including this header to select the
// 3D staggered-grid derivative macros; otherwise the default 2D set is used.
#if defined(TIDE_STAGGERED_GRID_3D) && TIDE_STAGGERED_GRID_3D

// FD_PAD is half the stencil width
#if TIDE_STENCIL == 2
#define FD_PAD 1
#elif TIDE_STENCIL == 4
#define FD_PAD 2
#elif TIDE_STENCIL == 6
#define FD_PAD 3
#elif TIDE_STENCIL == 8
#define FD_PAD 4
#else
#error "TIDE_STENCIL must be 2, 4, 6, or 8"
#endif

#if TIDE_STENCIL == 2
// 2nd order accuracy
#define DIFFZ1(F) ((F(0, 0, 0) - F(-1, 0, 0)) * rdz)
#define DIFFY1(F) ((F(0, 0, 0) - F(0, -1, 0)) * rdy)
#define DIFFX1(F) ((F(0, 0, 0) - F(0, 0, -1)) * rdx)
#define DIFFZH1(F) ((F(1, 0, 0) - F(0, 0, 0)) * rdz)
#define DIFFYH1(F) ((F(0, 1, 0) - F(0, 0, 0)) * rdy)
#define DIFFXH1(F) ((F(0, 0, 1) - F(0, 0, 0)) * rdx)

#elif TIDE_STENCIL == 4
// 4th order accuracy
#define DIFFZ1(F) \
  ((TIDE_DTYPE)(9.0/8.0) * (F(0, 0, 0) - F(-1, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (F(1, 0, 0) - F(-2, 0, 0))) * rdz

#define DIFFY1(F) \
  ((TIDE_DTYPE)(9.0/8.0) * (F(0, 0, 0) - F(0, -1, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (F(0, 1, 0) - F(0, -2, 0))) * rdy

#define DIFFX1(F) \
  ((TIDE_DTYPE)(9.0/8.0) * (F(0, 0, 0) - F(0, 0, -1)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (F(0, 0, 1) - F(0, 0, -2))) * rdx

#define DIFFZH1(F) \
  ((TIDE_DTYPE)(9.0/8.0) * (F(1, 0, 0) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (F(2, 0, 0) - F(-1, 0, 0))) * rdz

#define DIFFYH1(F) \
  ((TIDE_DTYPE)(9.0/8.0) * (F(0, 1, 0) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (F(0, 2, 0) - F(0, -1, 0))) * rdy

#define DIFFXH1(F) \
  ((TIDE_DTYPE)(9.0/8.0) * (F(0, 0, 1) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (F(0, 0, 2) - F(0, 0, -1))) * rdx

#elif TIDE_STENCIL == 6
// 6th order accuracy
#define DIFFZ1(F) \
  ((TIDE_DTYPE)(75.0/64.0) * (F(0, 0, 0) - F(-1, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (F(1, 0, 0) - F(-2, 0, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (F(2, 0, 0) - F(-3, 0, 0))) * rdz

#define DIFFY1(F) \
  ((TIDE_DTYPE)(75.0/64.0) * (F(0, 0, 0) - F(0, -1, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (F(0, 1, 0) - F(0, -2, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (F(0, 2, 0) - F(0, -3, 0))) * rdy

#define DIFFX1(F) \
  ((TIDE_DTYPE)(75.0/64.0) * (F(0, 0, 0) - F(0, 0, -1)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (F(0, 0, 1) - F(0, 0, -2)) \
   + (TIDE_DTYPE)(3.0/640.0) * (F(0, 0, 2) - F(0, 0, -3))) * rdx

#define DIFFZH1(F) \
  ((TIDE_DTYPE)(75.0/64.0) * (F(1, 0, 0) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (F(2, 0, 0) - F(-1, 0, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (F(3, 0, 0) - F(-2, 0, 0))) * rdz

#define DIFFYH1(F) \
  ((TIDE_DTYPE)(75.0/64.0) * (F(0, 1, 0) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (F(0, 2, 0) - F(0, -1, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (F(0, 3, 0) - F(0, -2, 0))) * rdy

#define DIFFXH1(F) \
  ((TIDE_DTYPE)(75.0/64.0) * (F(0, 0, 1) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (F(0, 0, 2) - F(0, 0, -1)) \
   + (TIDE_DTYPE)(3.0/640.0) * (F(0, 0, 3) - F(0, 0, -2))) * rdx

#elif TIDE_STENCIL == 8
// 8th order accuracy
#define DIFFZ1(F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (F(0, 0, 0) - F(-1, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (F(1, 0, 0) - F(-2, 0, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (F(2, 0, 0) - F(-3, 0, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (F(3, 0, 0) - F(-4, 0, 0))) * rdz

#define DIFFY1(F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (F(0, 0, 0) - F(0, -1, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (F(0, 1, 0) - F(0, -2, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (F(0, 2, 0) - F(0, -3, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (F(0, 3, 0) - F(0, -4, 0))) * rdy

#define DIFFX1(F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (F(0, 0, 0) - F(0, 0, -1)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (F(0, 0, 1) - F(0, 0, -2)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (F(0, 0, 2) - F(0, 0, -3)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (F(0, 0, 3) - F(0, 0, -4))) * rdx

#define DIFFZH1(F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (F(1, 0, 0) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (F(2, 0, 0) - F(-1, 0, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (F(3, 0, 0) - F(-2, 0, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (F(4, 0, 0) - F(-3, 0, 0))) * rdz

#define DIFFYH1(F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (F(0, 1, 0) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (F(0, 2, 0) - F(0, -1, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (F(0, 3, 0) - F(0, -2, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (F(0, 4, 0) - F(0, -3, 0))) * rdy

#define DIFFXH1(F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (F(0, 0, 1) - F(0, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (F(0, 0, 2) - F(0, 0, -1)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (F(0, 0, 3) - F(0, 0, -2)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (F(0, 0, 4) - F(0, 0, -3))) * rdx

#endif

/*
 * Adjoint derivative operators for backward pass (3D scaffold).
 * These are the transpose operators of the forward staggered derivatives.
 */

#if TIDE_STENCIL == 2
#define DIFFZ1_ADJ(C, F) \
  ((C(0, 0, 0) * F(0, 0, 0) - C(1, 0, 0) * F(1, 0, 0)) * rdz)
#define DIFFY1_ADJ(C, F) \
  ((C(0, 0, 0) * F(0, 0, 0) - C(0, 1, 0) * F(0, 1, 0)) * rdy)
#define DIFFX1_ADJ(C, F) \
  ((C(0, 0, 0) * F(0, 0, 0) - C(0, 0, 1) * F(0, 0, 1)) * rdx)
#define DIFFZH1_ADJ(C, F) \
  ((C(-1, 0, 0) * F(-1, 0, 0) - C(0, 0, 0) * F(0, 0, 0)) * rdz)
#define DIFFYH1_ADJ(C, F) \
  ((C(0, -1, 0) * F(0, -1, 0) - C(0, 0, 0) * F(0, 0, 0)) * rdy)
#define DIFFXH1_ADJ(C, F) \
  ((C(0, 0, -1) * F(0, 0, -1) - C(0, 0, 0) * F(0, 0, 0)) * rdx)

#elif TIDE_STENCIL == 4
#define DIFFZ1_ADJ(C, F) \
  ((TIDE_DTYPE)(9.0/8.0) * (C(0, 0, 0) * F(0, 0, 0) - C(1, 0, 0) * F(1, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (C(-1, 0, 0) * F(-1, 0, 0) - C(2, 0, 0) * F(2, 0, 0))) * rdz
#define DIFFY1_ADJ(C, F) \
  ((TIDE_DTYPE)(9.0/8.0) * (C(0, 0, 0) * F(0, 0, 0) - C(0, 1, 0) * F(0, 1, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (C(0, -1, 0) * F(0, -1, 0) - C(0, 2, 0) * F(0, 2, 0))) * rdy
#define DIFFX1_ADJ(C, F) \
  ((TIDE_DTYPE)(9.0/8.0) * (C(0, 0, 0) * F(0, 0, 0) - C(0, 0, 1) * F(0, 0, 1)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (C(0, 0, -1) * F(0, 0, -1) - C(0, 0, 2) * F(0, 0, 2))) * rdx
#define DIFFZH1_ADJ(C, F) \
  ((TIDE_DTYPE)(9.0/8.0) * (C(-1, 0, 0) * F(-1, 0, 0) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (C(-2, 0, 0) * F(-2, 0, 0) - C(1, 0, 0) * F(1, 0, 0))) * rdz
#define DIFFYH1_ADJ(C, F) \
  ((TIDE_DTYPE)(9.0/8.0) * (C(0, -1, 0) * F(0, -1, 0) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (C(0, -2, 0) * F(0, -2, 0) - C(0, 1, 0) * F(0, 1, 0))) * rdy
#define DIFFXH1_ADJ(C, F) \
  ((TIDE_DTYPE)(9.0/8.0) * (C(0, 0, -1) * F(0, 0, -1) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-1.0/24.0) * (C(0, 0, -2) * F(0, 0, -2) - C(0, 0, 1) * F(0, 0, 1))) * rdx

#elif TIDE_STENCIL == 6
#define DIFFZ1_ADJ(C, F) \
  ((TIDE_DTYPE)(75.0/64.0) * (C(0, 0, 0) * F(0, 0, 0) - C(1, 0, 0) * F(1, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (C(-1, 0, 0) * F(-1, 0, 0) - C(2, 0, 0) * F(2, 0, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (C(-2, 0, 0) * F(-2, 0, 0) - C(3, 0, 0) * F(3, 0, 0))) * rdz
#define DIFFY1_ADJ(C, F) \
  ((TIDE_DTYPE)(75.0/64.0) * (C(0, 0, 0) * F(0, 0, 0) - C(0, 1, 0) * F(0, 1, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (C(0, -1, 0) * F(0, -1, 0) - C(0, 2, 0) * F(0, 2, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (C(0, -2, 0) * F(0, -2, 0) - C(0, 3, 0) * F(0, 3, 0))) * rdy
#define DIFFX1_ADJ(C, F) \
  ((TIDE_DTYPE)(75.0/64.0) * (C(0, 0, 0) * F(0, 0, 0) - C(0, 0, 1) * F(0, 0, 1)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (C(0, 0, -1) * F(0, 0, -1) - C(0, 0, 2) * F(0, 0, 2)) \
   + (TIDE_DTYPE)(3.0/640.0) * (C(0, 0, -2) * F(0, 0, -2) - C(0, 0, 3) * F(0, 0, 3))) * rdx
#define DIFFZH1_ADJ(C, F) \
  ((TIDE_DTYPE)(75.0/64.0) * (C(-1, 0, 0) * F(-1, 0, 0) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (C(-2, 0, 0) * F(-2, 0, 0) - C(1, 0, 0) * F(1, 0, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (C(-3, 0, 0) * F(-3, 0, 0) - C(2, 0, 0) * F(2, 0, 0))) * rdz
#define DIFFYH1_ADJ(C, F) \
  ((TIDE_DTYPE)(75.0/64.0) * (C(0, -1, 0) * F(0, -1, 0) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (C(0, -2, 0) * F(0, -2, 0) - C(0, 1, 0) * F(0, 1, 0)) \
   + (TIDE_DTYPE)(3.0/640.0) * (C(0, -3, 0) * F(0, -3, 0) - C(0, 2, 0) * F(0, 2, 0))) * rdy
#define DIFFXH1_ADJ(C, F) \
  ((TIDE_DTYPE)(75.0/64.0) * (C(0, 0, -1) * F(0, 0, -1) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-25.0/384.0) * (C(0, 0, -2) * F(0, 0, -2) - C(0, 0, 1) * F(0, 0, 1)) \
   + (TIDE_DTYPE)(3.0/640.0) * (C(0, 0, -3) * F(0, 0, -3) - C(0, 0, 2) * F(0, 0, 2))) * rdx

#elif TIDE_STENCIL == 8
#define DIFFZ1_ADJ(C, F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (C(0, 0, 0) * F(0, 0, 0) - C(1, 0, 0) * F(1, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (C(-1, 0, 0) * F(-1, 0, 0) - C(2, 0, 0) * F(2, 0, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (C(-2, 0, 0) * F(-2, 0, 0) - C(3, 0, 0) * F(3, 0, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (C(-3, 0, 0) * F(-3, 0, 0) - C(4, 0, 0) * F(4, 0, 0))) * rdz
#define DIFFY1_ADJ(C, F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (C(0, 0, 0) * F(0, 0, 0) - C(0, 1, 0) * F(0, 1, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (C(0, -1, 0) * F(0, -1, 0) - C(0, 2, 0) * F(0, 2, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (C(0, -2, 0) * F(0, -2, 0) - C(0, 3, 0) * F(0, 3, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (C(0, -3, 0) * F(0, -3, 0) - C(0, 4, 0) * F(0, 4, 0))) * rdy
#define DIFFX1_ADJ(C, F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (C(0, 0, 0) * F(0, 0, 0) - C(0, 0, 1) * F(0, 0, 1)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (C(0, 0, -1) * F(0, 0, -1) - C(0, 0, 2) * F(0, 0, 2)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (C(0, 0, -2) * F(0, 0, -2) - C(0, 0, 3) * F(0, 0, 3)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (C(0, 0, -3) * F(0, 0, -3) - C(0, 0, 4) * F(0, 0, 4))) * rdx
#define DIFFZH1_ADJ(C, F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (C(-1, 0, 0) * F(-1, 0, 0) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (C(-2, 0, 0) * F(-2, 0, 0) - C(1, 0, 0) * F(1, 0, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (C(-3, 0, 0) * F(-3, 0, 0) - C(2, 0, 0) * F(2, 0, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (C(-4, 0, 0) * F(-4, 0, 0) - C(3, 0, 0) * F(3, 0, 0))) * rdz
#define DIFFYH1_ADJ(C, F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (C(0, -1, 0) * F(0, -1, 0) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (C(0, -2, 0) * F(0, -2, 0) - C(0, 1, 0) * F(0, 1, 0)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (C(0, -3, 0) * F(0, -3, 0) - C(0, 2, 0) * F(0, 2, 0)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (C(0, -4, 0) * F(0, -4, 0) - C(0, 3, 0) * F(0, 3, 0))) * rdy
#define DIFFXH1_ADJ(C, F) \
  ((TIDE_DTYPE)(1225.0/1024.0) * (C(0, 0, -1) * F(0, 0, -1) - C(0, 0, 0) * F(0, 0, 0)) \
   + (TIDE_DTYPE)(-245.0/3072.0) * (C(0, 0, -2) * F(0, 0, -2) - C(0, 0, 1) * F(0, 0, 1)) \
   + (TIDE_DTYPE)(49.0/5120.0) * (C(0, 0, -3) * F(0, 0, -3) - C(0, 0, 2) * F(0, 0, 2)) \
   + (TIDE_DTYPE)(-5.0/7168.0) * (C(0, 0, -4) * F(0, 0, -4) - C(0, 0, 3) * F(0, 0, 3))) * rdx

#endif


#else

#if TIDE_STENCIL == 2
// 2nd order accuracy
// Note: explicit (TIDE_DTYPE) cast on rdy/rdx avoids ambiguous operator* when
// TIDE_DTYPE=half and rdy/rdx are float (MSVC host compiler rejects
// half*float).
#define DIFFY1(F) ((F(0, 0) - F(-1, 0)) * (TIDE_DTYPE)rdy)
#define DIFFX1(F) ((F(0, 0) - F(0, -1)) * (TIDE_DTYPE)rdx)
#define DIFFYH1(F) ((F(1, 0) - F(0, 0)) * (TIDE_DTYPE)rdy)
#define DIFFXH1(F) ((F(0, 1) - F(0, 0)) * (TIDE_DTYPE)rdx)

#elif TIDE_STENCIL == 4
// 4th order accuracy
#define DIFFY1(F)                                                              \
  ((TIDE_DTYPE)(9.0 / 8.0) * (F(0, 0) - F(-1, 0)) +                            \
   (TIDE_DTYPE)(-1.0 / 24.0) * (F(1, 0) - F(-2, 0))) *                         \
      (TIDE_DTYPE)rdy

#define DIFFX1(F)                                                              \
  ((TIDE_DTYPE)(9.0 / 8.0) * (F(0, 0) - F(0, -1)) +                            \
   (TIDE_DTYPE)(-1.0 / 24.0) * (F(0, 1) - F(0, -2))) *                         \
      (TIDE_DTYPE)rdx

#define DIFFYH1(F)                                                             \
  ((TIDE_DTYPE)(9.0 / 8.0) * (F(1, 0) - F(0, 0)) +                             \
   (TIDE_DTYPE)(-1.0 / 24.0) * (F(2, 0) - F(-1, 0))) *                         \
      (TIDE_DTYPE)rdy

#define DIFFXH1(F)                                                             \
  ((TIDE_DTYPE)(9.0 / 8.0) * (F(0, 1) - F(0, 0)) +                             \
   (TIDE_DTYPE)(-1.0 / 24.0) * (F(0, 2) - F(0, -1))) *                         \
      (TIDE_DTYPE)rdx

#elif TIDE_STENCIL == 6
// 6th order accuracy
#define DIFFY1(F)                                                              \
  ((TIDE_DTYPE)(75.0 / 64.0) * (F(0, 0) - F(-1, 0)) +                          \
   (TIDE_DTYPE)(-25.0 / 384.0) * (F(1, 0) - F(-2, 0)) +                        \
   (TIDE_DTYPE)(3.0 / 640.0) * (F(2, 0) - F(-3, 0))) *                         \
      (TIDE_DTYPE)rdy

#define DIFFX1(F)                                                              \
  ((TIDE_DTYPE)(75.0 / 64.0) * (F(0, 0) - F(0, -1)) +                          \
   (TIDE_DTYPE)(-25.0 / 384.0) * (F(0, 1) - F(0, -2)) +                        \
   (TIDE_DTYPE)(3.0 / 640.0) * (F(0, 2) - F(0, -3))) *                         \
      (TIDE_DTYPE)rdx

#define DIFFYH1(F)                                                             \
  ((TIDE_DTYPE)(75.0 / 64.0) * (F(1, 0) - F(0, 0)) +                           \
   (TIDE_DTYPE)(-25.0 / 384.0) * (F(2, 0) - F(-1, 0)) +                        \
   (TIDE_DTYPE)(3.0 / 640.0) * (F(3, 0) - F(-2, 0))) *                         \
      (TIDE_DTYPE)rdy

#define DIFFXH1(F)                                                             \
  ((TIDE_DTYPE)(75.0 / 64.0) * (F(0, 1) - F(0, 0)) +                           \
   (TIDE_DTYPE)(-25.0 / 384.0) * (F(0, 2) - F(0, -1)) +                        \
   (TIDE_DTYPE)(3.0 / 640.0) * (F(0, 3) - F(0, -2))) *                         \
      (TIDE_DTYPE)rdx

#elif TIDE_STENCIL == 8
// 8th order accuracy
#define DIFFY1(F)                                                              \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (F(0, 0) - F(-1, 0)) +                      \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (F(1, 0) - F(-2, 0)) +                      \
   (TIDE_DTYPE)(49.0 / 5120.0) * (F(2, 0) - F(-3, 0)) +                        \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (F(3, 0) - F(-4, 0))) *                       \
      (TIDE_DTYPE)rdy

#define DIFFX1(F)                                                              \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (F(0, 0) - F(0, -1)) +                      \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (F(0, 1) - F(0, -2)) +                      \
   (TIDE_DTYPE)(49.0 / 5120.0) * (F(0, 2) - F(0, -3)) +                        \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (F(0, 3) - F(0, -4))) *                       \
      (TIDE_DTYPE)rdx

#define DIFFYH1(F)                                                             \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (F(1, 0) - F(0, 0)) +                       \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (F(2, 0) - F(-1, 0)) +                      \
   (TIDE_DTYPE)(49.0 / 5120.0) * (F(3, 0) - F(-2, 0)) +                        \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (F(4, 0) - F(-3, 0))) *                       \
      (TIDE_DTYPE)rdy

#define DIFFXH1(F)                                                             \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (F(0, 1) - F(0, 0)) +                       \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (F(0, 2) - F(0, -1)) +                      \
   (TIDE_DTYPE)(49.0 / 5120.0) * (F(0, 3) - F(0, -2)) +                        \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (F(0, 4) - F(0, -3))) *                       \
      (TIDE_DTYPE)rdx

#endif

/*
 * Adjoint derivative operators for backward pass
 * These compute the transpose of the forward derivative operators
 */

#if TIDE_STENCIL == 2
#define DIFFY1_ADJ(C, F)                                                       \
  ((C(0, 0) * F(0, 0) - C(1, 0) * F(1, 0)) * (TIDE_DTYPE)rdy)

#define DIFFX1_ADJ(C, F)                                                       \
  ((C(0, 0) * F(0, 0) - C(0, 1) * F(0, 1)) * (TIDE_DTYPE)rdx)

#define DIFFYH1_ADJ(C, F)                                                      \
  ((C(-1, 0) * F(-1, 0) - C(0, 0) * F(0, 0)) * (TIDE_DTYPE)rdy)

#define DIFFXH1_ADJ(C, F)                                                      \
  ((C(0, -1) * F(0, -1) - C(0, 0) * F(0, 0)) * (TIDE_DTYPE)rdx)

#elif TIDE_STENCIL == 4
#define DIFFY1_ADJ(C, F)                                                       \
  ((TIDE_DTYPE)(9.0 / 8.0) * (C(0, 0) * F(0, 0) - C(1, 0) * F(1, 0)) +         \
   (TIDE_DTYPE)(-1.0 / 24.0) * (C(-1, 0) * F(-1, 0) - C(2, 0) * F(2, 0))) *    \
      (TIDE_DTYPE)rdy

#define DIFFX1_ADJ(C, F)                                                       \
  ((TIDE_DTYPE)(9.0 / 8.0) * (C(0, 0) * F(0, 0) - C(0, 1) * F(0, 1)) +         \
   (TIDE_DTYPE)(-1.0 / 24.0) * (C(0, -1) * F(0, -1) - C(0, 2) * F(0, 2))) *    \
      (TIDE_DTYPE)rdx

#define DIFFYH1_ADJ(C, F)                                                      \
  ((TIDE_DTYPE)(9.0 / 8.0) * (C(-1, 0) * F(-1, 0) - C(0, 0) * F(0, 0)) +       \
   (TIDE_DTYPE)(-1.0 / 24.0) * (C(-2, 0) * F(-2, 0) - C(1, 0) * F(1, 0))) *    \
      (TIDE_DTYPE)rdy

#define DIFFXH1_ADJ(C, F)                                                      \
  ((TIDE_DTYPE)(9.0 / 8.0) * (C(0, -1) * F(0, -1) - C(0, 0) * F(0, 0)) +       \
   (TIDE_DTYPE)(-1.0 / 24.0) * (C(0, -2) * F(0, -2) - C(0, 1) * F(0, 1))) *    \
      (TIDE_DTYPE)rdx

#elif TIDE_STENCIL == 6
#define DIFFY1_ADJ(C, F)                                                       \
  ((TIDE_DTYPE)(75.0 / 64.0) * (C(0, 0) * F(0, 0) - C(1, 0) * F(1, 0)) +       \
   (TIDE_DTYPE)(-25.0 / 384.0) * (C(-1, 0) * F(-1, 0) - C(2, 0) * F(2, 0)) +   \
   (TIDE_DTYPE)(3.0 / 640.0) * (C(-2, 0) * F(-2, 0) - C(3, 0) * F(3, 0))) *    \
      (TIDE_DTYPE)rdy

#define DIFFX1_ADJ(C, F)                                                       \
  ((TIDE_DTYPE)(75.0 / 64.0) * (C(0, 0) * F(0, 0) - C(0, 1) * F(0, 1)) +       \
   (TIDE_DTYPE)(-25.0 / 384.0) * (C(0, -1) * F(0, -1) - C(0, 2) * F(0, 2)) +   \
   (TIDE_DTYPE)(3.0 / 640.0) * (C(0, -2) * F(0, -2) - C(0, 3) * F(0, 3))) *    \
      (TIDE_DTYPE)rdx

#define DIFFYH1_ADJ(C, F)                                                      \
  ((TIDE_DTYPE)(75.0 / 64.0) * (C(-1, 0) * F(-1, 0) - C(0, 0) * F(0, 0)) +     \
   (TIDE_DTYPE)(-25.0 / 384.0) * (C(-2, 0) * F(-2, 0) - C(1, 0) * F(1, 0)) +   \
   (TIDE_DTYPE)(3.0 / 640.0) * (C(-3, 0) * F(-3, 0) - C(2, 0) * F(2, 0))) *    \
      (TIDE_DTYPE)rdy

#define DIFFXH1_ADJ(C, F)                                                      \
  ((TIDE_DTYPE)(75.0 / 64.0) * (C(0, -1) * F(0, -1) - C(0, 0) * F(0, 0)) +     \
   (TIDE_DTYPE)(-25.0 / 384.0) * (C(0, -2) * F(0, -2) - C(0, 1) * F(0, 1)) +   \
   (TIDE_DTYPE)(3.0 / 640.0) * (C(0, -3) * F(0, -3) - C(0, 2) * F(0, 2))) *    \
      (TIDE_DTYPE)rdx

#elif TIDE_STENCIL == 8
#define DIFFY1_ADJ(C, F)                                                       \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (C(0, 0) * F(0, 0) - C(1, 0) * F(1, 0)) +   \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (C(-1, 0) * F(-1, 0) - C(2, 0) * F(2, 0)) + \
   (TIDE_DTYPE)(49.0 / 5120.0) * (C(-2, 0) * F(-2, 0) - C(3, 0) * F(3, 0)) +   \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (C(-3, 0) * F(-3, 0) - C(4, 0) * F(4, 0))) *  \
      (TIDE_DTYPE)rdy

#define DIFFX1_ADJ(C, F)                                                       \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (C(0, 0) * F(0, 0) - C(0, 1) * F(0, 1)) +   \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (C(0, -1) * F(0, -1) - C(0, 2) * F(0, 2)) + \
   (TIDE_DTYPE)(49.0 / 5120.0) * (C(0, -2) * F(0, -2) - C(0, 3) * F(0, 3)) +   \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (C(0, -3) * F(0, -3) - C(0, 4) * F(0, 4))) *  \
      (TIDE_DTYPE)rdx

#define DIFFYH1_ADJ(C, F)                                                      \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (C(-1, 0) * F(-1, 0) - C(0, 0) * F(0, 0)) + \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (C(-2, 0) * F(-2, 0) - C(1, 0) * F(1, 0)) + \
   (TIDE_DTYPE)(49.0 / 5120.0) * (C(-3, 0) * F(-3, 0) - C(2, 0) * F(2, 0)) +   \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (C(-4, 0) * F(-4, 0) - C(3, 0) * F(3, 0))) *  \
      (TIDE_DTYPE)rdy

#define DIFFXH1_ADJ(C, F)                                                      \
  ((TIDE_DTYPE)(1225.0 / 1024.0) * (C(0, -1) * F(0, -1) - C(0, 0) * F(0, 0)) + \
   (TIDE_DTYPE)(-245.0 / 3072.0) * (C(0, -2) * F(0, -2) - C(0, 1) * F(0, 1)) + \
   (TIDE_DTYPE)(49.0 / 5120.0) * (C(0, -3) * F(0, -3) - C(0, 2) * F(0, 2)) +   \
   (TIDE_DTYPE)(-5.0 / 7168.0) * (C(0, -4) * F(0, -4) - C(0, 3) * F(0, 3))) *  \
      (TIDE_DTYPE)rdx

#endif


#endif

#endif // STAGGERED_GRID_H
