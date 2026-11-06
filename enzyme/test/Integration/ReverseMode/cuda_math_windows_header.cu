// The clang plugin keeps every wrapper in clang's device math header alive so
// the libdevice function a derivative needs is still declared when the program
// itself never called it: d/dx pow(x, x) uses log, d/dx sin uses cos and d/dx
// sinh uses cosh. The header is located by file name, which must also match the
// backslash paths a Windows install reports. Without the wrappers the
// derivative falls back to llvm.log.f64 and friends, for which NVPTX has no
// libcall. Stand-ins for the header and libdevice avoid needing a CUDA install.

// RUN: %clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | FileCheck %s
// RUN: %clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_70 -nocudainc -nocudalib -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | llc -march=nvptx64 -mcpu=sm_70 -o - | FileCheck %s --check-prefix=PTX

using Kernel = void (*)(double *, double *);
extern "C" __attribute__((device)) double __nv_pow(double, double);
extern "C" __attribute__((device)) double __nv_log(double);
extern "C" __attribute__((device)) double __nv_sin(double);
extern "C" __attribute__((device)) double __nv_cos(double);
extern "C" __attribute__((device)) double __nv_sinh(double);
extern "C" __attribute__((device)) double __nv_cosh(double);

extern __attribute__((device)) void __enzyme_autodiff(Kernel, double *,
                                                      double *, double *,
                                                      double *);

#line 1 "C:\\Program Files\\LLVM\\lib\\clang\\19\\include\\__clang_cuda_math.h"
static __attribute__((device, always_inline)) double pow(double a, double b) {
  return __nv_pow(a, b);
}
static __attribute__((device, always_inline)) double log(double a) {
  return __nv_log(a);
}
static __attribute__((device, always_inline)) double sin(double a) {
  return __nv_sin(a);
}
static __attribute__((device, always_inline)) double cos(double a) {
  return __nv_cos(a);
}
static __attribute__((device, always_inline)) double sinh(double a) {
  return __nv_sinh(a);
}
static __attribute__((device, always_inline)) double cosh(double a) {
  return __nv_cosh(a);
}
#line 40 "cuda_math_windows_header.cu"

__attribute__((device)) void residual(double *x_in, double *x_out) {
  x_out[0] = x_in[0] * pow(x_in[0], x_in[0]) + sin(x_in[0]) + sinh(x_in[0]);
}

__attribute__((global)) void residual_grad(double *x, double *d_x, double *y,
                                           double *d_y) {
  __enzyme_autodiff(residual, x, d_x, y, d_y);
}

// CHECK-LABEL: define dso_local void @_Z13residual_gradPdS_S_S_(
// CHECK-NOT: @llvm.log.f64
// CHECK-NOT: @llvm.cos.f64
// CHECK-NOT: @cosh(
// CHECK: call fast double @__nv_cosh(
// CHECK: call fast double @__nv_cos(
// CHECK: call fast double @__nv_log(

// PTX: .extern .func (.param .b64 func_retval0) __nv_log
// PTX: .extern .func (.param .b64 func_retval0) __nv_cos
// PTX: .extern .func (.param .b64 func_retval0) __nv_cosh
