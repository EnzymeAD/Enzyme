//===- poseidon.h - public header for applications ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What an application includes to use Poseidon: the annotations and calls that
// mark the code it optimizes, and poseidon_metric.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_POSEIDON_H
#define POSEIDON_POSEIDON_H

// Both forms carry one argument so that every Poseidon entry of
// llvm.global.annotations has one type: clang types the array after its first
// entry, and on AMDGPU an entry without arguments has another. The int 0 is no
// accuracy target.
#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon", 0)))
#define POSEIDON_OPTIMIZE_TAU(t)                                               \
  __attribute__((annotate("poseidon", (double)(t))))

// The call through an identity the pass recognizes: the argument list is the
// call's own, so conversions apply and the return value comes back. In C++ the
// list chooses the identity: the plain one when fn takes the arguments as
// written; else the explicit one, which nothing defines, and the call has fn's
// return type and the arguments as given.
#if defined(__cplusplus) && defined(__clang__)
#define __POSEIDON_CALLEE(fn, ...)                                             \
  decltype(__poseidon_callee_of(0, fn __VA_OPT__(, ) __VA_ARGS__))
#define __poseidon_optimize(fn, ...)                                           \
  __builtin_choose_expr(                                                       \
      __POSEIDON_CALLEE(fn, __VA_ARGS__)::__is_explicit,                       \
      ((typename __POSEIDON_CALLEE(                                            \
          fn, __VA_ARGS__)::__type)__poseidon_site_explicit((void *)(fn))),    \
      ((__typeof__(fn) *)__poseidon_site((void *)(fn))))(__VA_ARGS__)
#define __poseidon_optimize_tau(t, fn, ...)                                    \
  __builtin_choose_expr(                                                       \
      __POSEIDON_CALLEE(fn, __VA_ARGS__)::__is_explicit,                       \
      ((typename __POSEIDON_CALLEE(fn, __VA_ARGS__)::__type)                   \
           __poseidon_site_explicit_tau((void *)(fn), (double)(t))),           \
      ((__typeof__(fn) *)__poseidon_site_tau((void *)(fn), (double)(t))))(     \
      __VA_ARGS__)
#else
#define __poseidon_optimize(fn, ...)                                           \
  ((__typeof__(fn) *)__poseidon_site((void *)(fn)))(__VA_ARGS__)
#define __poseidon_optimize_tau(t, fn, ...)                                    \
  ((__typeof__(fn) *)__poseidon_site_tau((void *)(fn), (double)(t)))(          \
      __VA_ARGS__)
#endif

// The identity keeps its plain name in C and C++ alike, never throws, and is
// callable from device code.
#ifdef __cplusplus
#define POSEIDON_SITE_HELPER inline __attribute__((nothrow))
#else
#define POSEIDON_SITE_HELPER static inline __attribute__((nothrow))
#endif
#if defined(__CUDA__) || defined(__HIP__)
#define POSEIDON_SITE_TARGETS __attribute__((host, device))
#else
#define POSEIDON_SITE_TARGETS
#endif

// Where a profiling run writes and a solve reads when neither is told
// otherwise; POSEIDON_PROFILE_DIR overrides it.
#define POSEIDON_DEFAULT_PROFILE_DIR "./poseidon.profile"

#ifdef __cplusplus
extern "C" {
#endif

void poseidon_metric(const char *name, double value);

POSEIDON_SITE_HELPER POSEIDON_SITE_TARGETS void *__poseidon_site(void *fn) {
  return fn;
}
POSEIDON_SITE_HELPER POSEIDON_SITE_TARGETS void *
__poseidon_site_tau(void *fn, double tau) {
  (void)tau;
  return fn;
}

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && defined(__clang__)
extern "C" POSEIDON_SITE_TARGETS __attribute__((nothrow)) void *
__poseidon_site_explicit(void *);
extern "C" POSEIDON_SITE_TARGETS __attribute__((nothrow)) void *
__poseidon_site_explicit_tau(void *, double);

// The callee type of a call of __poseidon_optimize, and which identity it goes
// through, by overload resolution in an unevaluated operand: the first
// overload is the better match whenever fn takes the arguments as written.
extern "C++" {
template <typename _Tp, bool _Explicit> struct __poseidon_callee {
  typedef _Tp __type;
  static const bool __is_explicit = _Explicit;
};
template <typename _Fn, typename... _Args>
POSEIDON_SITE_TARGETS auto __poseidon_callee_of(int, _Fn *__fn,
                                                _Args &&...__args)
    -> decltype((void)__fn(static_cast<_Args &&>(__args)...),
                __poseidon_callee<_Fn *, false>());
template <typename _Ret, typename... _Params, typename... _Args>
POSEIDON_SITE_TARGETS auto __poseidon_callee_of(long, _Ret (*)(_Params...),
                                                _Args...)
    -> __poseidon_callee<_Ret (*)(_Args...), true>;
}
#endif

// The libdevice (CUDA) or ocml (HIP) functions a rewrite or a derivative of a
// device site may call, kept in a device module that includes this header
// under POSEIDON_DEVICE_MATH, which a compile that loads the Poseidon plugin
// defines. Taking their addresses keeps them in the C calling convention; one
// only ever called directly may become fastcc. Each address is stored by a
// statement of its own: C++20 deprecates using the value of an assignment to a
// volatile object.
#if defined(POSEIDON_DEVICE_MATH) && defined(__CUDA_ARCH__) &&                 \
    defined(__CLANG_CUDA_LIBDEVICE_DECLARES_H__)
#define POSEIDON_KEEP(fn)                                                      \
  keep = (void *)&__nv_##fn;                                                   \
  keep = (void *)&__nv_##fn##f
extern "C" inline __device__ __attribute__((used)) void
__poseidon_device_math(void) {
  void *volatile keep;
  POSEIDON_KEEP(sin);
  POSEIDON_KEEP(cos);
  POSEIDON_KEEP(tan);
  POSEIDON_KEEP(asin);
  POSEIDON_KEEP(acos);
  POSEIDON_KEEP(atan);
  POSEIDON_KEEP(atan2);
  POSEIDON_KEEP(sinh);
  POSEIDON_KEEP(cosh);
  POSEIDON_KEEP(tanh);
  POSEIDON_KEEP(asinh);
  POSEIDON_KEEP(acosh);
  POSEIDON_KEEP(atanh);
  POSEIDON_KEEP(exp);
  POSEIDON_KEEP(exp2);
  POSEIDON_KEEP(expm1);
  POSEIDON_KEEP(log);
  POSEIDON_KEEP(log2);
  POSEIDON_KEEP(log10);
  POSEIDON_KEEP(log1p);
  POSEIDON_KEEP(sqrt);
  POSEIDON_KEEP(cbrt);
  POSEIDON_KEEP(pow);
  POSEIDON_KEEP(powi);
  POSEIDON_KEEP(hypot);
  POSEIDON_KEEP(erf);
  POSEIDON_KEEP(lgamma);
  POSEIDON_KEEP(tgamma);
  POSEIDON_KEEP(fabs);
  POSEIDON_KEEP(fma);
  POSEIDON_KEEP(ceil);
  POSEIDON_KEEP(floor);
  POSEIDON_KEEP(rint);
  POSEIDON_KEEP(round);
  POSEIDON_KEEP(trunc);
  POSEIDON_KEEP(copysign);
  POSEIDON_KEEP(fdim);
  POSEIDON_KEEP(fmod);
  POSEIDON_KEEP(remainder);
  POSEIDON_KEEP(fmin);
  POSEIDON_KEEP(fmax);
  (void)keep;
}
#undef POSEIDON_KEEP
#elif defined(POSEIDON_DEVICE_MATH) && defined(__HIP_DEVICE_COMPILE__) &&      \
    defined(__CLANG_HIP_LIBDEVICE_DECLARES_H__)
// HIP spells these as LLVM intrinsics, so its header does not declare them.
extern "C" __device__ float __ocml_exp_f32(float);
extern "C" __device__ float __ocml_exp2_f32(float);
extern "C" __device__ float __ocml_log_f32(float);
extern "C" __device__ float __ocml_log2_f32(float);
extern "C" __device__ float __ocml_log10_f32(float);
#define POSEIDON_KEEP(fn)                                                      \
  keep = (void *)&__ocml_##fn##_f64;                                           \
  keep = (void *)&__ocml_##fn##_f32
extern "C" inline __device__ __attribute__((used)) void
__poseidon_device_math(void) {
  void *volatile keep;
  POSEIDON_KEEP(sin);
  POSEIDON_KEEP(cos);
  POSEIDON_KEEP(tan);
  POSEIDON_KEEP(asin);
  POSEIDON_KEEP(acos);
  POSEIDON_KEEP(atan);
  POSEIDON_KEEP(atan2);
  POSEIDON_KEEP(sinh);
  POSEIDON_KEEP(cosh);
  POSEIDON_KEEP(tanh);
  POSEIDON_KEEP(asinh);
  POSEIDON_KEEP(acosh);
  POSEIDON_KEEP(atanh);
  POSEIDON_KEEP(exp);
  POSEIDON_KEEP(exp2);
  POSEIDON_KEEP(expm1);
  POSEIDON_KEEP(log);
  POSEIDON_KEEP(log2);
  POSEIDON_KEEP(log10);
  POSEIDON_KEEP(log1p);
  POSEIDON_KEEP(cbrt);
  POSEIDON_KEEP(pow);
  POSEIDON_KEEP(hypot);
  POSEIDON_KEEP(erf);
  POSEIDON_KEEP(lgamma);
  POSEIDON_KEEP(tgamma);
  POSEIDON_KEEP(fdim);
  POSEIDON_KEEP(fmod);
  POSEIDON_KEEP(remainder);
  (void)keep;
}
#undef POSEIDON_KEEP
#endif

#undef POSEIDON_SITE_HELPER
#undef POSEIDON_SITE_TARGETS

#endif // POSEIDON_POSEIDON_H
