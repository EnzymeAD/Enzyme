//===- poseidon.h - public header for applications ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// How an application opts in, simplest first:
//
// 1. POSEIDON_OPTIMIZE on a function or a kernel, and -poseidon-tau=<t> on the
//    optimizing compile. Every call of the function is the site: its whole
//    body is profiled under -poseidon-profile-generate and rewritten under
//    -poseidon-profile-use.
// 2. POSEIDON_OPTIMIZE_TAU(t): the same, with that site's own target, which
//    wins over -poseidon-tau. t may be a literal, a macro or a constant
//    expression.
// 3. __poseidon_optimize(fn, args...), or __poseidon_optimize_tau(t, fn,
//    args...) with that call's own target: the call fn(args...), and only that
//    call, is the site. It has its own profile, target and rewritten copy of
//    fn, so two calls of one function can be rewritten differently; other
//    calls of fn run it as written. fn names a function defined in the same
//    translation unit, on the host or in device code; t is a compile-time
//    constant. In device code a pointer the call passes points into a local
//    object or a by-value aggregate of the caller, or the call reaches a
//    constant range of memory through it. Without the plugin the call runs as
//    written.
//    The arguments may instead be an explicit list, as Enzyme takes them:
//    enzyme_dup before a pointer and its shadow after it, enzyme_const before
//    one that needs none, poseidon_tau, t for the call's target. The caller
//    then provides the shadows, for pointers the pass cannot size or a seed of
//    its own, and the site is fn itself. In C++ compiled by clang the same
//    macros take either list; in C an explicit list is a call of
//    __poseidon_fp_optimize(void *, ...), which the program declares. Without
//    the plugin a call with an explicit list does not link.
//    __poseidon_fp_optimize and __enzyme_fp_optimize are accepted as other
//    names of this call.
// 4. -poseidon-kernels=<regex>|all: GPU kernels opted in without editing their
//    source; only those above -poseidon-min-cost-share of the profiled cost
//    are optimized.
// 5. poseidon_metric(name, value), once at the end of the profiling run: the
//    quantity of interest the target refers to. The profiler runtime writes it
//    to <profile dir>/metric.txt. Under `poseidon-probe -- <program> [args...]`
//    (or POSEIDON_PROBE=1) the program runs itself again at exit to measure
//    each site's condition number against it; without, it says what measuring
//    would take, and profile use stops at the sites until they are measured.
//    With POSEIDON_PROBE=0 nothing is measured and each site is solved against
//    its own target. Optional.
//
// The target is the per-operation relative rounding level a site is allowed to
// behave at, so 1e-16 is FP64, 6e-8 is FP32 and 1e-15 is a two-component
// floating-point expansion. The solve takes the cheapest rewrite whose
// modelled error stays under it, and refuses the site outright when no rewrite
// (including leaving it alone) can reach it.
//
// Precedence for a site's accuracy target:
//   its own (POSEIDON_OPTIMIZE_TAU, __poseidon_optimize_tau, or poseidon_tau
//   in an explicit list, not both) > -poseidon-tau > none
// The target is the same number wherever it is written. A site with none is
// solved against -poseidon-cost-budget instead, a share of the site's profiled
// cost; at its default of 0 that is the most accurate rewrite costing no more
// than the code as written, which may be the code as written.
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
// device site may call, kept in every device module that includes this header.
// Taking their addresses keeps them in the C calling convention; one only ever
// called directly may become fastcc. Each address is stored by a statement of
// its own: C++20 deprecates using the value of an assignment to a volatile
// object.
#if defined(__CUDA_ARCH__) && defined(__CLANG_CUDA_LIBDEVICE_DECLARES_H__)
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
#elif defined(__HIP_DEVICE_COMPILE__) &&                                       \
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
