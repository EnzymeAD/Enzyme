//===- poseidon.h - public header for applications ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The annotations that mark the code Poseidon optimizes, and poseidon_metric.
// A site is code whose floating-point operations Poseidon profiles and
// rewrites: a function marked POSEIDON_OPTIMIZE or POSEIDON_OPTIMIZE_TAU, or
// one call made through __poseidon_optimize or __poseidon_optimize_tau. A site
// without an accuracy target of its own is solved under the compile's
// -poseidon-tau or -poseidon-cost-budget.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_POSEIDON_H
#define POSEIDON_POSEIDON_H

/// Marks the function it precedes as a site: a host or device function, a CUDA
/// or HIP kernel, or a lambda. POSEIDON_OPTIMIZE_TAU(t) also gives the site
/// the relative accuracy target t, a constant.
//
// Both forms carry one argument so that every Poseidon entry of
// llvm.global.annotations has one type: clang types the array after its first
// entry, and on AMDGPU an entry without arguments has another. The int 0 is no
// accuracy target.
#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon", 0)))
#define POSEIDON_OPTIMIZE_TAU(t)                                               \
  __attribute__((annotate("poseidon", (double)(t))))

/// __poseidon_optimize(fn, args...) calls fn(args...) and makes that one call a
/// site, with the call's own argument conversions and return value;
/// __poseidon_optimize_tau(t, fn, args...) also gives it the accuracy target t.
//
// The call goes through an identity the pass recognizes. In C++ the argument
// list chooses the identity: the plain one when fn takes the arguments as
// written; else the explicit one, which nothing defines, and the call has fn's
// return type and the arguments as given.
#if defined(__cplusplus) && defined(__clang__)
#define POSEIDON_DETAIL_CALLEE(fn, ...)                                        \
  decltype(::poseidon::detail::calleeOf(0, fn __VA_OPT__(, ) __VA_ARGS__))
#define __poseidon_optimize(fn, ...)                                           \
  __builtin_choose_expr(                                                       \
      POSEIDON_DETAIL_CALLEE(fn, __VA_ARGS__)::IsExplicit,                     \
      ((typename POSEIDON_DETAIL_CALLEE(                                       \
          fn, __VA_ARGS__)::Type)__poseidon_site_explicit((void *)(fn))),      \
      ((__typeof__(fn) *)__poseidon_site((void *)(fn))))(__VA_ARGS__)
#define __poseidon_optimize_tau(t, fn, ...)                                    \
  __builtin_choose_expr(                                                       \
      POSEIDON_DETAIL_CALLEE(fn, __VA_ARGS__)::IsExplicit,                     \
      ((typename POSEIDON_DETAIL_CALLEE(fn, __VA_ARGS__)::Type)                \
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

/// The forms above lower to a marker call the plugin also accepts as written,
/// declared by the program and defined by nothing:
///
///   __poseidon_fp_optimize(fn, entry...)
///
/// It makes the call of fn its arguments describe a site and is replaced by
/// that call. Each parameter of fn is given by one entry, in order:
///
///   enzyme_const x      x, constant to the derivative
///   enzyme_out x        a floating-point x whose derivative is taken
///   enzyme_dup p dp     a pointer p and the shadow dp that holds its adjoint
///   poseidon_shadow p   a pointer p whose shadow the profiler makes
///
/// and anywhere among them, once,
///
///   poseidon_tau t      the site's relative accuracy target, a constant
///
/// The markers are int globals of those names, passed as with Enzyme, whose
/// other argument markers (enzyme_primal_return, ...) are read as Enzyme reads
/// them. A scalar may go without a marker; a pointer may not, as whether its
/// shadow follows it is what the marker says. The name __enzyme_fp_optimize
/// is the same marker.

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

/// Where a profiling run writes its profile and -poseidon-profile-use reads it
/// when neither is told otherwise; POSEIDON_PROFILE_DIR overrides it for the
/// run.
#define POSEIDON_DEFAULT_PROFILE_DIR "./poseidon.profile"

#ifdef __cplusplus
extern "C" {
#endif

/// Declares the quantity of interest of a profiling run, the program's result,
/// as `name` = `value`; the last call counts. poseidon-probe measures each
/// site's condition number against it.
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
namespace poseidon {
namespace detail {
template <typename T, bool Explicit> struct Callee {
  typedef T Type;
  static const bool IsExplicit = Explicit;
};
template <typename Fn, typename... Args>
POSEIDON_SITE_TARGETS auto calleeOf(int, Fn *fn, Args &&...args)
    -> decltype((void)fn(static_cast<Args &&>(args)...), Callee<Fn *, false>());
template <typename Ret, typename... Params, typename... Args>
POSEIDON_SITE_TARGETS auto calleeOf(long, Ret (*)(Params...), Args...)
    -> Callee<Ret (*)(Args...), true>;
} // namespace detail
} // namespace poseidon
}
#endif

/// Under POSEIDON_DEVICE_MATH, which a compile that loads the Poseidon plugin
/// defines, a device module keeps the libdevice (CUDA) or ocml (HIP) functions
/// a rewrite or a derivative of a device site may call.
//
// Taking their addresses keeps them in the C calling convention; one only ever
// called directly may become fastcc. Each address is stored by a statement of
// its own: C++20 deprecates using the value of an assignment to a volatile
// object.
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

/// Tells the plugin that this device compile defines POSEIDON_DEVICE_MATH,
/// which it requires of a device module with a site under a profile action.
#if defined(POSEIDON_DEVICE_MATH) &&                                           \
    (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
extern "C" inline __attribute__((device, used)) void
__poseidon_device_math_defined(void) {}
#endif

#undef POSEIDON_SITE_HELPER
#undef POSEIDON_SITE_TARGETS

#endif // POSEIDON_POSEIDON_H
