//===- poseidon.h - public header for applications -----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
//    calls of fn run it as written. fn names a host function defined in the
//    same translation unit; t is a compile-time constant. Without the plugin
//    the call runs as written.
// 4. -poseidon-kernels=<regex>|all: GPU kernels opted in without editing their
//    source; only those above -poseidon-min-cost-share of the profiled cost
//    are optimized.
// 5. poseidon_metric(name, value), once at the end of the profiling run: the
//    quantity of interest the target refers to. The profiler runtime writes it
//    to <profile dir>/metric.txt and measures each site's condition number
//    against it. Optional.
// 6. __poseidon_fp_optimize(fn, args...): one call with Enzyme activity
//    markers and shadows the caller provides, for pointers the profiler cannot
//    size or a custom seed; poseidon_tau, t among the arguments sets that
//    call's target. Advanced.
//
// The target is the per-operation relative rounding level a site is allowed to
// behave at, so 1e-16 is FP64, 6e-8 is FP32 and 1e-15 is a two-component
// floating-point expansion. The solve takes the cheapest rewrite whose
// modelled error stays under it, and refuses the site outright when no rewrite
// (including leaving it alone) can reach it.
//
// Precedence for a site's accuracy target:
//   its own (POSEIDON_OPTIMIZE_TAU, __poseidon_optimize_tau, or poseidon_tau
//   in the call) > -poseidon-tau > none
// The target is the same number wherever it is written. A site with none is
// solved against -poseidon-comp-cost-budget instead; at its default of 0 that
// is the most accurate rewrite costing no more than the code as written, which
// may be the code as written.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PROFILE_H
#define POSEIDON_PROFILE_H

#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon")))
#define POSEIDON_OPTIMIZE_TAU(t)                                               \
  __attribute__((annotate("poseidon", (double)(t))))

// The call through an identity the pass recognizes: the argument list is the
// call's own, so conversions apply and the return value comes back.
#define __poseidon_optimize(fn, ...)                                           \
  ((__typeof__(fn) *)__poseidon_site((void *)(fn)))(__VA_ARGS__)
#define __poseidon_optimize_tau(t, fn, ...)                                    \
  ((__typeof__(fn) *)__poseidon_site_tau((void *)(fn), (double)(t)))(          \
      __VA_ARGS__)

// The identity keeps its plain name in C and C++ alike, never throws, and is
// callable from device code, where the pass leaves the call as written.
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

#endif // POSEIDON_PROFILE_H
