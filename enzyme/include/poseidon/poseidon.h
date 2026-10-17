//===- poseidon.h - public header for applications -----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The two lines an application writes for Poseidon.
//
// POSEIDON_OPTIMIZE on a kernel makes that kernel an optimization site: the
// whole kernel body is the annotated computation, profiled under
// -poseidon-profile-generate and rewritten under -poseidon-profile-use.
//
// POSEIDON_OPTIMIZE_TAU(t) is POSEIDON_OPTIMIZE with an accuracy target for
// that one site: t is the per-operation relative rounding level the site is
// allowed to behave at, so 1e-16 is FP64, 6e-8 is FP32 and 1e-15 is a
// two-component floating-point expansion. The solve takes the cheapest rewrite
// whose modelled error stays under it, and refuses the site outright when no
// rewrite (including leaving it alone) can reach it.
//
// Precedence for a site's accuracy target:
//   POSEIDON_OPTIMIZE_TAU value  >  -poseidon-tau  >  none (site left as
//   written)
// -poseidon-tau is the target of every site that carries none. A site with
// no matrix product is then solved under it exactly as under its own value; a
// site with matrix products has them selected against it and its elementwise
// work left alone, whereas a site value also solves the elementwise work.
//
// poseidon_metric declares the quantity of interest the accuracy target refers
// to, once, at the end of the profiling workload. The profiler runtime writes
// it to <profile dir>/metric.txt and measures each site's condition number
// against it.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PROFILE_H
#define POSEIDON_PROFILE_H

#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon")))
#define POSEIDON_OPTIMIZE_TAU(t) __attribute__((annotate("poseidon;tau=" #t)))

// Where a profiling run writes and a solve reads when neither is told
// otherwise; POSEIDON_PROFILE_DIR overrides it.
#define POSEIDON_DEFAULT_PROFILE_DIR "./poseidon.profile"

#ifdef __cplusplus
extern "C" {
#endif

void poseidon_metric(const char *name, double value);

#ifdef __cplusplus
}
#endif

#endif // POSEIDON_PROFILE_H
