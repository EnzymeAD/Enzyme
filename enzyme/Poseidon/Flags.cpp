//=- Flags.cpp - definitions of every Poseidon command-line flag ----------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Flags.h"
#include "matmul/Matmul.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>

using namespace llvm;

namespace poseidon {
namespace flags {

// Profiling.
cl::opt<bool>
    ProfileGenerate("poseidon-profile-generate", cl::init(false), cl::Hidden,
                    cl::desc("Generate instrumented program for FP profiling"));
cl::opt<std::string> ProfileUse(
    "poseidon-profile-use", cl::Hidden, cl::value_desc("directory"),
    cl::ValueOptional,
    cl::desc("FP profile directory to read from for FP optimization"));
cl::opt<std::string> Kernels(
    "poseidon-kernels", cl::init(""), cl::Hidden, cl::value_desc("regex"),
    cl::desc("Treat every kernel whose name matches this regular expression as "
             "an optimization site, as if it carried POSEIDON_OPTIMIZE ('all' "
             "matches every kernel; empty disables)"));
cl::opt<double> MinCostShare(
    "poseidon-min-cost-share", cl::init(0.01), cl::Hidden,
    cl::value_desc("fraction"),
    cl::desc("Share of the profiled FP cost of all -poseidon-kernels sites a "
             "site must reach to be optimized (0 disables); sites named by the "
             "attribute or the marker are never filtered"));
cl::opt<bool> LooseCoverage(
    "poseidon-loose-coverage", cl::init(false), cl::Hidden,
    cl::desc("Allow unexecuted FP instructions in subgraph indentification"));
cl::opt<double> GradFloorRatio(
    "poseidon-grad-floor-ratio", cl::init(1e-6), cl::Hidden,
    cl::desc("Floor on a profiled instruction's accuracy weight, as a fraction "
             "of the profiled function's largest per-execution gradient (0 "
             "disables the floor)"));
cl::opt<double> GradNullRatio(
    "poseidon-grad-null-ratio", cl::init(1e-9), cl::Hidden,
    cl::desc("Per-execution gradient ratio below which a profiled instruction "
             "is reported as a degenerate adjoint seed rather than a genuine "
             "insensitivity"));
cl::opt<bool> GradFloorAbort(
    "poseidon-grad-floor-abort", cl::init(false), cl::Hidden,
    cl::desc("Refuse the solve when the degenerate-adjoint guard fires instead "
             "of flooring and continuing"));

// Candidate classes.
cl::opt<bool>
    EnableHerbie("poseidon-enable-herbie", cl::init(true), cl::Hidden,
                 cl::desc("Use Herbie to rewrite floating-point expressions"));
cl::opt<bool> EnablePT(
    "poseidon-enable-pt", cl::init(true), cl::Hidden,
    cl::desc("Consider precision changes of floating-point expressions"));
cl::opt<bool> EnableMultifloat(
    "poseidon-enable-multifloat", cl::init(true), cl::Hidden,
    cl::desc("Allow floating-point expansion (double-single, 2xFP32) precision "
             "candidates; set =0 to ablate the emulation lever (keeps "
             "FP32/FP16/BF16/FP64)"));
cl::opt<unsigned> ExpansionComponents(
    "poseidon-expansion-components", cl::init(2), cl::Hidden,
    cl::desc("Widest FP32 floating-point expansion admitted as a precision "
             "candidate: 2 = double-single only (default), 3 = also "
             "triple-float, 4 = also quad-float"));
cl::opt<bool>
    EnableThreeTier("poseidon-enable-three-tier", cl::init(true), cl::Hidden,
                    cl::desc("Emit three-tier precision-change candidates"));
cl::opt<int>
    TwoTierStep("poseidon-two-tier-step", cl::init(10), cl::Hidden,
                cl::desc("Percent step for two-tier split-point sweep"));
cl::opt<int>
    ThreeTierStep("poseidon-three-tier-step", cl::init(20), cl::Hidden,
                  cl::desc("Percent step for three-tier split-point sweep"));
cl::opt<bool> RaiseWMMA("poseidon-raise-wmma", cl::init(true), cl::Hidden,
                        cl::desc("Detect scalar FMA-reduction loops as "
                                 "raise-to-WMMA Poseidon candidates"));
cl::opt<bool> RaiseHostGemm(
    "poseidon-raise-host-gemm", cl::init(true), cl::Hidden,
    cl::desc("Recognize dense GEMMs written as thread-parallel reduction nests "
             "with RUNTIME dimensions (finite-element partial-assembly "
             "reduces) as host-dispatch candidates. Set false to ablate this "
             "candidate class."));
cl::opt<unsigned> MaxExprDepth(
    "poseidon-max-expr-depth", cl::init(100), cl::Hidden,
    cl::desc(
        "The maximum depth of expression construction; abort if exceeded"));
cl::opt<unsigned> MaxExprLength(
    "poseidon-max-expr-length", cl::init(10000), cl::Hidden,
    cl::desc("The maximum length of an expression; abort if exceeded"));
cl::opt<unsigned> MinUsesSplit(
    "poseidon-min-uses-split", cl::init(99), cl::Hidden,
    cl::desc("Minimum number of uses of bottleneck node to trigger split"));
cl::opt<unsigned>
    MinOpsSplit("poseidon-min-ops-split", cl::init(99), cl::Hidden,
                cl::desc("Minimum number of upstream operations of "
                         "bottleneck node to trigger split"));
// Under -ffp-contract=on a sum-factorization contraction body is a SINGLE
// llvm.fmuladd, so the flood-fill stops at one operation and the triviality
// filter discards it: no elementwise candidate is ever generated for a
// contraction and the tensor-core raise is the only family that can reach it.
// Admitting the body makes both families compete over the same instructions
// (the solver's footprint rule keeps them mutually exclusive). Off by default:
// it creates elementwise candidates for every recognized reduction, which can
// move the picks of benches whose frontiers are already stamped.
cl::opt<bool> ReductionSubgraphs(
    "poseidon-reduction-subgraphs", cl::init(false), cl::Hidden,
    cl::desc("Admit a one-instruction subgraph when its single operation is "
             "the accumulating FMA of a recognized scalar reduction loop"));
// Storage precision is a property of a memory object, not of a compute unit: a
// staging buffer narrows only when every reader agrees on a tier, so two units
// that both touch it cannot pick different tiers and still get the narrowed
// layout each was priced with. Merging them makes the solver's per-unit choice
// BE the buffer's tier. Off by default: with it off every solution the solver
// can reach is the one it could reach before.
cl::opt<bool> MergeSharedStaging(
    "poseidon-merge-shared-staging", cl::init(false), cl::Hidden,
    cl::desc("Merge FP subgraphs that share a staging buffer (pointer "
             "argument or addrspace(3) global) into one optimization unit, so "
             "the solver cannot pick two storage tiers for one buffer"));

// Solver and budgets.
cl::opt<int64_t> CompCostBudget(
    "poseidon-comp-cost-budget", cl::init(0L), cl::Hidden,
    cl::desc("The maximum computation cost budget for the solver"));
cl::opt<double>
    Tau("poseidon-tau", cl::init(0.0), cl::Hidden,
        cl::desc("Relative accuracy target of every site that carries none "
                 "(the dual of -poseidon-comp-cost-budget; 0 = disabled). A "
                 "site without a matrix product is solved under it like under "
                 "a site value; a site with matrix products has each take the "
                 "cheapest candidate whose domain error at -poseidon-confidence "
                 "is <= it, with the elementwise work left alone"));
// Matrix products only. Their accuracy model scores a PERCENTILE of the
// sampled relative errors and this is the percentile; an elementwise FP
// subgraph is scored by the MEAN error over its samples, so no elementwise
// candidate reads this. A call site overrides it with the marker key
// poseidon_confidence.
cl::opt<double> Confidence(
    "poseidon-confidence", cl::init(kDefaultConfidence), cl::Hidden,
    cl::desc("Confidence level of the accuracy target for matrix products: "
             "this fraction of the sampled inputs must meet the target "
             "(default 0.95)"));
// Every dispatch/raise price in the cost model is a measured wall-clock ratio
// that repeats to about 0.6% run to run, so a modelled cost gap smaller than
// that is not a statement about the hardware; the selector treats it as a tie
// and decides on modelled domain error instead.
cl::opt<double> CostTieBandRel(
    "poseidon-cost-tie-band-rel", cl::init(0.006), cl::Hidden,
    cl::desc("Relative cost band (fraction of the cheapest qualifying "
             "candidate's modelled cost) inside which the error-budget "
             "selector treats two costs as a tie and decides on modelled "
             "domain error; 0 disables"));
cl::opt<bool> TauCheapest(
    "poseidon-tau-cheapest", cl::init(true), cl::Hidden,
    cl::desc("Site-level tau selects the cheapest frontier point whose "
             "accuracy cost clears tau instead of the most accurate one"));
cl::opt<bool> EarlyPrune(
    "poseidon-early-prune", cl::init(true), cl::Hidden,
    cl::desc("Prune dominated candidates in expression transformation phases"));
cl::opt<bool> JointDP(
    "poseidon-joint-dp", cl::init(false), cl::Hidden,
    cl::desc("Solve all annotated sites in the module under one shared cost "
             "budget instead of an independent per-site solve"));
cl::opt<bool> AggressiveDCE(
    "poseidon-aggressive-dce", cl::init(false), cl::Hidden,
    cl::desc("Delete zero-gradient FP instructions that reach no comparison "
             "before the solve"));
cl::opt<std::string> ApplyRewrites(
    "poseidon-apply-rewrites", cl::init(""), cl::Hidden,
    cl::desc("Comma-separated rewrite IDs to apply (e.g.\n"
             "R0_1,R3_0,PT1_2). IDs are from the _rewrites.json\n"
             "report. Bypasses the DP solver. At most one candidate\n"
             "per expression (R) or subgraph (PT)."));

// Cost and accuracy model.
cl::opt<std::string> CostModel(
    "poseidon-cost-model", cl::init(""), cl::Hidden, cl::value_desc("csv"),
    cl::desc("Cost model to price candidates from; without it the model for "
             "the module's target-cpu is looked up in the user cache and then "
             "in what the installation ships"));
cl::opt<std::string> ScalarTypes(
    "poseidon-scalar-types", cl::init(""), cl::Hidden,
    cl::desc("Comma-separated list of supported scalar FP types "
             "(e.g., half,bf16,float,double). Overrides the CSV header."));
cl::opt<unsigned>
    NumSamples("poseidon-num-samples", cl::init(1024), cl::Hidden,
               cl::desc("Number of sampled points for input hypercube"));
cl::opt<unsigned>
    RandomSeed("poseidon-random-seed", cl::init(239778888), cl::Hidden,
               cl::desc("The random seed used in the Poseidon pass"));
cl::opt<unsigned> SampleLogBits(
    "poseidon-sample-log-bits", cl::init(0), cl::Hidden,
    cl::desc("If >0, sample matmul accuracy inputs log-uniformly in magnitude "
             "over [maxMag*2^-B, maxMag] (B = this value) instead of uniformly "
             "over [min,max] (0 = uniform)"));
cl::opt<bool> CancellationSampling(
    "poseidon-cancellation-sampling", cl::init(true), cl::Hidden,
    cl::desc("Add a coincidence stratum that drives leaf-subtraction operand "
             "pairs to near-equality so the accuracy model sees the "
             "cancellation regimes independent sampling misses; a no-op "
             "without a profiled near-singular sqrt argument or denominator"));
cl::opt<double> CancellationFraction(
    "poseidon-cancellation-fraction", cl::init(0.5), cl::Hidden,
    cl::desc("Fraction of sampled points devoted to the cancellation "
             "(close-encounter) stratum when it is active."));
cl::opt<double> CancellationThreshold(
    "poseidon-cancellation-threshold", cl::init(1e-12), cl::Hidden,
    cl::desc("A profiled sqrt-argument or division denominator whose minimum "
             "observed magnitude falls below this activates cancellation-aware "
             "sampling for the enclosing subgraph."));
// 0 keeps the double-precision pipeline: ground truth converged to 53 bits and
// compared in double, which cannot resolve anything below one double ULP, so a
// format wider than FP64 scores as FP64's equal. > 0 compares in MPFR at that
// many bits.
cl::opt<unsigned> AccuracyReferenceBits(
    "poseidon-accuracy-reference-bits", cl::init(0), cl::Hidden,
    cl::desc("If >0, judge accuracy against an MPFR reference of this many "
             "bits and compare in MPFR rather than in double (0 = the "
             "double-precision pipeline)"));
cl::opt<bool> RelativeError(
    "poseidon-relative-error", cl::init(false), cl::Hidden,
    cl::desc(
        "Score candidate accuracy by relative error |gold-candidate|/|gold| "
        "instead of absolute error"));
cl::opt<unsigned>
    MaxMPFRPrec("poseidon-max-mpfr-prec", cl::init(1024), cl::Hidden,
                cl::desc("Max precision for MPFR gold value computation"));
cl::opt<bool> StrictMode(
    "poseidon-strict-mode", cl::init(false), cl::Hidden,
    cl::desc(
        "Discard all candidates that produce NaN or inf outputs for any input "
        "point that originally produced finite outputs"));
cl::opt<double> ExponentPenalty(
    "poseidon-exponent-penalty", cl::init(1e30), cl::Hidden,
    cl::desc("Penalty added to a matmul candidate's accuracy cost when the "
             "candidate input format flushes the smallest nonzero operand "
             "magnitude the profiler observed to zero"));
cl::opt<double> NonfinitePenalty(
    "poseidon-nonfinite-penalty", cl::init(1e30), cl::Hidden,
    cl::desc("Relative-error penalty for a sample where the candidate is "
             "non-finite but the MPFR oracle is finite; must exceed any finite "
             "relative error yet stay far below DBL_MAX after aggregation"));
// A subgraph is priced as (cost per execution) x executions, with `executions`
// read off outputs[0]. That is only meaningful when every instruction in the
// subgraph runs the same number of times; the flood-fill can merge a reduction
// loop body with the once-per-thread code that consumes it, and billing the
// second group at the first group's trip count multiplies its price by K --
// exactly where a floating-point expansion pays its FP64 boundary conversions.
// With this on, each priced instruction is weighted by its own profiled count.
// Off by default so no stamped frontier moves; it is already a no-op on any
// subgraph whose instructions all execute the same number of times.
cl::opt<bool> FreqWeightedPricing(
    "poseidon-freq-weighted-pricing", cl::init(false), cl::Hidden,
    cl::desc("Price each instruction of a frequency-heterogeneous FP subgraph "
             "at its own profiled execution count instead of the subgraph's"));

// Herbie and the result cache.
cl::opt<std::string>
    Cache("poseidon-cache", cl::init(kDefaultCacheDir), cl::Hidden,
          cl::desc("Directory Poseidon keeps its Herbie results, DP tables and "
                   "device-to-host launch descriptors in"));
cl::opt<int> HerbieNumThreads("poseidon-herbie-num-threads", cl::init(8),
                              cl::Hidden,
                              cl::desc("Number of threads Herbie uses"));
cl::opt<int> HerbieTimeout("poseidon-herbie-timeout", cl::init(9999),
                           cl::Hidden,
                           cl::desc("Herbie's timeout to use for each "
                                    "candidate expressions."));
cl::opt<int>
    HerbieNumPts("poseidon-herbie-num-pts", cl::init(1024), cl::Hidden,
                 cl::desc("Number of input points Herbie uses to evaluate "
                          "candidate expressions."));
cl::opt<int> HerbieNumIters(
    "poseidon-herbie-num-iters", cl::init(6), cl::Hidden,
    cl::desc("Number of times Herbie attempts to improve accuracy."));
cl::opt<int> HerbieNumEnodes(
    "poseidon-herbie-num-enodes", cl::init(8000), cl::Hidden,
    cl::desc("Number of equivalence graph nodes to use when doing algebraic "
             "reasoning in Herbie."));
// Wall-clock budget for one subgraph's Herbie invocation, enforced twice on
// purpose: Herbie's own per-core --timeout is clamped to it so a timed-out
// core still writes results.json, and ExecuteAndWait's SecondsToWait backstops
// a Herbie that cannot exit (a crashed worker place leaves the manager
// waiting).
cl::opt<unsigned> HerbieSubgraphTimeout(
    "poseidon-herbie-subgraph-timeout", cl::init(1800),
    cl::desc("Wall-clock seconds allowed for one FP subgraph's Herbie "
             "invocation (0 = unbounded, the historical behaviour). Also "
             "clamps Herbie's own per-core --timeout."));
// Overrides the HERBIE_BINARY baked in at configure time. Herbie result caches
// are generation-specific, so pair a new binary with a fresh -poseidon-cache.
cl::opt<std::string> HerbieBinary(
    "poseidon-herbie-binary", cl::init(""), cl::Hidden,
    cl::desc("Path to the Herbie executable to invoke (empty = the binary "
             "configured at build time)."));
// Otherwise the platform is the one generated next to the resolved cost model
// (<csv>.herbie.rkt, written by poseidon-calibrate --only herbie-platform).
cl::opt<std::string> HerbiePlatform(
    "poseidon-herbie-platform", cl::init(""), cl::Hidden,
    cl::desc("Explicit Herbie platform: a path (contains '/' or ends in "
             ".rkt) to a generated platform file, or a name compiled into the "
             "Herbie binary (e.g. cuda-sm120)."));

// Matmul raising and host dispatch.
cl::opt<bool> OzakiHostDispatch(
    "poseidon-ozaki-host-dispatch", cl::init(true), cl::Hidden,
    cl::desc("Enable Ozaki-II as a host-dispatched GEMM library call, its only "
             "realization; set false to ablate it from the candidate pool"));
cl::opt<unsigned> OzakiForceNm(
    "poseidon-ozaki-force-nm", cl::init(0), cl::Hidden,
    cl::desc(
        "If >0, override the dispatched Ozaki-II num_moduli for every "
        "host-dispatched GEMM (an explicit 0 with "
        "-poseidon-ozaki-native-dgemm forces the native cuBLAS DGEMM path)"));
// On by default: the native cuBLAS DGEMM is an ordinary matrix-product
// candidate, priced from the measured ozaki_dispatch_rel,dgemm row like every
// other dispatch family, and the device decides whether it wins (it does on
// full-rate-FP64 parts, it loses to emulation on FP64-deprioritized ones).
// =false exists for debugging only: it removes the candidate from the solve.
cl::opt<bool> OzakiNativeDgemm(
    "poseidon-ozaki-native-dgemm", cl::init(true), cl::Hidden,
    cl::desc("Propose the zero-error native cuBLAS DGEMM host-dispatch "
             "candidate (num_moduli=0) priced from the measured "
             "ozaki_dispatch_rel,dgemm cost-model row; =false removes it "
             "(debugging only)"));
// Calibration harness only: an uncalibrated class is proposed at a placeholder
// price so poseidon-calibrate can materialize and time it; requires
// -poseidon-apply-rewrites so no DP solve ever consumes the placeholder.
cl::opt<bool> InKernelCalibration(
    "poseidon-inkernel-calibration", cl::init(false), cl::Hidden,
    cl::desc("Calibration harness only: propose in-kernel raise classes that "
             "carry no measured wmma_inkernel_rel row so they can be "
             "materialized and timed (requires -poseidon-apply-rewrites)"));
// The cap is a hardware/toolchain constant, exposed so the refusal can be
// switched off to measure what ptxas really rejects, and so a target with a
// different static cap needs no code change.
cl::opt<unsigned> InKernelSharedCap(
    "poseidon-inkernel-shared-cap", cl::init((unsigned)kStaticShmemCap),
    cl::desc("Static shared-memory bytes an in-kernel raise may occupy, "
             "including what the enclosing kernel already uses (0 = do not "
             "refuse on capacity, which lets ptxas be the oracle)."));

// Materialization.
cl::opt<bool> StageParamArrays(
    "poseidon-stage-param-arrays", cl::init(false), cl::Hidden,
    cl::desc("Extend the df64 staging narrowing to pointer kernel parameters: "
             "hoist the F64 -> df64 split into a pre-launch split kernel "
             "producing {hi,lo} limb arrays and rewrite the kernel to load "
             "limbs"));
// narrowSharedStaging roots its walk at an addrspace(3) global's addrspacecast
// and only rewrites GEPs hanging off that cast inside F. A kernel that declares
// its __shared__ scratch in the kernel and hands it to the annotated body as a
// plain `double *` therefore reports "buffer found, nothing rewritten": every
// element GEP hangs off the callee's POINTER PARAMETER. This arm follows the
// parameter. Off by default: it enlarges the set of buffers the pricing-time
// narrowing can see, which can move already-stamped frontiers.
cl::opt<bool> NarrowParamStaging(
    "poseidon-narrow-param-staging", cl::init(false), cl::Hidden,
    cl::desc("Narrow addrspace(3) staging buffers that reach the annotated "
             "body as pointer parameters"));
// The strict narrowing gate is value-preserving: it fires only when EVERY
// consumer of a staged load already rounds to the narrow tier. That is right
// for the materializer but wrong for the PRICE of a single unit's candidate:
// getCompCost clones the whole function and applies ONE subgraph's candidate,
// so the other units still read the buffer at FP64, the gate bails, and the
// candidate is charged for conversions the joint (all-units) code does not
// contain. With this on the PRICING clone narrows anyway; the cost walk is
// cone-restricted, so out-of-cone FP64 readers are not billed to this unit and
// an in-cone one (a partial tier) still is. The materializer keeps the strict
// gate, and materializeFPSolution reports when the price's assumption did not
// come true.
cl::opt<bool> NarrowStagingSpeculative(
    "poseidon-narrow-staging-speculative", cl::init(false), cl::Hidden,
    cl::desc("Price precision candidates as if the shared staging buffers they "
             "read were narrowed. Pricing only; the materializer keeps the "
             "value-preserving gate"));

// Diagnostics.
cl::opt<bool> Print("poseidon-print", cl::init(false), cl::Hidden,
                    cl::desc("Print Poseidon debug info"));
cl::opt<bool> ShowTable(
    "poseidon-show-table", cl::init(false), cl::Hidden,
    cl::desc(
        "Print the full DP table (highly verbose for large applications)"));
cl::opt<std::string> ReportPath(
    "poseidon-report-path", cl::init(""), cl::Hidden,
    cl::desc("Directory to write Poseidon optimization reports.\n"
             "Emits <func>.json (Pareto table with source locations),\n"
             "<func>.txt (human-readable), <func>_rewrites.json\n"
             "(curated per-rewrite analysis with IDs), and\n"
             "validate_config.json + validate.py (validation script)."));

} // namespace flags

void applyFlagDefaults() {
  // -poseidon-profile-use with no directory means the directory the profiling
  // run writes by default.
  if (flags::ProfileUse.getNumOccurrences() && flags::ProfileUse.empty())
    flags::ProfileUse = kDefaultProfileDir;

  // A confidence level is a fraction of the samples: 1.0 is the worst observed
  // sample and 0 names no sample at all. NaN fails the first test.
  if (!((double)flags::Confidence > 0.0) || (double)flags::Confidence > 1.0)
    report_fatal_error(Twine("Poseidon: -poseidon-confidence=") +
                       std::to_string((double)flags::Confidence) +
                       " is outside (0, 1]");
}

} // namespace poseidon
