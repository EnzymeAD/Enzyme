//=- Flags.cpp - definitions of every Poseidon command-line flag ----------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Flags.h"

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
    cl::desc("Allow unexecuted FP instructions in subgraph identification"));

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
cl::opt<bool> RaiseWMMA("poseidon-raise-wmma", cl::init(true), cl::Hidden,
                        cl::desc("Detect scalar FMA-reduction loops as "
                                 "raise-to-WMMA Poseidon candidates"));
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

// Solver and budgets.
cl::opt<int64_t> CompCostBudget(
    "poseidon-comp-cost-budget", cl::init(0L), cl::Hidden,
    cl::desc("The maximum computation cost budget for the solver"));
cl::opt<double> Tau(
    "poseidon-tau", cl::init(0.0), cl::Hidden,
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
    "poseidon-herbie-subgraph-timeout", cl::init(1800), cl::Hidden,
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
// Calibration harness only: an uncalibrated class is proposed at a placeholder
// price so poseidon-calibrate can materialize and time it; requires
// -poseidon-apply-rewrites so no DP solve ever consumes the placeholder.
cl::opt<bool> InKernelCalibration(
    "poseidon-inkernel-calibration", cl::init(false), cl::Hidden,
    cl::desc("Calibration harness only: propose in-kernel raise classes that "
             "carry no measured wmma_inkernel_rel row so they can be "
             "materialized and timed (requires -poseidon-apply-rewrites)"));

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
