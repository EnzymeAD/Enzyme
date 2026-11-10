//=- Flags.cpp - definitions of every Poseidon command-line flag ----------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Flags.h"

#include "poseidon/poseidon.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>

using namespace llvm;

namespace poseidon {

// The directory a profiling run writes and a solve reads when neither is told
// otherwise; the same constant the profiler runtimes default to.
constexpr const char *kDefaultProfileDir = POSEIDON_DEFAULT_PROFILE_DIR;

// Before the flags, which take their defaults from it.
static const Options kDefaults;

// One flag per line, in --help order. A flag's name is its variable name in
// kebab case: `Tau` is -poseidon-tau, `HerbieNumPts` is
// -poseidon-herbie-num-pts. PT, DP, WMMA and InKernel keep their case.
namespace flags {

// Profiling.
cl::opt<bool>
    ProfileGenerate("poseidon-profile-generate",
                    cl::init(kDefaults.Profiling.ProfileGenerate), cl::Hidden,
                    cl::desc("Generate instrumented program for FP profiling"));
cl::opt<std::string> ProfileUse(
    "poseidon-profile-use", cl::Hidden, cl::value_desc("directory"),
    cl::ValueOptional,
    cl::desc("FP profile directory to read from for FP optimization"));
cl::list<std::string> OutputProfiles(
    "poseidon-output-profiles", cl::Hidden, cl::CommaSeparated,
    cl::value_desc("dir,dir,..."),
    cl::desc("Profile directories of the same site, each recorded with one "
             "output seeded alone. Each output's error is then measured "
             "against that output's own sensitivity, and an operation is "
             "weighted by the output it matters most to, so no output can be "
             "traded away under a site-wide average"));
cl::opt<std::string> Kernels(
    "poseidon-kernels", cl::init(kDefaults.Profiling.Kernels), cl::Hidden,
    cl::value_desc("regex"),
    cl::desc("Treat every kernel whose name matches this regular expression as "
             "an optimization site, as if it carried POSEIDON_OPTIMIZE ('all' "
             "matches every kernel; empty disables)"));
cl::opt<double> MinCostShare(
    "poseidon-min-cost-share", cl::init(kDefaults.Profiling.MinCostShare),
    cl::Hidden, cl::value_desc("fraction"),
    cl::desc("Share of the profiled FP cost of all -poseidon-kernels sites a "
             "site must reach to be optimized (0 disables); sites named by the "
             "attribute or the marker are never filtered"));
cl::opt<bool> LooseCoverage(
    "poseidon-loose-coverage", cl::init(kDefaults.Profiling.LooseCoverage),
    cl::Hidden,
    cl::desc("Allow unexecuted FP instructions in subgraph identification"));

// Candidate classes.
cl::opt<bool>
    EnableHerbie("poseidon-enable-herbie",
                 cl::init(kDefaults.Candidates.EnableHerbie), cl::Hidden,
                 cl::desc("Use Herbie to rewrite floating-point expressions"));
cl::opt<bool> EnablePT(
    "poseidon-enable-pt", cl::init(kDefaults.Candidates.EnablePT), cl::Hidden,
    cl::desc("Consider precision changes of floating-point expressions"));
cl::opt<bool> EnableMultifloat(
    "poseidon-enable-multifloat",
    cl::init(kDefaults.Candidates.EnableMultifloat), cl::Hidden,
    cl::desc("Allow floating-point expansion (double-single, 2xFP32) precision "
             "candidates; set =0 to ablate the emulation lever (keeps "
             "FP32/FP16/BF16/FP64)"));
cl::opt<unsigned> ExpansionComponents(
    "poseidon-expansion-components",
    cl::init(kDefaults.Candidates.ExpansionComponents), cl::Hidden,
    cl::desc("Widest FP32 floating-point expansion admitted as a precision "
             "candidate: 2 = double-single only (default), 3 = also "
             "triple-float, 4 = also quad-float"));
cl::opt<bool>
    EnableThreeTier("poseidon-enable-three-tier",
                    cl::init(kDefaults.Candidates.EnableThreeTier), cl::Hidden,
                    cl::desc("Emit three-tier precision-change candidates"));
cl::opt<int>
    TwoTierStep("poseidon-two-tier-step",
                cl::init(kDefaults.Candidates.TwoTierStep), cl::Hidden,
                cl::desc("Percent step for two-tier split-point sweep"));
cl::opt<bool> RaiseWMMA("poseidon-raise-wmma",
                        cl::init(kDefaults.Candidates.RaiseWMMA), cl::Hidden,
                        cl::desc("Detect scalar FMA-reduction loops as "
                                 "raise-to-WMMA Poseidon candidates"));
cl::opt<unsigned> MaxExprLength(
    "poseidon-max-expr-length", cl::init(kDefaults.Candidates.MaxExprLength),
    cl::Hidden,
    cl::desc("The maximum length of an expression; abort if exceeded"));
cl::opt<unsigned> MinUsesSplit(
    "poseidon-min-uses-split", cl::init(kDefaults.Candidates.MinUsesSplit),
    cl::Hidden,
    cl::desc("Minimum number of uses of bottleneck node to trigger split"));
cl::opt<unsigned>
    MinOpsSplit("poseidon-min-ops-split",
                cl::init(kDefaults.Candidates.MinOpsSplit), cl::Hidden,
                cl::desc("Minimum number of upstream operations of "
                         "bottleneck node to trigger split"));
// Under -ffp-contract=on a contraction body is a single llvm.fmuladd, which the
// triviality filter discards, so only a tensor-core raise could reach it. Off
// by default: it adds elementwise candidates to every recognized reduction.
cl::opt<bool> ReductionSubgraphs(
    "poseidon-reduction-subgraphs",
    cl::init(kDefaults.Candidates.ReductionSubgraphs), cl::Hidden,
    cl::desc("Admit a one-instruction subgraph when its single operation is "
             "the accumulating FMA of a recognized scalar reduction loop"));
// A staging buffer narrows only when every reader agrees on a tier, so units
// that share one must be one unit for the per-unit tier to be the buffer's.
cl::opt<bool> MergeSharedStaging(
    "poseidon-merge-shared-staging",
    cl::init(kDefaults.Candidates.MergeSharedStaging), cl::Hidden,
    cl::desc("Merge FP subgraphs that share a staging buffer (pointer "
             "argument or addrspace(3) global) into one optimization unit, so "
             "the solver cannot pick two storage tiers for one buffer"));
cl::list<std::string> SplitCalls(
    "poseidon-split-calls", cl::Hidden, cl::CommaSeparated,
    cl::value_desc("op,op,..."),
    cl::desc("Give every call to these operators (Herbie names, e.g. pow) its "
             "own FP subgraph with the operations only it uses, whatever "
             "-poseidon-min-uses-split/-poseidon-min-ops-split say"));

// Solver and budgets.
cl::opt<int64_t> CompCostBudget(
    "poseidon-comp-cost-budget", cl::init(kDefaults.Solver.CompCostBudget),
    cl::Hidden, cl::desc("The maximum computation cost budget for the solver"));
cl::opt<double> Tau(
    "poseidon-tau", cl::init(kDefaults.Solver.Tau), cl::Hidden,
    cl::desc("Relative accuracy target of every site that carries none "
             "(the dual of -poseidon-comp-cost-budget; 0 = disabled); a site "
             "is solved under it exactly as under a target of its own"));
cl::opt<bool> TauCheapest(
    "poseidon-tau-cheapest", cl::init(kDefaults.Solver.TauCheapest), cl::Hidden,
    cl::desc("Site-level tau selects the cheapest frontier point whose "
             "accuracy cost clears tau instead of the most accurate one"));
cl::opt<bool> EarlyPrune(
    "poseidon-early-prune", cl::init(kDefaults.Solver.EarlyPrune), cl::Hidden,
    cl::desc("Drop dominated partial solutions after every solver unit. Off: "
             "solve exactly, dropping a partial solution only for one that is "
             "cheaper and no less accurate and that every later unit treats "
             "alike, so the frontier is that of all footprint-compatible "
             "strategies, at the price of pricing each precision change "
             "against every set of rewritten outputs on its subgraph"));
cl::opt<bool> JointDP(
    "poseidon-joint-dp", cl::init(kDefaults.Solver.JointDP), cl::Hidden,
    cl::desc("Solve all annotated sites in the module under one shared cost "
             "budget instead of an independent per-site solve"));
cl::opt<bool> AggressiveDCE(
    "poseidon-aggressive-dce", cl::init(kDefaults.Solver.AggressiveDCE),
    cl::Hidden,
    cl::desc("Delete zero-gradient FP instructions that reach no comparison "
             "before the solve"));
cl::opt<std::string> ApplyRewrites(
    "poseidon-apply-rewrites", cl::init(kDefaults.Solver.ApplyRewrites),
    cl::Hidden,
    cl::desc("Comma-separated candidates to apply instead of a solve, each\n"
             "[<site>:]<unit>_<ordinal> (R0_1, PT1_2, M0_3) or\n"
             "[<site>:]<unit>=<name> (M0=native-dgemm cublas f64/f64),\n"
             "<name> being the candidate's expression, precision\n"
             "description or matrix-product label. <site> is the site's\n"
             "symbol as the pin line gives it, or its unqualified name\n"
             "when no other site of the module has it; an item without\n"
             "one applies to every site. At most one candidate per unit.\n"
             "Every solve prints the value that applies its choice\n"
             "again as '[poseidon] pin: <value>'."));

// Cost and accuracy model.
cl::opt<std::string> CostModel(
    "poseidon-cost-model", cl::init(kDefaults.Model.CostModel), cl::Hidden,
    cl::value_desc("csv"),
    cl::desc("Cost model to price candidates from; without it the model for "
             "the module's target-cpu is looked up in the user cache and then "
             "in what the installation ships"));
cl::opt<unsigned>
    NumSamples("poseidon-num-samples", cl::init(kDefaults.Model.NumSamples),
               cl::Hidden,
               cl::desc("Number of sampled points for input hypercube"));
cl::opt<unsigned>
    RandomSeed("poseidon-random-seed", cl::init(kDefaults.Model.RandomSeed),
               cl::Hidden,
               cl::desc("The random seed used in the Poseidon pass"));
cl::opt<unsigned> SampleLogBits(
    "poseidon-sample-log-bits", cl::init(kDefaults.Model.SampleLogBits),
    cl::Hidden,
    cl::desc("If >0, sample matmul accuracy inputs log-uniformly in magnitude "
             "over [maxMag*2^-B, maxMag] (B = this value) instead of uniformly "
             "over [min,max] (0 = uniform)"));
cl::opt<bool> StrictMode(
    "poseidon-strict-mode", cl::init(kDefaults.Model.StrictMode), cl::Hidden,
    cl::desc(
        "Discard all candidates that produce NaN or inf outputs for any input "
        "point that originally produced finite outputs"));
cl::opt<unsigned> MinArmSamples(
    "poseidon-min-arm-samples", cl::init(kDefaults.Model.MinArmSamples),
    cl::Hidden,
    cl::desc("Every arm of a regime-split (if) rewrite must be priced on at "
             "least this many samples that take it, drawn from the profiled "
             "box by rejection when the base samples miss it; a rewrite with "
             "an arm that stays unsampled is dropped (0 = off)"));
cl::opt<unsigned> ArmSearchFactor(
    "poseidon-arm-search-factor", cl::init(kDefaults.Model.ArmSearchFactor),
    cl::Hidden,
    cl::desc("Rejection-sampling budget for an under-sampled regime-split arm, "
             "in multiples of -poseidon-num-samples"));
cl::opt<double> MaxBrokenShare(
    "poseidon-max-broken-share", cl::init(kDefaults.Model.MaxBrokenShare),
    cl::Hidden,
    cl::desc("Drop a Herbie candidate whose relative error reaches 1 (wrong "
             "sign or magnitude) on more than this share of the samples the "
             "original gets within 1; the gradient-weighted accuracy model is "
             "first order and cannot price such a change (negative = off)"));
cl::opt<double> ExponentPenalty(
    "poseidon-exponent-penalty", cl::init(kDefaults.Model.ExponentPenalty),
    cl::Hidden,
    cl::desc("Penalty added to a matmul candidate's accuracy cost when the "
             "candidate input format flushes the smallest nonzero operand "
             "magnitude the profiler observed to zero"));
// A subgraph is billed at outputs[0]'s execution count, which over-bills the
// once-per-thread code a reduction body feeds by the trip count; that is where
// an expansion pays its FP64 boundary conversions.
cl::opt<bool> FreqWeightedPricing(
    "poseidon-freq-weighted-pricing",
    cl::init(kDefaults.Model.FreqWeightedPricing), cl::Hidden,
    cl::desc("Price each instruction of a frequency-heterogeneous FP subgraph "
             "at its own profiled execution count instead of the subgraph's"));

// Herbie and the result cache.
cl::opt<std::string>
    Cache("poseidon-cache", cl::init(kDefaults.Herbie.Cache), cl::Hidden,
          cl::desc("Directory Poseidon keeps its Herbie results, DP tables and "
                   "device-to-host launch descriptors in"));
cl::opt<std::string> Unit(
    "poseidon-unit", cl::init(kDefaults.Herbie.Unit), cl::Hidden,
    cl::value_desc("name"),
    cl::desc("Name of the translation unit whose device and host halves are "
             "compiled; the host half takes the launch descriptors the device "
             "half of the same source and unit wrote (poseidon-clang passes "
             "one to both halves of each compile)"));
cl::opt<std::string> ImportAccuracyMemo(
    "poseidon-import-accuracy-memo",
    cl::init(kDefaults.Herbie.ImportAccuracyMemo), cl::Hidden,
    cl::value_desc("directory"),
    cl::desc("Directory of matmulacc_<hash>.txt matrix accuracy results "
             "written before the memo stored its key; a result missing from "
             "the cache is taken from there when the file of the same inputs "
             "exists, and stored in the cache under its key"));
cl::opt<int> HerbieNumThreads("poseidon-herbie-num-threads",
                              cl::init(kDefaults.Herbie.HerbieNumThreads),
                              cl::Hidden,
                              cl::desc("Number of threads Herbie uses"));
cl::opt<int> HerbieTimeout("poseidon-herbie-timeout",
                           cl::init(kDefaults.Herbie.HerbieTimeout), cl::Hidden,
                           cl::desc("Herbie's timeout to use for each "
                                    "candidate expressions."));
cl::opt<int>
    HerbieNumPts("poseidon-herbie-num-pts",
                 cl::init(kDefaults.Herbie.HerbieNumPts), cl::Hidden,
                 cl::desc("Number of input points Herbie uses to evaluate "
                          "candidate expressions."));
cl::opt<int> HerbieNumIters(
    "poseidon-herbie-num-iters", cl::init(kDefaults.Herbie.HerbieNumIters),
    cl::Hidden,
    cl::desc("Number of times Herbie attempts to improve accuracy."));
cl::opt<int> HerbieNumEnodes(
    "poseidon-herbie-num-enodes", cl::init(kDefaults.Herbie.HerbieNumEnodes),
    cl::Hidden,
    cl::desc("Number of equivalence graph nodes to use when doing algebraic "
             "reasoning in Herbie."));
// Wall-clock budget for one subgraph's Herbie invocation, enforced twice on
// purpose: Herbie's own per-core --timeout is clamped to it so a timed-out
// core still writes results.json, and ExecuteAndWait's SecondsToWait backstops
// a Herbie that cannot exit (a crashed worker place leaves the manager
// waiting).
cl::opt<unsigned> HerbieSubgraphTimeout(
    "poseidon-herbie-subgraph-timeout",
    cl::init(kDefaults.Herbie.HerbieSubgraphTimeout), cl::Hidden,
    cl::desc("Wall-clock seconds allowed for one FP subgraph's Herbie "
             "invocation (0 = unbounded, the historical behaviour). Also "
             "clamps Herbie's own per-core --timeout."));
// Herbie's worker places share one Racket heap, so a single process with many
// workers spends most of its time in collection; separate processes do not.
cl::opt<unsigned> HerbieProcesses(
    "poseidon-herbie-processes", cl::init(kDefaults.Herbie.HerbieProcesses),
    cl::desc("Herbie processes one subgraph's FPCores are split across "
             "(0 = one per -poseidon-herbie-num-threads worker, each with one "
             "worker; 1 = a single process with all workers)."));
cl::opt<unsigned> HerbiePreFloorBits(
    "poseidon-herbie-pre-floor-bits",
    cl::init(kDefaults.Herbie.HerbiePreFloorBits),
    cl::desc("For a profiled range [lo, hi] with lo < 0 < hi, exclude "
             "|x| < max(-lo, hi) * 2^-K from the Herbie :pre (0 = off)."));
// Overrides the HERBIE_BINARY baked in at configure time. Herbie result caches
// are generation-specific, so pair a new binary with a fresh -poseidon-cache.
cl::opt<std::string> HerbieBinary(
    "poseidon-herbie-binary", cl::init(kDefaults.Herbie.HerbieBinary),
    cl::Hidden,
    cl::desc("Path to the Herbie executable to invoke (empty = the binary "
             "configured at build time)."));
// Otherwise the platform is the one generated next to the resolved cost model
// (<csv>.herbie.rkt, written by poseidon-calibrate --only herbie-platform).
cl::opt<std::string> HerbiePlatform(
    "poseidon-herbie-platform", cl::init(kDefaults.Herbie.HerbiePlatform),
    cl::Hidden,
    cl::desc("Explicit Herbie platform: a path (contains '/' or ends in "
             ".rkt) to a generated platform file, or a name compiled into the "
             "Herbie binary (e.g. cuda-sm120)."));
// One (array ...) core per precision for a subgraph's outputs, with the
// subterms they share let*-bound: Herbie searches them jointly and the rewrite
// is priced as one unit, so work the outputs share is costed and erased once.
cl::opt<bool> HerbieArrays(
    "poseidon-herbie-arrays", cl::init(kDefaults.Herbie.HerbieArrays),
    cl::Hidden,
    cl::desc("Send a multi-output subgraph to Herbie as one array FPCore per "
             "precision; per-output cores are sent only for an array core "
             "that returns nothing."));
cl::opt<std::string> HerbieArrayGrouping(
    "poseidon-herbie-array-grouping",
    cl::init(kDefaults.Herbie.HerbieArrayGrouping), cl::Hidden,
    cl::desc("Which outputs share an array core: 'inputs' (same inputs and "
             "execution count) or 'executions' (same execution count)."));
cl::opt<std::string> HerbieArrayWeights(
    "poseidon-herbie-array-weights",
    cl::init(kDefaults.Herbie.HerbieArrayWeights), cl::Hidden,
    cl::desc("How array elements are weighted (:herbie-weights): "
             "'sensitivity' (profiled sum of |gradient * value|), 'gradient' "
             "(|gradient| times the mean magnitude of the profiled range, the "
             "weight of an ULP in the accuracy model), or 'none'."));

// Matmul raising and host dispatch.
cl::opt<bool> OzakiHostDispatch(
    "poseidon-ozaki-host-dispatch",
    cl::init(kDefaults.Matmul.OzakiHostDispatch), cl::Hidden,
    cl::desc("Enable Ozaki-II as a host-dispatched GEMM library call, its only "
             "realization; set false to ablate it from the candidate pool"));
// Calibration harness only: an uncalibrated class is proposed at a placeholder
// price so poseidon-calibrate can materialize and time it; requires
// -poseidon-apply-rewrites so no DP solve ever consumes the placeholder.
cl::opt<bool> InKernelCalibration(
    "poseidon-inkernel-calibration",
    cl::init(kDefaults.Matmul.InKernelCalibration), cl::Hidden,
    cl::desc("Calibration harness only: propose in-kernel raise classes that "
             "carry no measured wmma_inkernel_rel row so they can be "
             "materialized and timed (requires -poseidon-apply-rewrites)"));

// Materialization.
// The global-rooted narrowing only sees GEPs off the global's addrspacecast; a
// kernel that hands its __shared__ scratch to the annotated body as `double *`
// has every element GEP off the callee's pointer parameter instead.
cl::opt<bool> NarrowParamStaging(
    "poseidon-narrow-param-staging",
    cl::init(kDefaults.Materialization.NarrowParamStaging), cl::Hidden,
    cl::desc("Narrow addrspace(3) staging buffers that reach the annotated "
             "body as pointer parameters"));
// A shared-memory kernel that uses one __shared__ array through views of
// different shapes and constant offsets (MFEM's) has element GEPs of several
// strides, which the global-rooted narrowing gives up on.
cl::opt<bool> NarrowMultiviewStaging(
    "poseidon-narrow-multiview-staging",
    cl::init(kDefaults.Materialization.NarrowMultiviewStaging), cl::Hidden,
    cl::desc("With -poseidon-narrow-param-staging, also narrow addrspace(3) "
             "staging buffers the body accesses directly, through views of "
             "any shape"));

// Diagnostics.
cl::opt<bool> Print("poseidon-print", cl::init(kDefaults.Diagnostics.Print),
                    cl::Hidden, cl::desc("Print Poseidon debug info"));
cl::opt<bool> ShowTable(
    "poseidon-show-table", cl::init(kDefaults.Diagnostics.ShowTable),
    cl::Hidden,
    cl::desc(
        "Print the full DP table (highly verbose for large applications)"));
cl::opt<std::string> ReportPath(
    "poseidon-report-path", cl::init(kDefaults.Diagnostics.ReportPath),
    cl::Hidden,
    cl::desc("Directory to write Poseidon optimization reports.\n"
             "Emits <func>.json (Pareto table with source locations),\n"
             "<func>.txt (human-readable), <func>_rewrites.json\n"
             "(curated per-rewrite analysis with IDs), and\n"
             "validate_config.json + validate.py (validation script)."));
cl::opt<std::string> Explain(
    "poseidon-explain", cl::init(kDefaults.Diagnostics.Explain), cl::Hidden,
    cl::desc("Directory to write <func>.explain.json, what each site's solve\n"
             "changed, considered and rejected, at source locations (also\n"
             "written to -poseidon-report-path)."));

} // namespace flags

static void checkFlagClassification(const Options &O) {
  StringMap<unsigned> listed;
  for (const auto &kv : solveFlagValues(O))
    ++listed[kv.first];
  for (const auto &kv : notSolveInputs(O))
    ++listed[kv.first];
  SmallVector<StringRef> misplaced;
  for (const auto &[name, opt] : cl::getRegisteredOptions())
    if (name.starts_with("poseidon-") && listed.lookup(name) != 1)
      misplaced.push_back(name);
  if (misplaced.empty())
    return;
  // The first by name: the option table holds its names in hash order.
  llvm::sort(misplaced);
  StringRef name = misplaced.front();
  report_fatal_error(Twine("Poseidon: -") + name + " is in " +
                     (listed.lookup(name) ? "both" : "neither") +
                     " of solveFlagValues() and notSolveInputs() in "
                     "Core/Options.cpp; a flag the DP table depends on goes "
                     "in the first, any other in the second.");
}

Options readOptions() {
  using namespace flags;
  Options O;

  ProfilingOptions &P = O.Profiling;
  P.ProfileGenerate = ProfileGenerate;
  // -poseidon-profile-use with no directory means the directory the profiling
  // run writes by default.
  P.ProfileUse = ProfileUse.getNumOccurrences() && ProfileUse.empty()
                     ? std::string(kDefaultProfileDir)
                     : ProfileUse.getValue();
  P.OutputProfiles.assign(OutputProfiles.begin(), OutputProfiles.end());
  P.Kernels = Kernels;
  P.MinCostShare = MinCostShare;
  P.LooseCoverage = LooseCoverage;

  CandidateOptions &C = O.Candidates;
  C.EnableHerbie = EnableHerbie;
  C.EnablePT = EnablePT;
  C.EnableMultifloat = EnableMultifloat;
  C.ExpansionComponents = ExpansionComponents;
  C.EnableThreeTier = EnableThreeTier;
  C.TwoTierStep = TwoTierStep;
  C.RaiseWMMA = RaiseWMMA;
  C.MaxExprLength = MaxExprLength;
  C.MinUsesSplit = MinUsesSplit;
  C.MinOpsSplit = MinOpsSplit;
  C.ReductionSubgraphs = ReductionSubgraphs;
  C.MergeSharedStaging = MergeSharedStaging;
  C.SplitCalls.assign(SplitCalls.begin(), SplitCalls.end());

  SolverOptions &S = O.Solver;
  S.CompCostBudget = CompCostBudget;
  S.Tau = Tau;
  S.TauCheapest = TauCheapest;
  S.EarlyPrune = EarlyPrune;
  S.JointDP = JointDP;
  S.AggressiveDCE = AggressiveDCE;
  S.ApplyRewrites = ApplyRewrites;

  ModelOptions &M = O.Model;
  M.CostModel = CostModel;
  M.NumSamples = NumSamples;
  M.RandomSeed = RandomSeed;
  M.SampleLogBits = SampleLogBits;
  M.StrictMode = StrictMode;
  M.MinArmSamples = MinArmSamples;
  M.ArmSearchFactor = ArmSearchFactor;
  M.MaxBrokenShare = MaxBrokenShare;
  M.ExponentPenalty = ExponentPenalty;
  M.FreqWeightedPricing = FreqWeightedPricing;

  HerbieOptions &H = O.Herbie;
  H.Cache = Cache;
  H.Unit = Unit;
  H.ImportAccuracyMemo = ImportAccuracyMemo;
  H.HerbieNumThreads = HerbieNumThreads;
  H.HerbieTimeout = HerbieTimeout;
  H.HerbieNumPts = HerbieNumPts;
  H.HerbieNumIters = HerbieNumIters;
  H.HerbieNumEnodes = HerbieNumEnodes;
  H.HerbieSubgraphTimeout = HerbieSubgraphTimeout;
  H.HerbieProcesses = HerbieProcesses;
  H.HerbiePreFloorBits = HerbiePreFloorBits;
  H.HerbieBinary = HerbieBinary;
  H.HerbiePlatform = HerbiePlatform;
  H.HerbieArrays = HerbieArrays;
  H.HerbieArrayWeights = HerbieArrayWeights;
  H.HerbieArrayGrouping = HerbieArrayGrouping;

  O.Matmul.OzakiHostDispatch = OzakiHostDispatch;
  O.Matmul.InKernelCalibration = InKernelCalibration;

  O.Materialization.NarrowParamStaging = NarrowParamStaging;
  O.Materialization.NarrowMultiviewStaging = NarrowMultiviewStaging;

  DiagnosticOptions &D = O.Diagnostics;
  D.Print = Print;
  D.ShowTable = ShowTable;
  D.ReportPath = ReportPath;
  D.Explain = Explain;

  checkFlagClassification(O);
  return O;
}

} // namespace poseidon
