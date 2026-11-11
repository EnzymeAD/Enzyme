//=- Options.cpp - what one compile was asked to do ------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Options.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

namespace poseidon {

namespace {

std::string value(bool v) { return v ? "1" : "0"; }
std::string value(double v) { return formatv("{0:e16}", v).str(); }
std::string value(const std::string &v) { return v; }
std::string value(const std::vector<std::string> &v) { return join(v, ","); }
std::string value(const std::vector<unsigned> &v) {
  return join(map_range(v, [](unsigned n) { return std::to_string(n); }), ",");
}
template <typename T> std::string value(T v) { return std::to_string(v); }

template <typename T> void add(FlagValues &out, StringRef name, const T &v) {
  out.emplace_back(name, value(v));
}

} // namespace

void ProfilingOptions::appendSolveInputs(FlagValues &out) const {
  add(out, "poseidon-loose-coverage", LooseCoverage);
}

void ProfilingOptions::appendOtherValues(FlagValues &out) const {
  add(out, "poseidon-profile-generate", ProfileGenerate);
  add(out, "poseidon-profile-use", ProfileUse);
  add(out, "poseidon-output-profiles", OutputProfiles);
  add(out, "poseidon-kernels", Kernels);
  add(out, "poseidon-min-cost-share", MinCostShare);
}

void CandidateOptions::appendSolveInputs(FlagValues &out) const {
  add(out, "poseidon-enable-herbie", EnableHerbie);
  add(out, "poseidon-enable-pt", EnablePT);
  add(out, "poseidon-enable-multifloat", EnableMultifloat);
  add(out, "poseidon-expansion-components", ExpansionComponents);
  add(out, "poseidon-enable-three-tier", EnableThreeTier);
  add(out, "poseidon-two-tier-step", TwoTierStep);
  add(out, "poseidon-raise-wmma", RaiseWMMA);
  add(out, "poseidon-max-expr-length", MaxExprLength);
  add(out, "poseidon-min-uses-split", MinUsesSplit);
  add(out, "poseidon-min-ops-split", MinOpsSplit);
  add(out, "poseidon-split-calls", SplitCalls);
}

void CandidateOptions::appendLaterSolveInputs(FlagValues &out) const {
  add(out, "poseidon-reduction-subgraphs", ReductionSubgraphs);
  add(out, "poseidon-merge-shared-staging", MergeSharedStaging);
}

void SolverOptions::appendSolveInputs(FlagValues &out) const {
  add(out, "poseidon-early-prune", EarlyPrune);
  add(out, "poseidon-aggressive-dce", AggressiveDCE);
}

void SolverOptions::appendOtherValues(FlagValues &out) const {
  add(out, "poseidon-cost-budget", CostBudget);
  add(out, "poseidon-tau", Tau);
  add(out, "poseidon-tau-cheapest", TauCheapest);
  add(out, "poseidon-joint-dp", JointDP);
  add(out, "poseidon-apply-rewrites", ApplyRewrites);
}

void ModelOptions::appendSolveInputs(FlagValues &out) const {
  add(out, "poseidon-num-samples", NumSamples);
  add(out, "poseidon-random-seed", RandomSeed);
  add(out, "poseidon-sample-log-bits", SampleLogBits);
  add(out, "poseidon-strict-mode", StrictMode);
  add(out, "poseidon-min-arm-samples", MinArmSamples);
  add(out, "poseidon-arm-search-factor", ArmSearchFactor);
  add(out, "poseidon-max-broken-share", MaxBrokenShare);
  add(out, "poseidon-exponent-penalty", ExponentPenalty);
}

void ModelOptions::appendLaterSolveInputs(FlagValues &out) const {
  add(out, "poseidon-freq-weighted-pricing", FreqWeightedPricing);
}

void ModelOptions::appendOtherValues(FlagValues &out) const {
  add(out, "poseidon-cost-model", CostModel);
}

void HerbieOptions::appendSolveInputs(FlagValues &out) const {
  add(out, "poseidon-herbie-num-pts", HerbieNumPts);
  add(out, "poseidon-herbie-num-iters", HerbieNumIters);
  add(out, "poseidon-herbie-num-enodes", HerbieNumEnodes);
  add(out, "poseidon-herbie-num-taylor-terms", HerbieNumTaylorTerms);
  add(out, "poseidon-herbie-pre-floor-bits", HerbiePreFloorBits);
  add(out, "poseidon-herbie-arrays", HerbieArrays);
  add(out, "poseidon-herbie-array-grouping", HerbieArrayGrouping);
  add(out, "poseidon-herbie-array-weights", HerbieArrayWeights);
}

void HerbieOptions::appendOtherValues(FlagValues &out) const {
  add(out, "poseidon-cache", Cache);
  add(out, "poseidon-unit", Unit);
  add(out, "poseidon-import-accuracy-memo", ImportAccuracyMemo);
  add(out, "poseidon-import-herbie-answers", ImportHerbieAnswers);
  add(out, "poseidon-herbie-num-threads", HerbieNumThreads);
  add(out, "poseidon-herbie-timeout", HerbieTimeout);
  add(out, "poseidon-herbie-subgraph-timeout", HerbieSubgraphTimeout);
  add(out, "poseidon-herbie-processes", HerbieProcesses);
  add(out, "poseidon-herbie-binary", HerbieBinary);
  add(out, "poseidon-herbie-platform", HerbiePlatform);
}

void MatmulOptions::appendSolveInputs(FlagValues &out) const {
  add(out, "poseidon-ozaki-host-dispatch", OzakiHostDispatch);
  add(out, "poseidon-inkernel-calibration", InKernelCalibration);
}

void MaterializationOptions::appendSolveInputs(FlagValues &out) const {
  add(out, "poseidon-narrow-param-staging", NarrowParamStaging);
  add(out, "poseidon-narrow-multiview-staging", NarrowMultiviewStaging);
}

void DiagnosticOptions::appendOtherValues(FlagValues &out) const {
  add(out, "poseidon-print", Print);
  add(out, "poseidon-show-table", ShowTable);
  add(out, "poseidon-report-path", ReportPath);
  add(out, "poseidon-explain", Explain);
}

// The DP key's order is part of what the launch manifest of a unit without a
// name is named after (allFlagValues).
FlagValues solveFlagValues(const Options &O) {
  FlagValues out;
  O.Profiling.appendSolveInputs(out);
  O.Candidates.appendSolveInputs(out);
  O.Solver.appendSolveInputs(out);
  O.Model.appendSolveInputs(out);
  O.Herbie.appendSolveInputs(out);
  O.Matmul.appendSolveInputs(out);
  O.Candidates.appendLaterSolveInputs(out);
  O.Model.appendLaterSolveInputs(out);
  O.Materialization.appendSolveInputs(out);
  return out;
}

FlagValues notSolveInputs(const Options &O) {
  FlagValues out;
  O.Profiling.appendOtherValues(out);
  O.Solver.appendOtherValues(out);
  O.Model.appendOtherValues(out);
  O.Herbie.appendOtherValues(out);
  O.Diagnostics.appendOtherValues(out);
  return out;
}

FlagValues allFlagValues(const Options &O) {
  FlagValues all = solveFlagValues(O);
  all.append(notSolveInputs(O));
  return all;
}

} // namespace poseidon
