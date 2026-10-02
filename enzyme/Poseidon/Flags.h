//=- Flags.h - every Poseidon command-line flag, declared once ------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Poseidon's flags are defined in Flags.cpp and declared here, so a
// flag's name, type and default live in one place and no translation unit has
// to re-declare one. Every flag is spelled -poseidon-<name>, reaches the pass
// as -mllvm -poseidon-<name>, and is supplied by poseidon-clang from the
// -poseidon-<name> the user writes. Defaults are part of the paper
// configuration: changing one changes every banked result.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_FLAGS_H
#define POSEIDON_FLAGS_H

#include "poseidon/poseidon.h"

#include "llvm/Support/CommandLine.h"

#include <cstdint>
#include <string>

namespace poseidon {

// The directory a profiling run writes and a solve reads when neither is told
// otherwise; the same constant the profiler runtimes default to.
constexpr const char *kDefaultProfileDir = POSEIDON_DEFAULT_PROFILE_DIR;
constexpr const char *kDefaultCacheDir = "./.poseidon";

// The confidence level every accuracy-cache entry banked before
// -poseidon-confidence existed was computed at; the cache key carries the
// level only when it differs from this, so those entries stay valid.
constexpr double kDefaultConfidence = 0.95;

// Resolve the flag spellings that carry a default only when written without a
// value. Idempotent; called at the entry of each phase.
void applyFlagDefaults();

// One flag per line, in --help order. A flag's name is its variable name in
// kebab case: `Tau` is -poseidon-tau, `HerbieNumPts` is
// -poseidon-herbie-num-pts. PT, DP, MPFR, WMMA and InKernel keep their case.
namespace flags {

// Profiling.
extern llvm::cl::opt<bool> ProfileGenerate;
extern llvm::cl::opt<std::string> ProfileUse;
extern llvm::cl::opt<std::string> Kernels;
extern llvm::cl::opt<double> MinCostShare;
extern llvm::cl::opt<bool> LooseCoverage;
extern llvm::cl::opt<double> GradFloorRatio;
extern llvm::cl::opt<double> GradNullRatio;
extern llvm::cl::opt<bool> GradFloorAbort;

// Candidate classes.
extern llvm::cl::opt<bool> EnableHerbie;
extern llvm::cl::opt<bool> EnablePT;
extern llvm::cl::opt<bool> EnableMultifloat;
extern llvm::cl::opt<unsigned> ExpansionComponents;
extern llvm::cl::opt<bool> EnableThreeTier;
extern llvm::cl::opt<int> TwoTierStep;
extern llvm::cl::opt<int> ThreeTierStep;
extern llvm::cl::opt<bool> RaiseWMMA;
extern llvm::cl::opt<bool> RaiseHostGemm;
extern llvm::cl::opt<unsigned> MaxExprDepth;
extern llvm::cl::opt<unsigned> MaxExprLength;
extern llvm::cl::opt<unsigned> MinUsesSplit;
extern llvm::cl::opt<unsigned> MinOpsSplit;
extern llvm::cl::opt<bool> ReductionSubgraphs;
extern llvm::cl::opt<bool> MergeSharedStaging;

// Solver and budgets.
extern llvm::cl::opt<int64_t> CompCostBudget;
extern llvm::cl::opt<double> Tau;
extern llvm::cl::opt<double> Confidence;
extern llvm::cl::opt<double> CostTieBandRel;
extern llvm::cl::opt<bool> TauCheapest;
extern llvm::cl::opt<bool> EarlyPrune;
extern llvm::cl::opt<bool> JointDP;
extern llvm::cl::opt<bool> AggressiveDCE;
extern llvm::cl::opt<std::string> ApplyRewrites;

// Cost and accuracy model.
extern llvm::cl::opt<std::string> CostModel;
extern llvm::cl::opt<std::string> ScalarTypes;
extern llvm::cl::opt<unsigned> NumSamples;
extern llvm::cl::opt<unsigned> RandomSeed;
extern llvm::cl::opt<unsigned> SampleLogBits;
extern llvm::cl::opt<bool> CancellationSampling;
extern llvm::cl::opt<double> CancellationFraction;
extern llvm::cl::opt<double> CancellationThreshold;
extern llvm::cl::opt<unsigned> AccuracyReferenceBits;
extern llvm::cl::opt<bool> RelativeError;
extern llvm::cl::opt<unsigned> MaxMPFRPrec;
extern llvm::cl::opt<bool> StrictMode;
extern llvm::cl::opt<double> ExponentPenalty;
extern llvm::cl::opt<double> NonfinitePenalty;
extern llvm::cl::opt<bool> FreqWeightedPricing;

// Herbie and the result cache.
extern llvm::cl::opt<std::string> Cache;
extern llvm::cl::opt<int> HerbieNumThreads;
extern llvm::cl::opt<int> HerbieTimeout;
extern llvm::cl::opt<int> HerbieNumPts;
extern llvm::cl::opt<int> HerbieNumIters;
extern llvm::cl::opt<int> HerbieNumEnodes;
extern llvm::cl::opt<unsigned> HerbieSubgraphTimeout;
extern llvm::cl::opt<std::string> HerbieBinary;
extern llvm::cl::opt<std::string> HerbiePlatform;

// Matmul raising and host dispatch.
extern llvm::cl::opt<bool> OzakiHostDispatch;
extern llvm::cl::opt<unsigned> OzakiForceNm;
extern llvm::cl::opt<bool> OzakiNativeDgemm;
extern llvm::cl::opt<bool> InKernelCalibration;
extern llvm::cl::opt<unsigned> InKernelSharedCap;

// Materialization.
extern llvm::cl::opt<bool> StageParamArrays;
extern llvm::cl::opt<bool> NarrowParamStaging;
extern llvm::cl::opt<bool> NarrowStagingSpeculative;

// Diagnostics.
extern llvm::cl::opt<bool> Print;
extern llvm::cl::opt<bool> ShowTable;
extern llvm::cl::opt<std::string> ReportPath;

} // namespace flags
} // namespace poseidon
#endif // POSEIDON_FLAGS_H
