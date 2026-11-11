//=- Options.h - what one compile was asked to do --------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One field per -poseidon-* flag, with the flag's type and meaning, grouped as
// Pass/Flags.cpp declares the flags. Pass/Flags.cpp reads the command line into
// one Options per compile and is the only reader of a flag. A field's
// initializer is its flag's default, the one place it is written (the flag
// takes it from a default-constructed Options), so a default-constructed
// Options is a compile given no flag. Defaults are part of the paper
// configuration: changing one changes every banked result.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_OPTIONS_H
#define POSEIDON_OPTIONS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace poseidon {

// Option values by flag name, the name without its leading dash.
using FlagValues = llvm::SmallVector<std::pair<llvm::StringRef, std::string>>;

// Each group appends the name and value of every option a DP table depends on
// (appendSolveInputs) and of every other option (appendOtherValues).
struct ProfilingOptions {
  bool ProfileGenerate = false;
  // The directory a solve reads its profiles from; empty when no
  // -poseidon-profile-use was given. Given without a directory, the one a
  // profiling run writes by default.
  std::string ProfileUse;
  std::vector<std::string> OutputProfiles;
  std::string Kernels;
  double MinCostShare = 0.01;
  bool LooseCoverage = false;

  // The compile solves from profiles.
  bool usesProfile() const { return !ProfileUse.empty(); }
  // Whether this compile generates or uses a profile: anything else leaves
  // every site as written.
  bool profileAction() const { return ProfileGenerate || usesProfile(); }

  void appendSolveInputs(FlagValues &out) const;
  void appendOtherValues(FlagValues &out) const;
};

struct CandidateOptions {
  bool EnableHerbie = true;
  bool EnablePT = true;
  bool EnableMultifloat = true;
  unsigned ExpansionComponents = 2;
  bool EnableThreeTier = true;
  int TwoTierStep = 10;
  bool RaiseWMMA = true;
  unsigned MaxExprLength = 10000;
  unsigned MinUsesSplit = 99;
  unsigned MinOpsSplit = 99;
  bool ReductionSubgraphs = false;
  bool MergeSharedStaging = false;
  std::vector<std::string> SplitCalls;

  void appendSolveInputs(FlagValues &out) const;
  // ReductionSubgraphs and MergeSharedStaging, which the DP key lists after
  // the matrix options.
  void appendLaterSolveInputs(FlagValues &out) const;
};

struct SolverOptions {
  // The share of the solve's profiled cost a solution may add; a negative
  // share is one it has to save.
  double CostBudget = 0.0;
  double Tau = 0.0;
  bool TauCheapest = true;
  bool EarlyPrune = true;
  bool JointDP = false;
  bool AggressiveDCE = false;
  std::string ApplyRewrites;

  void appendSolveInputs(FlagValues &out) const;
  void appendOtherValues(FlagValues &out) const;
};

// The cost and accuracy model.
struct ModelOptions {
  std::string CostModel;
  unsigned NumSamples = 1024;
  unsigned RandomSeed = 239778888;
  unsigned SampleLogBits = 0;
  bool StrictMode = false;
  unsigned MinArmSamples = 32;
  unsigned ArmSearchFactor = 64;
  double MaxBrokenShare = 0.01;
  double ExponentPenalty = 1e30;
  bool FreqWeightedPricing = false;

  void appendSolveInputs(FlagValues &out) const;
  // FreqWeightedPricing, which the DP key lists after the matrix options.
  void appendLaterSolveInputs(FlagValues &out) const;
  void appendOtherValues(FlagValues &out) const;
};

// Herbie and the result cache.
struct HerbieOptions {
  std::string Cache = "./.poseidon";
  std::string Unit;
  std::string ImportAccuracyMemo;
  std::string ImportHerbieAnswers;
  int HerbieNumThreads = 8;
  int HerbieTimeout = 9999;
  int HerbieNumPts = 1024;
  int HerbieNumIters = 6;
  int HerbieNumEnodes = 8000;
  // Ascending, each once; empty passes Herbie no term count.
  std::vector<unsigned> HerbieNumTaylorTerms;
  unsigned HerbieSubgraphTimeout = 1800;
  unsigned HerbieProcesses = 0;
  unsigned HerbiePreFloorBits = 0;
  std::string HerbieBinary;
  std::string HerbiePlatform;
  bool HerbieArrays = true;
  std::string HerbieArrayWeights = "sensitivity";
  std::string HerbieArrayGrouping = "inputs";

  void appendSolveInputs(FlagValues &out) const;
  void appendOtherValues(FlagValues &out) const;
};

// Matrix raising and host dispatch.
struct MatmulOptions {
  bool OzakiHostDispatch = true;
  bool InKernelCalibration = false;

  void appendSolveInputs(FlagValues &out) const;
};

struct MaterializationOptions {
  bool NarrowParamStaging = false;
  bool NarrowMultiviewStaging = false;

  void appendSolveInputs(FlagValues &out) const;
};

struct DiagnosticOptions {
  bool Print = false;
  bool ShowTable = false;
  std::string ReportPath;
  std::string Explain;

  void appendOtherValues(FlagValues &out) const;
};

struct Options {
  ProfilingOptions Profiling;
  CandidateOptions Candidates;
  SolverOptions Solver;
  ModelOptions Model;
  HerbieOptions Herbie;
  MatmulOptions Matmul;
  MaterializationOptions Materialization;
  DiagnosticOptions Diagnostics;
};

// The name and value of every option a DP table depends on, that is every
// option that changes the candidate set or its pricing, in the order the DP
// key lists them.
FlagValues solveFlagValues(const Options &O);
// What a solved table does not depend on: what is profiled or which sites are
// solved, what is chosen from a solved table, a path whose contents the key
// holds instead (profiles, cost model, Herbie binary and platform), where
// results computed earlier are kept or imported from or handed to the host
// half, a solve that never reads a table (joint, manual), how long Herbie may
// run and on how many workers, and output.
FlagValues notSolveInputs(const Options &O);
// Both, in that order.
FlagValues allFlagValues(const Options &O);

} // namespace poseidon
#endif // POSEIDON_OPTIONS_H
