#ifndef ENZYME_POSEIDON_H
#define ENZYME_POSEIDON_H

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <mpfr.h>

namespace llvm {
class FunctionPass;
class TargetTransformInfo;
class Function;
class Module;
class AnalysisUsage;
} // namespace llvm

using namespace llvm;

extern "C" {
extern llvm::cl::opt<bool> EnzymeEnableFPOpt;
extern llvm::cl::opt<bool> EnzymePrintFPOpt;
extern llvm::cl::opt<bool> FPOptPrintPreproc;
}

extern llvm::cl::opt<bool> EnzymePrintHerbie;
extern llvm::cl::opt<std::string> FPOptLogPath;
extern llvm::cl::opt<std::string> FPOptCostModelPath;
extern llvm::cl::opt<std::string> FPOptTargetFuncRegex;
extern llvm::cl::opt<bool> FPOptEnableHerbie;
extern llvm::cl::opt<bool> FPOptEnablePT;
extern llvm::cl::opt<int> HerbieNumThreads;
extern llvm::cl::opt<int> HerbieTimeout;
extern llvm::cl::opt<std::string> FPOptCachePath;
extern llvm::cl::opt<int> HerbieNumPoints;
extern llvm::cl::opt<int> HerbieNumIters;
extern llvm::cl::opt<bool> HerbieDisableNumerics;
extern llvm::cl::opt<bool> HerbieDisableArithmetic;
extern llvm::cl::opt<bool> HerbieDisableFractions;
extern llvm::cl::opt<bool> HerbieDisableTaylor;
extern llvm::cl::opt<bool> HerbieDisableSetupSimplify;
extern llvm::cl::opt<bool> HerbieDisableGenSimplify;
extern llvm::cl::opt<bool> HerbieDisableRegime;
extern llvm::cl::opt<bool> HerbieDisableBranchExpr;
extern llvm::cl::opt<bool> HerbieDisableAvgError;
extern llvm::cl::opt<bool> FPOptEnableSolver;
extern llvm::cl::opt<std::string> FPOptSolverType;
extern llvm::cl::opt<bool> FPOptStrictMode;
extern llvm::cl::opt<std::string> FPOptReductionProf;
extern llvm::cl::opt<std::string> FPOptReductionEval;
extern llvm::cl::opt<double> FPOptGeoMeanEps;
extern llvm::cl::opt<bool> FPOptLooseCoverage;
extern llvm::cl::opt<bool> FPOptShowTable;
extern llvm::cl::list<int64_t> FPOptShowTableCosts;
extern llvm::cl::opt<bool> FPOptShowPTDetails;
extern llvm::cl::opt<int64_t> FPOptComputationCostBudget;
extern llvm::cl::opt<unsigned> FPOptMaxFPCCDepth;
extern llvm::cl::opt<unsigned> FPOptMaxExprDepth;
extern llvm::cl::opt<unsigned> FPOptMaxExprLength;
extern llvm::cl::opt<unsigned> FPOptRandomSeed;
extern llvm::cl::opt<unsigned> FPOptNumSamples;
extern llvm::cl::opt<unsigned> FPOptMaxMPFRPrec;
extern llvm::cl::opt<double> FPOptWidenRange;
extern llvm::cl::opt<bool> FPOptEarlyPrune;
extern llvm::cl::opt<double> FPOptCostDominanceThreshold;
extern llvm::cl::opt<double> FPOptAccuracyDominanceThreshold;


// Classes

class FPOpt final : public FunctionPass {
public:
  static char ID;
  FPOpt();

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};

llvm::FunctionPass *createFPOptPass();

class FPOptNewPM final : public llvm::AnalysisInfoMixin<FPOptNewPM> {
  friend struct llvm::AnalysisInfoMixin<FPOptNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  FPOptNewPM() {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

#endif // ENZYME_POSEIDON_H
