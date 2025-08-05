#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Demangle/Demangle.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <mpfr.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "Poseidon.h"
#include "PoseidonEvaluators.h"
#include "PoseidonHerbieUtils.h"
#include "PoseidonPrecUtils.h"
#include "PoseidonTypes.h"
#include "PoseidonUtils.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "fp-opt"

extern "C" {
cl::opt<bool> EnzymeEnableFPOpt("enzyme-enable-fpopt", cl::init(false),
                                cl::Hidden, cl::desc("Run the FPOpt pass"));
cl::opt<bool> EnzymePrintFPOpt("enzyme-print-fpopt", cl::init(false),
                               cl::Hidden,
                               cl::desc("Enable Enzyme to print FPOpt info"));
cl::opt<bool> FPOptPrintPreproc(
    "fpopt-print-preproc", cl::init(false), cl::Hidden,
    cl::desc("Enable Enzyme to print FPOpt preprocesing info"));
}

// Command line options that are not in extern "C"
cl::opt<bool>
    EnzymePrintHerbie("enzyme-print-herbie", cl::init(false), cl::Hidden,
                      cl::desc("Enable Enzyme to print Herbie expressions"));
cl::opt<std::string>
    FPOptLogPath("fpopt-log-path", cl::init(""), cl::Hidden,
                 cl::desc("Which log to use in the FPOpt pass"));
cl::opt<std::string>
    FPOptCostModelPath("fpopt-cost-model-path", cl::init(""), cl::Hidden,
                       cl::desc("Use a custom cost model in the FPOpt pass"));
cl::opt<std::string> FPOptTargetFuncRegex(
    "fpopt-target-func-regex", cl::init(".*"), cl::Hidden,
    cl::desc("Regex pattern to match target functions in the FPOpt pass"));
cl::opt<bool> FPOptEnableHerbie(
    "fpopt-enable-herbie", cl::init(true), cl::Hidden,
    cl::desc("Use Herbie to rewrite floating-point expressions"));
cl::opt<bool> FPOptEnablePT(
    "fpopt-enable-pt", cl::init(false), cl::Hidden,
    cl::desc("Consider precision changes of floating-point expressions"));
cl::opt<int> HerbieNumThreads("herbie-num-threads", cl::init(1), cl::Hidden,
                              cl::desc("Number of threads Herbie uses"));
cl::opt<int> HerbieTimeout("herbie-timeout", cl::init(120), cl::Hidden,
                           cl::desc("Herbie's timeout to use for each "
                                    "candidate expressions."));
cl::opt<std::string> FPOptCachePath("fpopt-cache-path", cl::init("cache"),
                                    cl::Hidden,
                                    cl::desc("Path to cache Herbie results"));
cl::opt<int>
    HerbieNumPoints("herbie-num-pts", cl::init(1024), cl::Hidden,
                    cl::desc("Number of input points Herbie uses to evaluate "
                             "candidate expressions."));
cl::opt<int> HerbieNumIters(
    "herbie-num-iters", cl::init(6), cl::Hidden,
    cl::desc("Number of times Herbie attempts to improve accuracy."));
cl::opt<bool> HerbieDisableNumerics(
    "herbie-disable-numerics", cl::init(false), cl::Hidden,
    cl::desc("Disable Herbie rewrite rules that produce numerical shorthands "
             "expm1, log1p, fma, and hypot"));
cl::opt<bool> HerbieDisableArithmetic(
    "herbie-disable-arithmetic", cl::init(false), cl::Hidden,
    cl::desc("Disable Herbie rewrite rules on basic arithmetic fasts."));
cl::opt<bool> HerbieDisableFractions(
    "herbie-disable-fractions", cl::init(false), cl::Hidden,
    cl::desc("Disable Herbie rewrite rules on fraction arithmetic."));
cl::opt<bool>
    HerbieDisableTaylor("herbie-disable-taylor", cl::init(false), cl::Hidden,
                        cl::desc("Disable Herbie's series expansion"));
cl::opt<bool> HerbieDisableSetupSimplify(
    "herbie-disable-setup-simplify", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from pre-simplifying expressions"));
cl::opt<bool> HerbieDisableGenSimplify(
    "herbie-disable-gen-simplify", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from simplifying expressions "
             "during the main improvement loop"));
cl::opt<bool> HerbieDisableRegime(
    "herbie-disable-regime", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from branching between expressions candidates"));
cl::opt<bool> HerbieDisableBranchExpr(
    "herbie-disable-branch-expr", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from branching on expressions"));
cl::opt<bool> HerbieDisableAvgError(
    "herbie-disable-avg-error", cl::init(false), cl::Hidden,
    cl::desc("Make Herbie choose the candidates with the least maximum error"));
cl::opt<bool> FPOptEnableSolver(
    "fpopt-enable-solver", cl::init(false), cl::Hidden,
    cl::desc("Use the solver to select desirable rewrite candidates; when "
             "disabled, apply all Herbie's first choices"));
cl::opt<std::string> FPOptSolverType("fpopt-solver-type", cl::init("dp"),
                                     cl::Hidden,
                                     cl::desc("Which solver to use; "
                                              "either 'dp' or 'greedy'"));
cl::opt<bool> FPOptStrictMode(
    "fpopt-strict-mode", cl::init(false), cl::Hidden,
    cl::desc(
        "Discard all FPOpt candidates that produce NaN or inf outputs for any "
        "input point that originally produced finite outputs"));
cl::opt<std::string> FPOptReductionProf(
    "fpopt-reduction-prof", cl::init("geomean"), cl::Hidden,
    cl::desc("Which reduction result to extract from profiles. "
             "Options are 'geomean', 'arithmean', and 'maxabs'"));
cl::opt<std::string> FPOptReductionEval(
    "fpopt-reduction-eval", cl::init("geomean"), cl::Hidden,
    cl::desc("Which reduction result to use in candidate evaluation. "
             "Options are 'geomean', 'arithmean', and 'maxabs'"));
cl::opt<double> FPOptGeoMeanEps(
    "fpopt-geo-mean-eps", cl::init(0.0), cl::Hidden,
    cl::desc("The offset used in the geometric mean "
             "calculation; if = 0, zeros are replaced with ULPs"));
cl::opt<bool> FPOptLooseCoverage(
    "fpopt-loose-coverage", cl::init(false), cl::Hidden,
    cl::desc("Allow unexecuted FP instructions in subgraph indentification"));
cl::opt<bool> FPOptShowTable(
    "fpopt-show-table", cl::init(false), cl::Hidden,
    cl::desc(
        "Print the full DP table (highly verbose for large applications)"));
cl::list<int64_t> FPOptShowTableCosts(
    "fpopt-show-table-costs", cl::ZeroOrMore, cl::CommaSeparated, cl::Hidden,
    cl::desc(
        "Comma-separated list of computation costs for which to print DP table "
        "entries. If provided, only specified computation costs are "
        "printed. "));
cl::opt<bool> FPOptShowPTDetails(
    "fpopt-show-pt-details", cl::init(false), cl::Hidden,
    cl::desc("Print details of precision tuning candidates along with the DP "
             "table (highly verbose for large applications)"));
cl::opt<int64_t> FPOptComputationCostBudget(
    "fpopt-comp-cost-budget", cl::init(100000000000L), cl::Hidden,
    cl::desc("The maximum computation cost budget for the solver"));
// TODO: Fix this
cl::opt<unsigned> FPOptMaxFPCCDepth(
    "fpopt-max-fpcc-depth", cl::init(99999), cl::Hidden,
    cl::desc("The maximum depth of a floating-point connected component"));
cl::opt<unsigned> FPOptMaxExprDepth(
    "fpopt-max-expr-depth", cl::init(100), cl::Hidden,
    cl::desc(
        "The maximum depth of expression construction; abort if exceeded"));
cl::opt<unsigned> FPOptMaxExprLength(
    "fpopt-max-expr-length", cl::init(10000), cl::Hidden,
    cl::desc("The maximum length of an expression; abort if exceeded"));
cl::opt<unsigned>
    FPOptRandomSeed("fpopt-random-seed", cl::init(239778888), cl::Hidden,
                    cl::desc("The random seed used in the FPOpt pass"));
cl::opt<unsigned>
    FPOptNumSamples("fpopt-num-samples", cl::init(1024), cl::Hidden,
                    cl::desc("Number of sampled points for input hypercube"));
cl::opt<unsigned>
    FPOptMaxMPFRPrec("fpopt-max-mpfr-prec", cl::init(1024), cl::Hidden,
                     cl::desc("Max precision for MPFR gold value computation"));
cl::opt<double>
    FPOptWidenRange("fpopt-widen-range", cl::init(1), cl::Hidden,
                    cl::desc("Ablation study only: widen the range of input "
                             "hypercube by this factor"));
cl::opt<bool> FPOptEarlyPrune(
    "fpopt-early-prune", cl::init(false), cl::Hidden,
    cl::desc("Prune dominated candidates in expression transformation phases"));
cl::opt<double> FPOptCostDominanceThreshold(
    "fpopt-cost-dom-thres", cl::init(0.05), cl::Hidden,
    cl::desc("The threshold for cost dominance in DP solver"));
cl::opt<double> FPOptAccuracyDominanceThreshold(
    "fpopt-acc-dom-thres", cl::init(0.05), cl::Hidden,
    cl::desc("The threshold for accuracy dominance in DP solver"));

#if LLVM_VERSION_MAJOR >= 21
#define GET_INSTRUCTION_COST(cost) (cost.getValue())
#else
#define GET_INSTRUCTION_COST(cost) (cost.getValue().value())
#endif

const std::unordered_set<std::string> LibmFuncs = {
    "sin",   "cos",   "tan",      "asin",  "acos",   "atan",  "atan2",
    "sinh",  "cosh",  "tanh",     "asinh", "acosh",  "atanh", "exp",
    "log",   "sqrt",  "cbrt",     "pow",   "fabs",   "fma",   "hypot",
    "expm1", "log1p", "ceil",     "floor", "erf",    "exp2",  "lgamma",
    "log10", "log2",  "rint",     "round", "tgamma", "trunc", "copysign",
    "fdim",  "fmod",  "remainder"};


void splitFPCC(FPCC &CC, SmallVector<FPCC, 1> &newCCs) {
  std::unordered_map<Instruction *, int> shortestDistances;

  for (auto &op : CC.operations) {
    shortestDistances[op] = std::numeric_limits<int>::max();
  }

  // find the shortest distance from inputs to each operation
  for (auto input : CC.inputs) {
    if (isa<Constant>(input)) {
      continue;
    }

    SmallVector<std::pair<Instruction *, int>, 8> todo;
    for (auto user : input->users()) {
      if (auto *I = dyn_cast<Instruction>(user)) {
        if (CC.operations.count(I))
          todo.emplace_back(I, 1);
      }
    }

    while (!todo.empty()) {
      auto [cur, dist] = todo.pop_back_val();
      if (dist < shortestDistances[cur]) {
        shortestDistances[cur] = dist;
        for (auto user : cur->users()) {
          if (auto *I = dyn_cast<Instruction>(user);
              I && CC.operations.count(I)) {
            todo.emplace_back(I, dist + 1);
          }
        }
      }
    }
  }

  // llvm::errs() << "Shortest distances:\n";
  // for (auto &[op, dist] : shortestDistances) {
  //   llvm::errs() << *op << ": " << dist << "\n";
  // }

  int maxDepth =
      std::max_element(shortestDistances.begin(), shortestDistances.end(),
                       [](const auto &lhs, const auto &rhs) {
                         return lhs.second < rhs.second;
                       })
          ->second;

  if (maxDepth <= FPOptMaxFPCCDepth) {
    newCCs.push_back(CC);
    return;
  }

  newCCs.resize(maxDepth / FPOptMaxFPCCDepth + 1);

  // Split `operations` based on the shortest distance
  for (const auto &[op, dist] : shortestDistances) {
    newCCs[dist / FPOptMaxFPCCDepth].operations.insert(op);
  }

  // Reconstruct `inputs` and `outputs` for new components
  for (auto &newCC : newCCs) {
    for (auto &op : newCC.operations) {
      auto operands =
          isa<CallInst>(op) ? cast<CallInst>(op)->args() : op->operands();
      for (auto &operand : operands) {
        if (newCC.inputs.count(operand) || isa<Constant>(operand)) {
          continue;
        }

        // Original non-Poseidonable operands or Poseidonable intermediate
        // operations
        if (CC.inputs.count(operand) ||
            !newCC.operations.count(cast<Instruction>(operand))) {
          newCC.inputs.insert(operand);
        }
      }

      for (auto user : op->users()) {
        if (auto *I = dyn_cast<Instruction>(user);
            I && !newCC.operations.count(I)) {
          newCC.outputs.insert(op);
        }
      }
    }
  }

  if (EnzymePrintFPOpt) {
    llvm::errs() << "Splitting the FPCC into " << newCCs.size()
                 << " components\n";
  }
}

void setUnifiedAccuracyCost(
    ApplicableFPCC &ACC,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

bool extractValueFromLog(const std::string &logPath,
                         const std::string &functionName, size_t blockIdx,
                         size_t instIdx, ValueInfo &data) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    llvm_unreachable("Failed to open log file");
  }

  std::string line;
  std::regex valuePattern("^Value:" + functionName + ":" +
                          std::to_string(blockIdx) + ":" +
                          std::to_string(instIdx) + "$");

  std::regex newEntryPattern("^(Value|Grad|Error):");

  std::regex minResPattern(R"(^\s*MinRes\s*=\s*([\d\.eE+\-]+))");
  std::regex maxResPattern(R"(^\s*MaxRes\s*=\s*([\d\.eE+\-]+))");
  std::regex executionsPattern(R"(^\s*Executions\s*=\s*(\d+))");
  std::regex geoMeanPattern(R"(^\s*GeoMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex arithMeanPattern(R"(^\s*ArithMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex maxAbsPattern(R"(^\s*MaxAbs\s*=\s*([\d\.eE+\-]+))");

  std::regex operandPattern(
      R"(^\s*Operand\[(\d+)\]\s*=\s*\[([\d\.eE+\-]+),\s*([\d\.eE+\-]+)\])");

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (std::regex_search(line, valuePattern)) {
      std::string minResLine, maxResLine, execLine;
      std::string geoMeanLine, arithMeanLine, maxAbsLine;

      if (std::getline(file, minResLine) && std::getline(file, maxResLine) &&
          std::getline(file, execLine) && std::getline(file, geoMeanLine) &&
          std::getline(file, arithMeanLine) && std::getline(file, maxAbsLine)) {

        auto stripCR = [](std::string &s) {
          if (!s.empty() && s.back() == '\r') {
            s.pop_back();
          }
        };
        stripCR(minResLine);
        stripCR(maxResLine);
        stripCR(execLine);
        stripCR(geoMeanLine);
        stripCR(arithMeanLine);
        stripCR(maxAbsLine);

        std::smatch mMinRes, mMaxRes, mExec, mGeo, mArith, mMaxAbs;
        if (std::regex_search(minResLine, mMinRes, minResPattern) &&
            std::regex_search(maxResLine, mMaxRes, maxResPattern) &&
            std::regex_search(execLine, mExec, executionsPattern) &&
            std::regex_search(geoMeanLine, mGeo, geoMeanPattern) &&
            std::regex_search(arithMeanLine, mArith, arithMeanPattern) &&
            std::regex_search(maxAbsLine, mMaxAbs, maxAbsPattern)) {

          data.minRes = stringToDouble(mMinRes[1]);
          data.maxRes = stringToDouble(mMaxRes[1]);
          data.executions = static_cast<unsigned>(std::stoul(mExec[1]));
          data.geoMean = stringToDouble(mGeo[1]);
          data.arithMean = stringToDouble(mArith[1]);
          data.maxAbs = stringToDouble(mMaxAbs[1]);
        } else {
          std::string error =
              "Failed to parse stats for: Function: " + functionName +
              ", BlockIdx: " + std::to_string(blockIdx) +
              ", InstIdx: " + std::to_string(instIdx);
          llvm_unreachable(error.c_str());
        }
      } else {
        std::string error =
            "Incomplete stats block for: Function: " + functionName +
            ", BlockIdx: " + std::to_string(blockIdx) +
            ", InstIdx: " + std::to_string(instIdx);
        llvm_unreachable(error.c_str());
      }

      while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
          line.pop_back();

        if (std::regex_search(line, newEntryPattern)) {
          if (FPOptWidenRange != 1.0) {
            double center = (data.minRes + data.maxRes) / 2.0;
            double half_range = (data.maxRes - data.minRes) / 2.0;
            double new_half_range = half_range * FPOptWidenRange;
            data.minRes = center - new_half_range;
            data.maxRes = center + new_half_range;

            for (size_t i = 0; i < data.minOperands.size(); ++i) {
              double op_center =
                  (data.minOperands[i] + data.maxOperands[i]) / 2.0;
              double op_half_range =
                  (data.maxOperands[i] - data.minOperands[i]) / 2.0;
              double op_new_half_range = op_half_range * FPOptWidenRange;
              data.minOperands[i] = op_center - op_new_half_range;
              data.maxOperands[i] = op_center + op_new_half_range;
            }
          }
          return true;
        }

        std::smatch operandMatch;
        if (std::regex_search(line, operandMatch, operandPattern)) {
          unsigned opIdx = static_cast<unsigned>(std::stoul(operandMatch[1]));
          double minVal = stringToDouble(operandMatch[2]);
          double maxVal = stringToDouble(operandMatch[3]);

          if (opIdx >= data.minOperands.size()) {
            data.minOperands.resize(opIdx + 1, 0.0);
            data.maxOperands.resize(opIdx + 1, 0.0);
          }
          data.minOperands[opIdx] = minVal;
          data.maxOperands[opIdx] = maxVal;
        }
      }
    }
  }

  llvm::errs() << "Failed to extract value info for: Function: " << functionName
               << ", BlockIdx: " << blockIdx << ", InstIdx: " << instIdx
               << "\n";
  return false;
}

bool extractGradFromLog(const std::string &logPath,
                        const std::string &functionName, size_t blockIdx,
                        size_t instIdx, GradInfo &data) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    llvm_unreachable("Failed to open log file");
  }

  std::string line;
  std::regex gradPattern("^Grad:" + functionName + ":" +
                         std::to_string(blockIdx) + ":" +
                         std::to_string(instIdx) + "$");

  std::regex geoMeanPattern(R"(^\s*GeoMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex arithMeanPattern(R"(^\s*ArithMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex maxAbsPattern(R"(^\s*MaxAbs\s*=\s*([\d\.eE+\-]+))");

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (std::regex_search(line, gradPattern)) {
      std::string geoMeanLine, arithMeanLine, maxAbsLine;
      if (std::getline(file, geoMeanLine) &&
          std::getline(file, arithMeanLine) && std::getline(file, maxAbsLine)) {

        auto stripCR = [](std::string &s) {
          if (!s.empty() && s.back() == '\r') {
            s.pop_back();
          }
        };
        stripCR(geoMeanLine);
        stripCR(arithMeanLine);
        stripCR(maxAbsLine);

        std::smatch mGeo, mArith, mMax;
        if (std::regex_search(geoMeanLine, mGeo, geoMeanPattern) &&
            std::regex_search(arithMeanLine, mArith, arithMeanPattern) &&
            std::regex_search(maxAbsLine, mMax, maxAbsPattern)) {

          data.geoMean = stringToDouble(mGeo[1]);
          data.arithMean = stringToDouble(mArith[1]);
          data.maxAbs = stringToDouble(mMax[1]);
          return true;
        }
      }

      llvm::errs() << "Incomplete gradient block for: Function: "
                   << functionName << ", BlockIdx: " << blockIdx
                   << ", InstIdx: " << instIdx << "\n";
      return false;
    }
  }

  llvm::errs() << "Failed to extract gradient for: Function: " << functionName
               << ", BlockIdx: " << blockIdx << ", InstIdx: " << instIdx
               << "\n";
  return false;
}

bool isLogged(const std::string &logPath, const std::string &functionName) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    assert(0 && "Failed to open log file");
  }

  std::regex functionRegex("^Value:" + functionName);

  std::string line;
  while (std::getline(file, line)) {
    if (std::regex_search(line, functionRegex)) {
      return true;
    }
  }

  return false;
}

// Given the cost budget `FPOptComputationCostBudget`, we want to minimize the
// accuracy cost of the rewritten expressions.
bool accuracyGreedySolver(
    SmallVector<ApplicableOutput, 4> &AOs, SmallVector<ApplicableFPCC, 4> &ACCs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy greedy solver with computation budget: "
               << FPOptComputationCostBudget << "\n";
  InstructionCost totalComputationCost = 0;

  SmallVector<size_t, 4> aoIndices;
  for (size_t i = 0; i < AOs.size(); ++i) {
    aoIndices.push_back(i);
  }
  std::mt19937 g(FPOptRandomSeed);
  std::shuffle(aoIndices.begin(), aoIndices.end(), g);

  for (size_t idx : aoIndices) {
    auto &AO = AOs[idx];
    int bestCandidateIndex = -1;
    double bestAccuracyCost = std::numeric_limits<double>::infinity();
    InstructionCost bestCandidateComputationCost;

    for (const auto &candidate : enumerate(AO.candidates)) {
      size_t i = candidate.index();
      auto candCompCost = AO.getCompCostDelta(i);
      auto candAccCost = AO.getAccCostDelta(i);
      // llvm::errs() << "AO Candidate " << i << " for " << AO.expr
      //              << " has accuracy cost: " << candAccCost
      //              << " and computation cost: " << candCompCost << "\n";

      if (totalComputationCost + candCompCost <= FPOptComputationCostBudget) {
        if (candAccCost < bestAccuracyCost) {
          // llvm::errs() << "AO Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candAccCost;
          bestCandidateComputationCost = candCompCost;
        }
      }
    }

    if (bestCandidateIndex != -1) {
      AO.apply(bestCandidateIndex, valueToNodeMap, symbolToValueMap);
      changed = true;
      totalComputationCost += bestCandidateComputationCost;
      if (EnzymePrintFPOpt) {
        llvm::errs() << "Greedy solver selected candidate "
                     << bestCandidateIndex << " for " << AO.expr
                     << " with accuracy cost: " << bestAccuracyCost
                     << " and computation cost: "
                     << bestCandidateComputationCost << "\n";
      }
    }
  }

  SmallVector<size_t, 4> accIndices;
  for (size_t i = 0; i < ACCs.size(); ++i) {
    accIndices.push_back(i);
  }
  std::shuffle(accIndices.begin(), accIndices.end(), g);

  for (size_t idx : accIndices) {
    auto &ACC = ACCs[idx];
    int bestCandidateIndex = -1;
    double bestAccuracyCost = std::numeric_limits<double>::infinity();
    InstructionCost bestCandidateComputationCost;

    for (const auto &candidate : enumerate(ACC.candidates)) {
      size_t i = candidate.index();
      auto candCompCost = ACC.getCompCostDelta(i);
      auto candAccCost = ACC.getAccCostDelta(i);
      // llvm::errs() << "ACC Candidate " << i << " (" << candidate.value().desc
      //              << ") has accuracy cost: " << candAccCost
      //              << " and computation cost: " << candCompCost << "\n";

      if (totalComputationCost + candCompCost <= FPOptComputationCostBudget) {
        if (candAccCost < bestAccuracyCost) {
          // llvm::errs() << "ACC Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candAccCost;
          bestCandidateComputationCost = candCompCost;
        }
      }
    }

    if (bestCandidateIndex != -1) {
      ACC.apply(bestCandidateIndex);
      changed = true;
      totalComputationCost += bestCandidateComputationCost;
      if (EnzymePrintFPOpt) {
        llvm::errs() << "Greedy solver selected candidate "
                     << bestCandidateIndex << " for "
                     << ACC.candidates[bestCandidateIndex].desc
                     << " with accuracy cost: " << bestAccuracyCost
                     << " and computation cost: "
                     << bestCandidateComputationCost << "\n";
      }
    }
  }

  llvm::errs() << "Greedy solver finished with total computation cost: "
               << totalComputationCost
               << "; total allowance: " << FPOptComputationCostBudget << "\n";

  return changed;
}

bool accuracyDPSolver(
    SmallVector<ApplicableOutput, 4> &AOs, SmallVector<ApplicableFPCC, 4> &ACCs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << FPOptComputationCostBudget << "\n";

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;

  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  CostMap newCostToAccuracyMap;
  SolutionMap newCostToSolutionMap;
  CostMap prunedCostToAccuracyMap;
  SolutionMap prunedCostToSolutionMap;

  std::string cacheFilePath = FPOptCachePath + "/table.json";

  if (llvm::sys::fs::exists(cacheFilePath)) {
    llvm::errs() << "Cache file found. Loading DP tables from cache.\n";

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFile(cacheFilePath);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Error reading cache file: " << ec.message() << "\n";
      return changed;
    }
    llvm::StringRef buffer = fileOrErr.get()->getBuffer();
    llvm::Expected<llvm::json::Value> jsonOrErr = llvm::json::parse(buffer);
    if (!jsonOrErr) {
      llvm::errs() << "Error parsing JSON from cache file: "
                   << llvm::toString(jsonOrErr.takeError()) << "\n";
      return changed;
    }

    llvm::json::Object *jsonObj = jsonOrErr->getAsObject();
    if (!jsonObj) {
      llvm::errs() << "Invalid JSON format in cache file.\n";
      return changed;
    }

    if (llvm::json::Object *costAccMap =
            jsonObj->getObject("costToAccuracyMap")) {
      for (auto &pair : *costAccMap) {
        InstructionCost compCost(std::stoll(pair.first.str()));
        double accCost = pair.second.getAsNumber().value();
        costToAccuracyMap[compCost] = accCost;
      }
    } else {
      llvm_unreachable("Invalid costToAccuracyMap in cache file.");
    }

    if (llvm::json::Object *costSolMap =
            jsonObj->getObject("costToSolutionMap")) {
      for (auto &pair : *costSolMap) {
        InstructionCost compCost(std::stoll(pair.first.str()));
        SmallVector<SolutionStep> solutionSteps;

        llvm::json::Array *stepsArray = pair.second.getAsArray();
        if (!stepsArray) {
          llvm::errs() << "Invalid steps array in cache file.\n";
          return changed;
        }

        for (llvm::json::Value &stepVal : *stepsArray) {
          llvm::json::Object *stepObj = stepVal.getAsObject();
          if (!stepObj) {
            llvm_unreachable("Invalid step object in cache file.");
          }

          StringRef itemType = stepObj->getString("itemType").value();
          size_t candidateIndex = stepObj->getInteger("candidateIndex").value();
          size_t itemIndex = stepObj->getInteger("itemIndex").value();

          if (itemType == "AO") {
            if (itemIndex >= AOs.size()) {
              llvm_unreachable("Invalid ApplicableOutput index in cache file.");
            }
            solutionSteps.emplace_back(&AOs[itemIndex], candidateIndex);
          } else if (itemType == "ACC") {
            if (itemIndex >= ACCs.size()) {
              llvm_unreachable("Invalid ApplicableFPCC index in cache file.");
            }
            solutionSteps.emplace_back(&ACCs[itemIndex], candidateIndex);
          } else {
            llvm_unreachable("Invalid itemType in cache file.");
          }
        }

        costToSolutionMap[compCost] = solutionSteps;
      }
    } else {
      llvm::errs() << "costToSolutionMap not found in cache file.\n";
      return changed;
    }

    llvm::errs() << "Loaded DP tables from cache.\n";

  } else {
    llvm::errs() << "Cache file not found. Proceeding to solve DP.\n";

    costToAccuracyMap[0] = 0;
    costToSolutionMap[0] = {};

    std::unordered_map<ApplicableOutput *, size_t> aoPtrToIndex;
    for (size_t i = 0; i < AOs.size(); ++i) {
      aoPtrToIndex[&AOs[i]] = i;
    }
    std::unordered_map<ApplicableFPCC *, size_t> accPtrToIndex;
    for (size_t i = 0; i < ACCs.size(); ++i) {
      accPtrToIndex[&ACCs[i]] = i;
    }

    int AOCounter = 0;

    for (auto &AO : AOs) {
      // It is possible to apply zero candidate for an AO.
      // When no candidate is applied, the resulting accuracy cost
      // and solution steps remain the same.
      newCostToAccuracyMap = costToAccuracyMap;
      newCostToSolutionMap = costToSolutionMap;

      for (const auto &pair : costToAccuracyMap) {
        InstructionCost currCompCost = pair.first;
        double currAccCost = pair.second;

        for (const auto &candidate : enumerate(AO.candidates)) {
          size_t i = candidate.index();
          auto candCompCost = AO.getCompCostDelta(i);
          auto candAccCost = AO.getAccCostDelta(i);

          // Don't ever try to apply a strictly useless candidate
          if (candCompCost >= 0 && candAccCost >= 0.) {
            continue;
          }

          InstructionCost newCompCost = currCompCost + candCompCost;
          double newAccCost = currAccCost + candAccCost;

          // if (EnzymePrintFPOpt)
          //   llvm::errs() << "AO candidate " << i
          //                << " has accuracy cost: " << candAccCost
          //                << " and computation cost: " << candCompCost <<
          //                "\n";

          if (newCostToAccuracyMap.find(newCompCost) ==
                  newCostToAccuracyMap.end() ||
              newCostToAccuracyMap[newCompCost] > newAccCost) {
            newCostToAccuracyMap[newCompCost] = newAccCost;
            newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
            newCostToSolutionMap[newCompCost].emplace_back(&AO, i);
            // if (EnzymePrintFPOpt)
            //   llvm::errs() << "Updating accuracy map (AO candidate " << i
            //                << "): computation cost " << newCompCost
            //                << " -> accuracy cost " << newAccCost << "\n";
          }
        }
      }

      // TODO: Do not prune AO parts of the DP table since AOs influence ACCs
      if (!FPOptEarlyPrune) {
        costToAccuracyMap = newCostToAccuracyMap;
        costToSolutionMap = newCostToSolutionMap;

        llvm::errs() << "##### Finished processing " << ++AOCounter << " of "
                     << AOs.size() << " AOs #####\n";
        llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                     << "\n";
        continue;
      }

      for (const auto &l : newCostToAccuracyMap) {
        InstructionCost currCompCost = l.first;
        double currAccCost = l.second;

        bool dominated = false;
        for (const auto &r : newCostToAccuracyMap) {
          InstructionCost otherCompCost = r.first;
          double otherAccCost = r.second;

          if (currCompCost - otherCompCost >
                  std::fabs(FPOptCostDominanceThreshold *
                            GET_INSTRUCTION_COST(otherCompCost)) &&
              currAccCost - otherAccCost >=
                  std::fabs(FPOptAccuracyDominanceThreshold * otherAccCost)) {
            // if (EnzymePrintFPOpt)
            //   llvm::errs() << "AO candidate with computation cost: "
            //                << currCompCost
            //                << " and accuracy cost: " << currAccCost
            //                << " is dominated by candidate with computation
            //                cost:"
            //                << otherCompCost
            //                << " and accuracy cost: " << otherAccCost << "\n";
            dominated = true;
            break;
          }
        }

        if (!dominated) {
          prunedCostToAccuracyMap[currCompCost] = currAccCost;
          prunedCostToSolutionMap[currCompCost] =
              newCostToSolutionMap[currCompCost];
        }
      }

      costToAccuracyMap = prunedCostToAccuracyMap;
      costToSolutionMap = prunedCostToSolutionMap;
      prunedCostToAccuracyMap.clear();
      prunedCostToSolutionMap.clear();

      llvm::errs() << "##### Finished processing " << ++AOCounter << " of "
                   << AOs.size() << " AOs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    int ACCCounter = 0;

    for (auto &ACC : ACCs) {
      // It is possible to apply zero candidate for an ACC.
      // When no candidate is applied, the resulting accuracy cost
      // and solution steps remain the same.
      newCostToAccuracyMap = costToAccuracyMap;
      newCostToSolutionMap = costToSolutionMap;

      for (const auto &pair : costToAccuracyMap) {
        InstructionCost currCompCost = pair.first;
        double currAccCost = pair.second;

        for (const auto &candidate : enumerate(ACC.candidates)) {
          size_t i = candidate.index();
          auto candCompCost =
              ACC.getAdjustedCompCostDelta(i, costToSolutionMap[currCompCost]);
          auto candAccCost =
              ACC.getAdjustedAccCostDelta(i, costToSolutionMap[currCompCost],
                                          valueToNodeMap, symbolToValueMap);

          // Don't ever try to apply a strictly useless candidate
          if (candCompCost >= 0 && candAccCost >= 0.) {
            continue;
          }

          InstructionCost newCompCost = currCompCost + candCompCost;
          double newAccCost = currAccCost + candAccCost;

          // if (EnzymePrintFPOpt)
          //   llvm::errs() << "ACC candidate " << i << " ("
          //                << candidate.value().desc
          //                << ") has accuracy cost: " << candAccCost
          //                << " and computation cost: " << candCompCost <<
          //                "\n";

          if (newCostToAccuracyMap.find(newCompCost) ==
                  newCostToAccuracyMap.end() ||
              newCostToAccuracyMap[newCompCost] > newAccCost) {
            newCostToAccuracyMap[newCompCost] = newAccCost;
            newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
            newCostToSolutionMap[newCompCost].emplace_back(&ACC, i);
            // if (EnzymePrintFPOpt) {
            // llvm::errs() << "ACC candidate " << i << " ("
            //              << candidate.value().desc
            //              << ") added; has accuracy cost: " << candAccCost
            //              << " and computation cost: " << candCompCost <<
            //              "\n";
            // llvm::errs() << "Updating accuracy map (ACC candidate " << i
            //              << "): computation cost " << newCompCost
            //              << " -> accuracy cost " << newAccCost << "\n";
            // }
          }
        }
      }

      for (const auto &l : newCostToAccuracyMap) {
        InstructionCost currCompCost = l.first;
        double currAccCost = l.second;

        bool dominated = false;
        for (const auto &r : newCostToAccuracyMap) {
          InstructionCost otherCompCost = r.first;
          double otherAccCost = r.second;

          if (currCompCost - otherCompCost >
                  std::fabs(FPOptCostDominanceThreshold *
                            GET_INSTRUCTION_COST(otherCompCost)) &&
              currAccCost - otherAccCost >=
                  std::fabs(FPOptAccuracyDominanceThreshold * otherAccCost)) {
            // if (EnzymePrintFPOpt)
            //   llvm::errs() << "ACC candidate with computation cost: "
            //                << currCompCost
            //                << " and accuracy cost: " << currAccCost
            //                << " is dominated by candidate with computation
            //                cost:"
            //                << otherCompCost
            //                << " and accuracy cost: " << otherAccCost << "\n";
            dominated = true;
            break;
          }
        }

        if (!dominated) {
          prunedCostToAccuracyMap[currCompCost] = currAccCost;
          prunedCostToSolutionMap[currCompCost] =
              newCostToSolutionMap[currCompCost];
        }
      }

      costToAccuracyMap = prunedCostToAccuracyMap;
      costToSolutionMap = prunedCostToSolutionMap;
      prunedCostToAccuracyMap.clear();
      prunedCostToSolutionMap.clear();

      llvm::errs() << "##### Finished processing " << ++ACCCounter << " of "
                   << ACCs.size() << " ACCs #####\n";
      llvm::errs() << "Current DP table sizes: " << costToAccuracyMap.size()
                   << "\n";
    }

    json::Object jsonObj;

    json::Object costAccMap;
    for (const auto &pair : costToAccuracyMap) {
      costAccMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] =
          pair.second;
    }
    jsonObj["costToAccuracyMap"] = std::move(costAccMap);

    json::Object costSolMap;
    for (const auto &pair : costToSolutionMap) {
      json::Array stepsArray;
      for (const auto &step : pair.second) {
        json::Object stepObj;
        stepObj["candidateIndex"] = static_cast<int64_t>(step.candidateIndex);

        std::visit(
            [&](auto *item) {
              using T = std::decay_t<decltype(*item)>;
              if constexpr (std::is_same_v<T, ApplicableOutput>) {
                stepObj["itemType"] = "AO";
                size_t index = aoPtrToIndex[item];
                stepObj["itemIndex"] = static_cast<int64_t>(index);
              } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
                stepObj["itemType"] = "ACC";
                size_t index = accPtrToIndex[item];
                stepObj["itemIndex"] = static_cast<int64_t>(index);
              }
            },
            step.item);
        stepsArray.push_back(std::move(stepObj));
      }
      costSolMap[std::to_string(GET_INSTRUCTION_COST(pair.first))] =
          std::move(stepsArray);
    }
    jsonObj["costToSolutionMap"] = std::move(costSolMap);

    std::error_code EC;
    llvm::raw_fd_ostream cacheFile(cacheFilePath, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      llvm::errs() << "Error writing cache file: " << EC.message() << "\n";
    } else {
      cacheFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(jsonObj)))
                << "\n";
      cacheFile.close();
      llvm::errs() << "DP tables cached to file.\n";
    }
  }

  if (EnzymePrintFPOpt) {
    if (FPOptShowTable) {
      llvm::errs() << "\n*** DP Table ***\n";
      for (const auto &pair : costToAccuracyMap) {
        if (!FPOptShowTableCosts.empty()) {
          bool shouldPrint = false;
          for (auto selectedCost : FPOptShowTableCosts)
            if (pair.first == selectedCost) {
              shouldPrint = true;
              break;
            }
          if (!shouldPrint)
            continue;
        }

        llvm::errs() << "Computation cost: " << pair.first
                     << ", Accuracy cost: " << pair.second << "\n";
        llvm::errs() << "\tSolution steps: \n";
        for (const auto &step : costToSolutionMap[pair.first]) {
          std::visit(
              [&](auto *item) {
                using T = std::decay_t<decltype(*item)>;
                if constexpr (std::is_same_v<T, ApplicableOutput>) {
                  llvm::errs()
                      << "\t\t" << item->expr << " --(" << step.candidateIndex
                      << ")-> " << item->candidates[step.candidateIndex].expr
                      << "\n";
                } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
                  llvm::errs() << "\t\tACC: "
                               << item->candidates[step.candidateIndex].desc
                               << " (#" << step.candidateIndex << ")\n";
                  if (FPOptShowPTDetails) {
                    auto &candidate = item->candidates[step.candidateIndex];
                    for (const auto &change : candidate.changes) {
                      llvm::errs()
                          << "\t\t\tChanging from "
                          << getPrecisionChangeTypeString(change.oldType)
                          << " to "
                          << getPrecisionChangeTypeString(change.newType)
                          << ":\n";
                      for (auto *val : change.nodes) {
                        llvm::errs() << "\t\t\t\t" << *val->value << "\n";
                      }
                    }
                  }
                } else {
                  llvm_unreachable(
                      "accuracyDPSolver: Unexpected type of solution step");
                }
              },
              step.item);
        }
      }
      llvm::errs() << "*** End of DP Table ***\n\n";
    }
  }

  std::string budgetsFile = FPOptCachePath + "/budgets.txt";
  if (!llvm::sys::fs::exists(budgetsFile)) {
    std::string budgetsStr;
    for (const auto &pair : costToAccuracyMap) {
      budgetsStr += std::to_string(GET_INSTRUCTION_COST(pair.first)) + ",";
    }

    if (!budgetsStr.empty())
      budgetsStr.pop_back();

    std::error_code EC;
    llvm::raw_fd_ostream Out(budgetsFile, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      llvm::errs() << "Error opening " << budgetsFile << ": " << EC.message()
                   << "\n";
    } else {
      Out << budgetsStr;
    }
  }

  llvm::errs() << "Critical computation cost range: ["
               << costToAccuracyMap.begin()->first << ", "
               << costToAccuracyMap.rbegin()->first << "]\n";

  llvm::errs() << "DP table contains " << costToAccuracyMap.size()
               << " entries.\n";

  double totalCandidateCompositions = 1.0;
  for (const auto &AO : AOs) {
    // +1 for the "do nothing" possibility
    totalCandidateCompositions *= AO.candidates.size() + 1;
  }
  for (const auto &ACC : ACCs) {
    totalCandidateCompositions *= ACC.candidates.size() + 1;
  }
  llvm::errs() << "Total candidate compositions: " << totalCandidateCompositions
               << "\n";

  if (costToSolutionMap.find(0) != costToSolutionMap.end()) {
    if (costToSolutionMap[0].empty()) {
      llvm::errs() << "WARNING: No-op solution (utilized cost budget = 0) is "
                      "considered Pareto-optimal.\n";
    }
  }

  double minAccCost = std::numeric_limits<double>::infinity();
  InstructionCost bestCompCost = 0;
  for (const auto &pair : costToAccuracyMap) {
    InstructionCost compCost = pair.first;
    double accCost = pair.second;

    if (compCost <= FPOptComputationCostBudget && accCost < minAccCost) {
      minAccCost = accCost;
      bestCompCost = compCost;
    }
  }

  if (minAccCost == std::numeric_limits<double>::infinity()) {
    llvm::errs() << "No solution found within the computation cost budget!\n";
    return changed;
  }

  llvm::errs() << "Minimum accuracy cost within budget: " << minAccCost << "\n";
  llvm::errs() << "Computation cost budget used: " << bestCompCost << "\n";

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "FPOpt DP solver: expected a solution!");

  llvm::errs() << "\n!!! DP solver: Applying solution ... !!!\n";
  for (const auto &solution : costToSolutionMap[bestCompCost]) {
    std::visit(
        [&](auto *item) {
          using T = std::decay_t<decltype(*item)>;
          if constexpr (std::is_same_v<T, ApplicableOutput>) {
            llvm::errs() << "Applying solution for " << item->expr << " --("
                         << solution.candidateIndex << ")-> "
                         << item->candidates[solution.candidateIndex].expr
                         << "\n";
            item->apply(solution.candidateIndex, valueToNodeMap,
                        symbolToValueMap);
          } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
            llvm::errs() << "Applying solution for ACC: "
                         << item->candidates[solution.candidateIndex].desc
                         << " (#" << solution.candidateIndex << ")\n";
            item->apply(solution.candidateIndex);
          } else {
            llvm_unreachable(
                "accuracyDPSolver: Unexpected type of solution step");
          }
        },
        solution.item);
    changed = true;
  }
  llvm::errs() << "!!! DP Solver: Solution applied !!!\n\n";

  return changed;
}

// Run (our choice of) floating point optimizations on function `F`.
// Return whether or not we change the function.
bool fpOptimize(Function &F, const TargetTransformInfo &TTI) {
  const std::string functionName = F.getName().str();
  std::string demangledName = llvm::demangle(functionName);
  size_t pos = demangledName.find('(');
  if (pos != std::string::npos) {
    demangledName = demangledName.substr(0, pos);
  }

  std::regex targetFuncRegex(FPOptTargetFuncRegex);
  if (!std::regex_match(demangledName, targetFuncRegex)) {
    if (EnzymePrintFPOpt)
      llvm::errs() << "Skipping function: " << demangledName
                   << " (demangled) since it does not match the target regex\n";
    return false;
  }

  if (!FPOptLogPath.empty()) {
    if (!isLogged(FPOptLogPath, functionName)) {
      if (EnzymePrintFPOpt)
        llvm::errs()
            << "Skipping matched function: " << demangledName
            << " (demangled) since this function is not found in the log\n";
      return false;
    }
  }

  if (!FPOptCachePath.empty()) {
    if (auto EC = llvm::sys::fs::create_directories(FPOptCachePath, true))
      llvm::errs() << "Warning: Could not create cache directory: "
                   << EC.message() << "\n";
  }

  // F.print(llvm::errs());

  bool changed = false;

  int symbolCounter = 0;
  auto getNextSymbol = [&symbolCounter]() -> std::string {
    return "v" + std::to_string(symbolCounter++);
  };

  // Extract change:

  // E1) create map<Value, FPNode> for all instructions I, map[I] = FPLLValue(I)
  // E2) for all instructions, if Poseidonable(I), map[I] = FPNode(operation(I),
  // map[operands(I)])
  // E3) floodfill for all starting locations I to find all distinct graphs /
  // outputs.

  /*
  B1:
    x = sin(arg)

  B2:
    y = 1 - x * x


  -> result y = cos(arg)^2

B1:
  nothing

B2:
  costmp = cos(arg)
  y = costmp * costmp

  */

  std::unordered_map<Value *, std::shared_ptr<FPNode>> valueToNodeMap;
  std::unordered_map<std::string, Value *> symbolToValueMap;

  llvm::errs() << "FPOpt: Starting Floodfill for " << F.getName() << "\n";

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!Poseidonable(I)) {
        valueToNodeMap[&I] =
            std::make_shared<FPLLValue>(&I, "__nh", "__nh"); // Non-Poseidonable
        if (EnzymePrintFPOpt)
          llvm::errs()
              << "Registered FPLLValue for non-Poseidonable instruction: " << I
              << "\n";
        continue;
      }

      std::string dtype;
      if (I.getType()->isFloatTy()) {
        dtype = "f32";
      } else if (I.getType()->isDoubleTy()) {
        dtype = "f64";
      } else {
        llvm_unreachable("Unexpected floating point type for instruction");
      }
      auto node = std::make_shared<FPLLValue>(&I, getHerbieOperator(I), dtype);

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I).args() : I.operands();
      for (auto &operand : operands) {
        if (!valueToNodeMap.count(operand)) {
          if (auto Arg = dyn_cast<Argument>(operand)) {
            std::string dtype;
            if (Arg->getType()->isFloatTy()) {
              dtype = "f32";
            } else if (Arg->getType()->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable("Unexpected floating point type for argument");
            }
            valueToNodeMap[operand] =
                std::make_shared<FPLLValue>(Arg, "__arg", dtype);
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for argument: " << *Arg
                           << "\n";
          } else if (auto C = dyn_cast<ConstantFP>(operand)) {
            SmallString<10> value;
            C->getValueAPF().toString(value);
            std::string dtype;
            if (C->getType()->isFloatTy()) {
              dtype = "f32";
            } else if (C->getType()->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable("Unexpected floating point type for constant");
            }
            valueToNodeMap[operand] =
                std::make_shared<FPConst>(value.c_str(), dtype);
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for " << dtype
                           << " constant: " << value << "\n";
          } else if (auto CI = dyn_cast<ConstantInt>(operand)) {
            // e.g., powi intrinsic has a constant int as its exponent
            double exponent = static_cast<double>(CI->getSExtValue());
            std::string dtype = "f64";
            std::string doubleStr = std::to_string(exponent);
            valueToNodeMap[operand] =
                std::make_shared<FPConst>(doubleStr.c_str(), dtype);
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for " << dtype
                           << " constant (casted from integer): " << doubleStr
                           << "\n";
          } else if (auto GV = dyn_cast<GlobalVariable>(operand)) {
            Type *elemType = GV->getValueType();

            assert(elemType->isFloatingPointTy() &&
                   "Global variable is not floating point type");
            std::string dtype;
            if (elemType->isFloatTy()) {
              dtype = "f32";
            } else if (elemType->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable(
                  "Unexpected floating point type for global variable");
            }
            valueToNodeMap[operand] =
                std::make_shared<FPLLValue>(GV, "__gv", dtype);
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for global variable: " << *GV
                           << "\n";
          } else {
            assert(0 && "Unknown operand");
          }
        }
        node->addOperand(valueToNodeMap[operand]);
      }
      valueToNodeMap[&I] = node;
    }
  }

  SmallSet<Value *, 8> component_seen;
  SmallVector<FPCC, 1> connected_components;
  for (auto &BB : F) {
    for (auto &I : BB) {
      // Not a Poseidonable instruction, doesn't make sense to create graph node
      // out of.
      if (!Poseidonable(I)) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Skipping non-Poseidonable instruction: " << I
                       << "\n";
        continue;
      }

      // Instruction is already in a set
      if (component_seen.contains(&I)) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Skipping already seen instruction: " << I << "\n";
        continue;
      }

      // if (!FPOptLogPath.empty()) {
      //   auto node = valueToNodeMap[&I];
      //   ValueInfo valueInfo;
      //   auto blockIt = std::find_if(
      //       I.getFunction()->begin(), I.getFunction()->end(),
      //       [&](const auto &block) { return &block == I.getParent(); });
      //   assert(blockIt != I.getFunction()->end() && "Block not found");
      //   size_t blockIdx = std::distance(I.getFunction()->begin(), blockIt);
      //   auto instIt =
      //       std::find_if(I.getParent()->begin(), I.getParent()->end(),
      //                    [&](const auto &curr) { return &curr == &I; });
      //   assert(instIt != I.getParent()->end() && "Instruction not found");
      //   size_t instIdx = std::distance(I.getParent()->begin(), instIt);

      //   bool found = extractValueFromLog(FPOptLogPath, functionName,
      //   blockIdx,
      //                                    instIdx, valueInfo);
      //   if (!found) {
      //     llvm::errs() << "Instruction " << I << " has no execution
      //     logged!\n"; continue;
      //   }
      // }

      if (EnzymePrintFPOpt)
        llvm::errs() << "Starting floodfill from: " << I << "\n";

      SmallVector<Value *, 8> todo;
      SetVector<Value *> input_seen;
      SetVector<Instruction *> output_seen;
      SetVector<Instruction *> operation_seen;
      todo.push_back(&I);
      while (!todo.empty()) {
        auto cur = todo.pop_back_val();
        assert(valueToNodeMap.count(cur) && "Node not found in valueToNodeMap");

        // We now can assume that this is a Poseidonable expression
        // Since we can only herbify instructions, let's assert that
        assert(isa<Instruction>(cur));
        auto I2 = cast<Instruction>(cur);

        // Don't repeat any instructions we've already seen (to avoid loops
        // for phi nodes)
        if (operation_seen.contains(I2)) {
          if (EnzymePrintFPOpt)
            llvm::errs() << "Skipping already seen instruction: " << *I2
                         << "\n";
          continue;
        }

        // Assume that a Poseidonable expression can only be in one connected
        // component.
        assert(!component_seen.contains(cur));

        if (EnzymePrintFPOpt)
          llvm::errs() << "Insert to operation_seen and component_seen: " << *I2
                       << "\n";
        operation_seen.insert(I2);
        component_seen.insert(cur);

        auto operands =
            isa<CallInst>(I2) ? cast<CallInst>(I2)->args() : I2->operands();

        for (const auto &operand_ : enumerate(operands)) {
          auto &operand = operand_.value();
          auto i = operand_.index();
          if (!Poseidonable(*operand)) {
            if (EnzymePrintFPOpt)
              llvm::errs() << "Non-Poseidonable input found: " << *operand
                           << "\n";

            // Don't mark constants as input `llvm::Value`s
            if (!isa<ConstantFP>(operand))
              input_seen.insert(operand);

            // look up error log to get bounds of non-Poseidonable inputs
            if (!FPOptLogPath.empty()) {
              ValueInfo data;
              auto blockIt = std::find_if(
                  I2->getFunction()->begin(), I2->getFunction()->end(),
                  [&](const auto &block) { return &block == I2->getParent(); });
              assert(blockIt != I2->getFunction()->end() && "Block not found");
              size_t blockIdx =
                  std::distance(I2->getFunction()->begin(), blockIt);
              auto instIt =
                  std::find_if(I2->getParent()->begin(), I2->getParent()->end(),
                               [&](const auto &curr) { return &curr == I2; });
              assert(instIt != I2->getParent()->end() &&
                     "Instruction not found");
              size_t instIdx = std::distance(I2->getParent()->begin(), instIt);

              bool res = extractValueFromLog(FPOptLogPath, functionName,
                                             blockIdx, instIdx, data);
              if (!res) {
                if (FPOptLooseCoverage)
                  continue;
                llvm::errs() << "FP Instruction " << *I2
                             << " has no execution logged!\n";
                llvm_unreachable(
                    "Unexecuted instruction found; set -fpopt-loose-coverage "
                    "to suppress this error\n");
              }
              auto node = valueToNodeMap[operand];
              node->updateBounds(data.minOperands[i], data.maxOperands[i]);

              if (EnzymePrintFPOpt) {
                llvm::errs() << "Range of " << *operand << " is ["
                             << node->getLowerBound() << ", "
                             << node->getUpperBound() << "]\n";
              }
            }
          } else {
            if (EnzymePrintFPOpt)
              llvm::errs() << "Adding operand to todo list: " << *operand
                           << "\n";
            todo.push_back(operand);
          }
        }

        for (auto U : I2->users()) {
          if (auto I3 = dyn_cast<Instruction>(U)) {
            if (!Poseidonable(*I3)) {
              if (EnzymePrintFPOpt)
                llvm::errs() << "Output instruction found: " << *I2 << "\n";
              output_seen.insert(I2);
            } else {
              if (EnzymePrintFPOpt)
                llvm::errs() << "Adding user to todo list: " << *I3 << "\n";
              todo.push_back(I3);
            }
          }
        }
      }

      // Don't bother with graphs without any Poseidonable operations
      if (!operation_seen.empty()) {
        if (EnzymePrintFPOpt) {
          llvm::errs() << "Found a connected component with "
                       << operation_seen.size() << " operations and "
                       << input_seen.size() << " inputs and "
                       << output_seen.size() << " outputs\n";

          llvm::errs() << "Inputs:\n";

          for (auto &input : input_seen) {
            llvm::errs() << *input << "\n";
          }

          llvm::errs() << "Outputs:\n";
          for (auto &output : output_seen) {
            llvm::errs() << *output << "\n";
          }

          llvm::errs() << "Operations:\n";
          for (auto &operation : operation_seen) {
            llvm::errs() << *operation << "\n";
          }
        }

        // TODO: Further check
        if (operation_seen.size() == 1) {
          if (EnzymePrintFPOpt)
            llvm::errs() << "Skipping trivial connected component\n";
          continue;
        }

        FPCC origCC{input_seen, output_seen, operation_seen};
        SmallVector<FPCC, 1> newCCs;
        splitFPCC(origCC, newCCs);

        for (auto &CC : newCCs) {
          for (auto *input : CC.inputs) {
            valueToNodeMap[input]->markAsInput();
          }
        }

        if (!FPOptLogPath.empty()) {
          for (auto &CC : newCCs) {
            // Extract grad and value info for all instructions.
            for (auto &op : CC.operations) {
              GradInfo grad;
              auto blockIt = std::find_if(
                  op->getFunction()->begin(), op->getFunction()->end(),
                  [&](const auto &block) { return &block == op->getParent(); });
              assert(blockIt != op->getFunction()->end() && "Block not found");
              size_t blockIdx =
                  std::distance(op->getFunction()->begin(), blockIt);
              auto instIt =
                  std::find_if(op->getParent()->begin(), op->getParent()->end(),
                               [&](const auto &curr) { return &curr == op; });
              assert(instIt != op->getParent()->end() &&
                     "Instruction not found");
              size_t instIdx = std::distance(op->getParent()->begin(), instIt);
              bool found = extractGradFromLog(FPOptLogPath, functionName,
                                              blockIdx, instIdx, grad);

              auto node = valueToNodeMap[op];
              if (FPOptReductionProf == "geomean") {
                node->grad = grad.geoMean;
              } else if (FPOptReductionProf == "arithmean") {
                node->grad = grad.arithMean;
              } else if (FPOptReductionProf == "maxabs") {
                node->grad = grad.maxAbs;
              } else {
                llvm_unreachable("Unknown FPOpt reduction type");
              }

              if (found) {
                ValueInfo valueInfo;
                extractValueFromLog(FPOptLogPath, functionName, blockIdx,
                                    instIdx, valueInfo);
                node->executions = valueInfo.executions;
                node->geoMean = valueInfo.geoMean;
                node->arithMean = valueInfo.arithMean;
                node->maxAbs = valueInfo.maxAbs;
                node->updateBounds(valueInfo.minRes, valueInfo.maxRes);

                if (EnzymePrintFPOpt) {
                  llvm::errs()
                      << "Range of " << *op << " is [" << node->getLowerBound()
                      << ", " << node->getUpperBound() << "]\n";
                }

                if (EnzymePrintFPOpt)
                  llvm::errs() << "Grad of " << *op << " is: " << node->grad
                               << " (" << FPOptReductionProf << ")\n"
                               << "Execution count of " << *op
                               << " is: " << node->executions << "\n";
              } else { // Unknown bounds
                if (EnzymePrintFPOpt)
                  llvm::errs()
                      << "Grad of " << *op
                      << " are not found in the log; using 0 instead\n";
              }
            }
          }
        }

        connected_components.insert(connected_components.end(), newCCs.begin(),
                                    newCCs.end());
      }
    }
  }

  llvm::errs() << "FPOpt: Found " << connected_components.size()
               << " connected components in " << F.getName() << "\n";

  // 1) Identify subgraphs of the computation which can be entirely represented
  // in herbie-style arithmetic
  // 2) Make the herbie FP-style expression by
  // converting llvm instructions into herbie string (FPNode ....)
  if (connected_components.empty()) {
    if (EnzymePrintFPOpt)
      llvm::errs() << "No Poseidonable connected components found\n";
    return false;
  }

  SmallVector<ApplicableOutput, 4> AOs;
  SmallVector<ApplicableFPCC, 4> ACCs;

  int componentCounter = 0;

  for (auto &component : connected_components) {
    assert(component.inputs.size() > 0 && "No inputs found for component");
    if (FPOptEnableHerbie) {
      for (const auto &input : component.inputs) {
        auto node = valueToNodeMap[input];
        if (node->op == "__const") {
          // Constants don't need a symbol
          continue;
        }
        if (!node->hasSymbol()) {
          node->symbol = getNextSymbol();
        }
        symbolToValueMap[node->symbol] = input;
        if (EnzymePrintFPOpt)
          llvm::errs() << "assigning symbol: " << node->symbol << " to "
                       << *input << "\n";
      }

      std::vector<std::string> herbieInputs;
      std::vector<ApplicableOutput> newAOs;
      int outputCounter = 0;

      assert(component.outputs.size() > 0 && "No outputs found for component");
      for (auto &output : component.outputs) {
        // 3) run fancy opts
        double grad = valueToNodeMap[output]->grad;
        unsigned executions = valueToNodeMap[output]->executions;

        // TODO: For now just skip if grad is 0
        if (!FPOptLogPath.empty() && grad == 0.) {
          llvm::errs() << "Skipping algebraic rewriting for " << *output
                       << " since gradient is 0\n";
          continue;
        }

        std::string expr =
            valueToNodeMap[output]->toFullExpression(valueToNodeMap);
        SmallSet<std::string, 8> args;
        getUniqueArgs(expr, args);

        std::string properties = ":herbie-conversions ([binary64 binary32])";
        if (valueToNodeMap[output]->dtype == "f32") {
          properties += " :precision binary32";
        } else if (valueToNodeMap[output]->dtype == "f64") {
          properties += " :precision binary64";
        } else {
          llvm_unreachable("Unexpected dtype");
        }

        if (!FPOptLogPath.empty()) {
          std::string precondition =
              getPrecondition(args, valueToNodeMap, symbolToValueMap);
          properties += " :pre " + precondition;
        }

        ApplicableOutput AO(component, output, expr, grad, executions, TTI);
        properties += " :name \"" + std::to_string(outputCounter++) + "\"";

        std::string argStr;
        for (const auto &arg : args) {
          if (!argStr.empty())
            argStr += " ";
          argStr += arg;
        }

        std::string herbieInput =
            "(FPCore (" + argStr + ") " + properties + " " + expr + ")";
        if (EnzymePrintHerbie)
          llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

        if (herbieInput.length() > FPOptMaxExprLength) {
          llvm::errs() << "WARNING: Skipping Herbie optimization for "
                       << *output
                       << " since expression length exceeds limit of "
                       << FPOptMaxExprLength << "\n";
          continue;
        }

        herbieInputs.push_back(herbieInput);
        newAOs.push_back(AO);
      }

      if (!herbieInputs.empty()) {
        if (!improveViaHerbie(herbieInputs, newAOs, F.getParent(), TTI,
                              valueToNodeMap, symbolToValueMap,
                              componentCounter)) {
          if (EnzymePrintHerbie)
            llvm::errs() << "Failed to optimize expressions using Herbie!\n";
        }

        AOs.insert(AOs.end(), newAOs.begin(), newAOs.end());
      }
    }

    if (FPOptEnablePT) {
      // Sort `component.operations` by the gradient and construct
      // `PrecisionChange`s.
      ApplicableFPCC ACC(component, TTI);
      auto *o0 = component.outputs[0];
      ACC.executions = valueToNodeMap[o0]->executions;

      const SmallVector<PrecisionChangeType> precTypes{
          PrecisionChangeType::FP32,
          PrecisionChangeType::FP64,
      };

      const auto &PTFuncs = getPTFuncs();

      // Check if we have a cached DP table
      std::string cacheFilePath = FPOptCachePath + "/table.json";
      bool skipEvaluation = FPOptSolverType == "dp" &&
                            !FPOptCachePath.empty() &&
                            llvm::sys::fs::exists(cacheFilePath);

      SmallVector<FPLLValue *, 8> operations;
      for (auto *I : component.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        auto node = cast<FPLLValue>(valueToNodeMap[I].get());
        if (PTFuncs.count(node->op) != 0) {
          operations.push_back(node);
        }
      }

      // Sort operations by the gradient
      llvm::sort(operations, [](const auto &a, const auto &b) {
        if (FPOptReductionEval == "geomean") {
          return std::fabs(a->grad * a->geoMean) >
                 std::fabs(b->grad * b->geoMean);
        } else if (FPOptReductionEval == "arithmean") {
          return std::fabs(a->grad * a->arithMean) >
                 std::fabs(b->grad * b->arithMean);
        } else if (FPOptReductionEval == "maxabs") {
          return std::fabs(a->grad * a->maxAbs) >
                 std::fabs(b->grad * b->maxAbs);
        } else {
          llvm_unreachable("Unknown FPOpt reduction type");
        }
      });

      // Create PrecisionChanges for 0-10%, 0-20%, ..., up to 0-100%
      for (int percent = 10; percent <= 100; percent += 10) {
        size_t numToChange = operations.size() * percent / 100;

        SetVector<FPLLValue *> opsToChange(operations.begin(),
                                           operations.begin() + numToChange);

        if (EnzymePrintFPOpt && !opsToChange.empty()) {
          llvm::errs() << "Created PrecisionChange for " << percent
                       << "% of Funcs (" << numToChange << ")\n";
        }

        for (auto prec : precTypes) {
          std::string precStr = getPrecisionChangeTypeString(prec).str();
          std::string desc =
              "Funcs 0% -- " + std::to_string(percent) + "% -> " + precStr;

          PrecisionChange change(
              opsToChange,
              getPrecisionChangeType(component.outputs[0]->getType()), prec);

          SmallVector<PrecisionChange, 1> changes{std::move(change)};
          PTCandidate candidate{std::move(changes), desc};

          if (!skipEvaluation) {
            candidate.CompCost = getCompCost(component, TTI, candidate);
          }

          ACC.candidates.push_back(std::move(candidate));
        }
      }

      // Create candidates by considering all operations without filtering
      SmallVector<FPLLValue *, 8> allOperations;
      for (auto *I : component.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        auto node = cast<FPLLValue>(valueToNodeMap[I].get());
        allOperations.push_back(node);
      }

      // Sort all operations by their sensitivity estimation (gradient-value
      // product)
      llvm::sort(allOperations, [](const auto &a, const auto &b) {
        return std::fabs(a->grad * a->geoMean) <
               std::fabs(b->grad * b->geoMean);
      });

      // Create PrecisionChanges for 0-10%, 0-20%, ..., up to 0-100%
      for (int percent = 10; percent <= 100; percent += 10) {
        size_t numToChange = allOperations.size() * percent / 100;

        SetVector<FPLLValue *> opsToChange(allOperations.begin(),
                                           allOperations.begin() + numToChange);

        if (EnzymePrintFPOpt && !opsToChange.empty()) {
          llvm::errs() << "Created PrecisionChange for " << percent
                       << "% of all operations (" << numToChange << ")\n";
        }

        for (auto prec : precTypes) {
          std::string precStr = getPrecisionChangeTypeString(prec).str();
          std::string desc =
              "All 0% -- " + std::to_string(percent) + "% -> " + precStr;

          PrecisionChange change(
              opsToChange,
              getPrecisionChangeType(component.outputs[0]->getType()), prec);

          SmallVector<PrecisionChange, 1> changes{std::move(change)};
          PTCandidate candidate{std::move(changes), desc};

          if (!skipEvaluation) {
            candidate.CompCost = getCompCost(component, TTI, candidate);
          }

          ACC.candidates.push_back(std::move(candidate));
        }
      }

      if (!skipEvaluation) {
        setUnifiedAccuracyCost(ACC, valueToNodeMap, symbolToValueMap);
      }

      ACCs.push_back(std::move(ACC));
    }
    llvm::errs() << "##### Finished synthesizing candidates for "
                 << ++componentCounter << " of " << connected_components.size()
                 << " connected components! #####\n";
  }

  // Perform rewrites
  if (EnzymePrintFPOpt) {
    if (FPOptEnableHerbie) {
      for (auto &AO : AOs) {
        llvm::errs() << "\n################################\n";
        llvm::errs() << "Initial AccuracyCost: " << AO.initialAccCost << "\n";
        llvm::errs() << "Initial ComputationCost: " << AO.initialCompCost
                     << "\n";
        llvm::errs() << "Initial HerbieCost: " << AO.initialHerbieCost << "\n";
        llvm::errs() << "Initial HerbieAccuracy: " << AO.initialHerbieAccuracy
                     << "\n";
        llvm::errs() << "Initial Expression: " << AO.expr << "\n";
        llvm::errs() << "Grad: " << AO.grad << "\n\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "Δ AccCost\t\tΔ "
                        "CompCost\t\tHerbieCost\t\tAccuracy\t\tExpression\n";
        llvm::errs() << "--------------------------------\n";
        for (size_t i = 0; i < AO.candidates.size(); ++i) {
          auto &candidate = AO.candidates[i];
          llvm::errs() << AO.getAccCostDelta(i) << "\t\t"
                       << AO.getCompCostDelta(i) << "\t\t"
                       << candidate.herbieCost << "\t\t"
                       << candidate.herbieAccuracy << "\t\t" << candidate.expr
                       << "\n";
        }
        llvm::errs() << "################################\n\n";
      }
    }
    if (FPOptEnablePT) {
      for (auto &ACC : ACCs) {
        llvm::errs() << "\n################################\n";
        llvm::errs() << "Initial AccuracyCost: " << ACC.initialAccCost << "\n";
        llvm::errs() << "Initial ComputationCost: " << ACC.initialCompCost
                     << "\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "Δ AccCost\t\tΔ CompCost\t\tDescription\n"
                     << "---------------------------\n";
        for (size_t i = 0; i < ACC.candidates.size(); ++i) {
          auto &candidate = ACC.candidates[i];
          llvm::errs() << ACC.getAccCostDelta(i) << "\t\t"
                       << ACC.getCompCostDelta(i) << "\t\t" << candidate.desc
                       << "\n";
        }
        llvm::errs() << "################################\n\n";
      }
    }
  }

  if (!FPOptEnableSolver) {
    if (FPOptEnableHerbie) {
      for (auto &AO : AOs) {
        AO.apply(0, valueToNodeMap, symbolToValueMap);
        changed = true;
      }
    }
  } else {
    if (FPOptLogPath.empty()) {
      llvm::errs() << "FPOpt: Solver enabled but no log file is provided\n";
      return false;
    }
    if (FPOptSolverType == "greedy") {
      changed =
          accuracyGreedySolver(AOs, ACCs, valueToNodeMap, symbolToValueMap);
    } else if (FPOptSolverType == "dp") {
      changed = accuracyDPSolver(AOs, ACCs, valueToNodeMap, symbolToValueMap);
    } else {
      llvm::errs() << "FPOpt: Unknown solver type: " << FPOptSolverType << "\n";
      return false;
    }
  }

  llvm::errs() << "FPOpt: Finished optimizing " << F.getName() << "\n";

  // Cleanup
  if (changed) {
    for (auto &component : connected_components) {
      if (component.outputs_rewritten != component.outputs.size()) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Skip erasing a connect component: only rewrote "
                       << component.outputs_rewritten << " of "
                       << component.outputs.size() << " outputs\n";
        continue; // Intermediate operations cannot be erased safely
      }
      for (auto *I : component.operations) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Erasing: " << *I << "\n";
        if (!I->use_empty()) {
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        }
        I->eraseFromParent();
      }
    }

    llvm::errs() << "FPOpt: Finished cleaning up " << F.getName() << "\n";
  }

  if (EnzymePrintFPOpt) {
    llvm::errs() << "FPOpt: Finished Optimization\n";
    // F.print(llvm::errs());
  }

  return changed;
}

namespace {} // namespace

char FPOpt::ID = 0;

FPOpt::FPOpt() : FunctionPass(ID) {}

void FPOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetTransformInfoWrapperPass>();
  FunctionPass::getAnalysisUsage(AU);
}

bool FPOpt::runOnFunction(Function &F) {
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  return fpOptimize(F, TTI);
}

static RegisterPass<FPOpt>
    X("fp-opt", "Run Enzyme/Poseidon Floating point optimizations");

FunctionPass *createFPOptPass() { return new FPOpt(); }

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddFPOptPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFPOptPass());
}

FPOptNewPM::Result FPOptNewPM::run(llvm::Module &M,
                                   llvm::ModuleAnalysisManager &MAM) {
  bool changed = false;
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      const auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
      changed |= fpOptimize(F, TTI);
    }
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey FPOptNewPM::Key;
