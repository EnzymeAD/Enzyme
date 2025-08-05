#include <llvm/Config/llvm-config.h>

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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Poseidon.h"
#include "PoseidonHerbieUtils.h"
#include "PoseidonLogger.h"
#include "PoseidonPrecUtils.h"
#include "PoseidonSolvers.h"
#include "PoseidonTypes.h"
#include "PoseidonUtils.h"
#include "../Utils.h"

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

cl::opt<std::string> FPOptTargetFuncRegex(
    "fpopt-target-func-regex", cl::init(".*"), cl::Hidden,
    cl::desc("Regex pattern to match target functions in the FPOpt pass"));
cl::opt<bool> FPOptEnableHerbie(
    "fpopt-enable-herbie", cl::init(true), cl::Hidden,
    cl::desc("Use Herbie to rewrite floating-point expressions"));
cl::opt<bool> FPOptEnablePT(
    "fpopt-enable-pt", cl::init(false), cl::Hidden,
    cl::desc("Consider precision changes of floating-point expressions"));
cl::opt<std::string> FPOptCachePath("fpopt-cache-path", cl::init("cache"),
                                    cl::Hidden,
                                    cl::desc("Path to cache Herbie results"));
cl::opt<bool> FPOptEnableSolver(
    "fpopt-enable-solver", cl::init(false), cl::Hidden,
    cl::desc("Use the solver to select desirable rewrite candidates; when "
             "disabled, apply all Herbie's first choices"));
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
cl::opt<std::string> FPOptReductionProf(
    "fpopt-reduction-prof", cl::init("geomean"), cl::Hidden,
    cl::desc("Which reduction result to extract from profiles. "
             "Options are 'geomean', 'arithmean', and 'maxabs'"));
cl::opt<std::string> FPOptReductionEval(
    "fpopt-reduction-eval", cl::init("geomean"), cl::Hidden,
    cl::desc("Which reduction result to use in candidate evaluation. "
             "Options are 'geomean', 'arithmean', and 'maxabs'"));

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
