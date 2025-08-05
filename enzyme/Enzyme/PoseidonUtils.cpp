//=- PoseidonUtils.cpp - Utility functions for Poseidon optimization pass --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for the Poseidon floating-point
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Poseidon.h"
#include "PoseidonTypes.h"
#include "PoseidonUtils.h"

using namespace llvm;

cl::opt<std::string>
    FPOptCostModelPath("fpopt-cost-model-path", cl::init(""), cl::Hidden,
                       cl::desc("Use a custom cost model in the FPOpt pass"));
cl::opt<unsigned>
    FPOptNumSamples("fpopt-num-samples", cl::init(1024), cl::Hidden,
                    cl::desc("Number of sampled points for input hypercube"));
cl::opt<unsigned>
    FPOptRandomSeed("fpopt-random-seed", cl::init(239778888), cl::Hidden,
                    cl::desc("The random seed used in the FPOpt pass"));

static const std::unordered_set<std::string> LibmFuncs = {
    "sin",   "cos",   "tan",      "asin",  "acos",   "atan",  "atan2",
    "sinh",  "cosh",  "tanh",     "asinh", "acosh",  "atanh", "exp",
    "log",   "sqrt",  "cbrt",     "pow",   "fabs",   "fma",   "hypot",
    "expm1", "log1p", "ceil",     "floor", "erf",    "exp2",  "lgamma",
    "log10", "log2",  "rint",     "round", "tgamma", "trunc", "copysign",
    "fdim",  "fmod",  "remainder"};

double getOneULP(double value) {
  assert(!std::isnan(value) && !std::isinf(value));

  double next = std::nextafter(value, std::numeric_limits<double>::infinity());
  double ulp = std::fabs(next - value);

  return ulp;
}

std::string getLibmFunctionForPrecision(StringRef funcName, Type *newType) {
  static const std::unordered_set<std::string> libmFunctions = {
      "sin",   "cos",   "tan",      "asin",  "acos",   "atan",  "atan2",
      "sinh",  "cosh",  "tanh",     "asinh", "acosh",  "atanh", "sqrt",
      "cbrt",  "pow",   "exp",      "log",   "fabs",   "fma",   "hypot",
      "expm1", "log1p", "ceil",     "floor", "erf",    "exp2",  "lgamma",
      "log10", "log2",  "rint",     "round", "tgamma", "trunc", "copysign",
      "fdim",  "fmod",  "remainder"};

  std::string baseName = funcName.str();
  if (baseName.back() == 'f' || baseName.back() == 'l') {
    baseName.pop_back();
  }

  if (libmFunctions.count(baseName)) {
    if (newType->isFloatTy()) {
      return baseName + "f";
    } else if (newType->isDoubleTy()) {
      return baseName;
    } else if (newType->isFP128Ty() || newType->isX86_FP80Ty()) {
      return baseName + "l";
    }
  }

  return "";
}

double stringToDouble(const std::string &str) {
  char *end;
  errno = 0;
  double result = std::strtod(str.c_str(), &end);

  if (errno == ERANGE) {
    if (result == HUGE_VAL) {
      result = std::numeric_limits<double>::infinity();
    } else if (result == -HUGE_VAL) {
      result = -std::numeric_limits<double>::infinity();
    }
  }

  return result; // Denormalized values are fine
}

void topoSort(const SetVector<Instruction *> &insts,
              SmallVectorImpl<Instruction *> &instsSorted) {
  SmallPtrSet<Instruction *, 8> visited;
  SmallPtrSet<Instruction *, 8> onStack;

  std::function<void(Instruction *)> dfsVisit = [&](Instruction *I) {
    if (visited.count(I))
      return;
    visited.insert(I);
    onStack.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
    for (auto &op : operands) {
      if (isa<Instruction>(op)) {
        Instruction *oI = cast<Instruction>(op);
        if (insts.contains(oI)) {
          if (onStack.count(oI)) {
            llvm_unreachable(
                "topoSort: Cycle detected in instruction dependencies!");
          }
          dfsVisit(oI);
        }
      }
    }

    onStack.erase(I);
    instsSorted.push_back(I);
  };

  for (auto *I : insts) {
    if (!visited.count(I)) {
      dfsVisit(I);
    }
  }

  std::reverse(instsSorted.begin(), instsSorted.end());
}

void getUniqueArgs(const std::string &expr, SmallSet<std::string, 8> &args) {
  std::regex argPattern("v\\d+");

  std::sregex_iterator begin(expr.begin(), expr.end(), argPattern);
  std::sregex_iterator end;

  while (begin != end) {
    args.insert(begin->str());
    ++begin;
  }
}

const std::map<std::pair<std::string, std::string>, InstructionCost> &
getCostModel() {
  static std::map<std::pair<std::string, std::string>, InstructionCost>
      CostModel;
  static bool Loaded = false;
  if (!Loaded) {
    std::ifstream CostFile(FPOptCostModelPath);
    if (!CostFile.is_open()) {
      std::string msg =
          "Cost model file could not be opened: " + FPOptCostModelPath;
      llvm_unreachable(msg.c_str());
    }
    std::string Line;
    while (std::getline(CostFile, Line)) {
      std::istringstream SS(Line);
      std::string OpcodeStr, PrecisionStr, CostStr;
      if (!std::getline(SS, OpcodeStr, ',')) {
        llvm_unreachable(
            ("Unexpected line in custom cost model: " + Line).c_str());
      }
      if (!std::getline(SS, PrecisionStr, ',')) {
        llvm_unreachable(
            ("Unexpected line in custom cost model: " + Line).c_str());
      }
      if (!std::getline(SS, CostStr)) {
        llvm_unreachable(
            ("Unexpected line in custom cost model: " + Line).c_str());
      }
      CostModel[{OpcodeStr, PrecisionStr}] = std::stoi(CostStr);
    }
    Loaded = true;
  }
  return CostModel;
}

InstructionCost queryCostModel(const std::string &OpcodeName,
                               const std::string &PrecisionName) {
  const auto &CostModel = getCostModel();
  auto Key = std::make_pair(OpcodeName, PrecisionName);
  auto It = CostModel.find(Key);
  if (It != CostModel.end())
    return It->second;

  std::string msg = "Custom cost model: entry not found for " + OpcodeName +
                    " @ " + PrecisionName;
  llvm::errs() << msg << "\n";
  llvm_unreachable(msg.c_str());
}

InstructionCost getInstructionCompCost(const Instruction *I,
                                       const TargetTransformInfo &TTI) {
  if (!I->getType()->isFPOrFPVectorTy())
    return 0;

  if (!FPOptCostModelPath.empty()) {
    std::string OpcodeName;
    switch (I->getOpcode()) {
    case Instruction::FNeg:
      OpcodeName = "fneg";
      break;
    case Instruction::FAdd:
      OpcodeName = "fadd";
      break;
    case Instruction::FSub:
      OpcodeName = "fsub";
      break;
    case Instruction::FMul:
      OpcodeName = "fmul";
      break;
    case Instruction::FDiv:
      OpcodeName = "fdiv";
      break;
    case Instruction::FCmp:
      OpcodeName = "fcmp";
      break;
    case Instruction::FPExt:
      OpcodeName = "fpext";
      break;
    case Instruction::FPTrunc:
      OpcodeName = "fptrunc";
      break;
    case Instruction::PHI:
      return 0;
    case Instruction::Call: {
      auto *Call = cast<CallInst>(I);
      if (auto *CalledFunc = Call->getCalledFunction()) {
        if (CalledFunc->isIntrinsic()) {
          switch (CalledFunc->getIntrinsicID()) {
          case Intrinsic::sin:
            OpcodeName = "sin";
            break;
          case Intrinsic::cos:
            OpcodeName = "cos";
            break;
#if LLVM_VERSION_MAJOR > 16
          case Intrinsic::tan:
            OpcodeName = "tan";
            break;
#endif
          case Intrinsic::exp:
            OpcodeName = "exp";
            break;
          case Intrinsic::log:
            OpcodeName = "log";
            break;
          case Intrinsic::sqrt:
            OpcodeName = "sqrt";
            break;
          case Intrinsic::fabs:
            OpcodeName = "fabs";
            break;
          case Intrinsic::fma:
            OpcodeName = "fma";
            break;
          case Intrinsic::pow:
            OpcodeName = "pow";
            break;
          case Intrinsic::powi:
            OpcodeName = "powi";
            break;
          case Intrinsic::fmuladd:
            OpcodeName = "fmuladd";
            break;
          case Intrinsic::maxnum:
            OpcodeName = "maxnum";
            break;
          case Intrinsic::minnum:
            OpcodeName = "minnum";
            break;
          case Intrinsic::ceil:
            OpcodeName = "ceil";
            break;
          case Intrinsic::floor:
            OpcodeName = "floor";
            break;
          case Intrinsic::exp2:
            OpcodeName = "exp2";
            break;
          case Intrinsic::log10:
            OpcodeName = "log10";
            break;
          case Intrinsic::log2:
            OpcodeName = "log2";
            break;
          case Intrinsic::rint:
            OpcodeName = "rint";
            break;
          case Intrinsic::round:
            OpcodeName = "round";
            break;
          case Intrinsic::trunc:
            OpcodeName = "trunc";
            break;
          case Intrinsic::copysign:
            OpcodeName = "copysign";
            break;
          default: {
            std::string msg = "Custom cost model: unsupported intrinsic " +
                              CalledFunc->getName().str();
            llvm_unreachable(msg.c_str());
          }
          }
        } else {
          std::string FuncName = CalledFunc->getName().str();
          if (!FuncName.empty() &&
              (FuncName.back() == 'f' || FuncName.back() == 'l'))
            FuncName.pop_back();

          if (LibmFuncs.count(FuncName))
            OpcodeName = FuncName;
          else {
            std::string msg =
                "Custom cost model: unknown function call " + FuncName;
            llvm_unreachable(msg.c_str());
          }
        }
      } else {
        llvm_unreachable("Custom cost model: unknown function call");
      }
      break;
    }
    default: {
      std::string msg = "Custom cost model: unexpected opcode " +
                        std::string(I->getOpcodeName());
      llvm_unreachable(msg.c_str());
    }
    }

    std::string PrecisionName;
    Type *Ty = I->getType();
    if (I->getOpcode() == Instruction::FCmp)
      Ty = I->getOperand(0)->getType();

    if (Ty->isBFloatTy())
      PrecisionName = "bf16";
    else if (Ty->isHalfTy())
      PrecisionName = "half";
    else if (Ty->isFloatTy())
      PrecisionName = "float";
    else if (Ty->isDoubleTy())
      PrecisionName = "double";
    else if (Ty->isX86_FP80Ty())
      PrecisionName = "fp80";
    else if (Ty->isFP128Ty())
      PrecisionName = "fp128";
    else {
      std::string msg = "Custom cost model: unsupported precision type!";
      llvm_unreachable(msg.c_str());
    }

    // For FPExt/FPTrunc, update the opcode name to include conversion info.
    if (I->getOpcode() == Instruction::FPExt ||
        I->getOpcode() == Instruction::FPTrunc) {
      Type *SrcTy = I->getOperand(0)->getType();
      std::string SrcPrecisionName;
      if (SrcTy->isBFloatTy())
        SrcPrecisionName = "bf16";
      else if (SrcTy->isHalfTy())
        SrcPrecisionName = "half";
      else if (SrcTy->isFloatTy())
        SrcPrecisionName = "float";
      else if (SrcTy->isDoubleTy())
        SrcPrecisionName = "double";
      else if (SrcTy->isX86_FP80Ty())
        SrcPrecisionName = "fp80";
      else if (SrcTy->isFP128Ty())
        SrcPrecisionName = "fp128";
      else {
        std::string msg = "Custom cost model: unsupported precision type!";
        llvm_unreachable(msg.c_str());
      }

      OpcodeName += "_" + SrcPrecisionName + "_to_" + PrecisionName;
      PrecisionName = SrcPrecisionName;
    }

    return queryCostModel(OpcodeName, PrecisionName);
  } else {
    llvm::errs() << "WARNING: Custom cost model not found, using TTI cost!\n";
    return TTI.getInstructionCost(I, TargetTransformInfo::TCK_RecipThroughput);
  }
}

const std::unordered_set<std::string> &getPTFuncs() {
  static const std::unordered_set<std::string> PTFuncs = []() {
    if (FPOptCostModelPath.empty())
      return std::unordered_set<std::string>{};
    std::unordered_set<std::string> funcs;
    for (const auto &func : LibmFuncs) {
      InstructionCost costFP32 = queryCostModel(func, "float");
      InstructionCost costFP64 = queryCostModel(func, "double");
      if (costFP32 < costFP64)
        funcs.insert(func);
    }
    return funcs;
  }();
  return PTFuncs;
}

InstructionCost computeMaxCost(
    BasicBlock *BB, std::unordered_map<BasicBlock *, InstructionCost> &MaxCost,
    std::unordered_set<BasicBlock *> &Visited, const TargetTransformInfo &TTI) {
  if (MaxCost.find(BB) != MaxCost.end())
    return MaxCost[BB];

  if (!Visited.insert(BB).second)
    return 0;

  InstructionCost BBCost = 0;
  for (const Instruction &I : *BB) {
    if (I.isTerminator())
      continue;

    auto instCost = getInstructionCompCost(&I, TTI);

    // if (EnzymePrintFPOpt)
    // llvm::errs() << "Cost of " << I << " is: " << instCost << "\n";

    BBCost += instCost;
  }

  InstructionCost succCost = 0;

  if (!succ_empty(BB)) {
    InstructionCost maxSuccCost = 0;
    for (BasicBlock *Succ : successors(BB)) {
      InstructionCost succBBCost = computeMaxCost(Succ, MaxCost, Visited, TTI);
      if (succBBCost > maxSuccCost)
        maxSuccCost = succBBCost;
    }
    // llvm::errs() << "Max succ cost: " << maxSuccCost << "\n";
    succCost = maxSuccCost;
  }

  InstructionCost totalCost = BBCost + succCost;
  // llvm::errs() << "BB " << BB->getName() << " cost: " << totalCost << "\n";
  MaxCost[BB] = totalCost;
  Visited.erase(BB);
  return totalCost;
}

void getSampledPoints(
    ArrayRef<Value *> inputs,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<MapVector<Value *, double>, 4> &sampledPoints) {
  std::default_random_engine gen;
  gen.seed(FPOptRandomSeed);
  std::uniform_real_distribution<> dis;

  MapVector<Value *, SmallVector<double, 2>> hypercube;
  for (const auto input : inputs) {
    const auto node = valueToNodeMap.at(input);

    double lower = node->getLowerBound();
    double upper = node->getUpperBound();

    hypercube.insert({input, {lower, upper}});
  }

  // llvm::errs() << "Hypercube:\n";
  // for (const auto &entry : hypercube) {
  //   Value *val = entry.first;
  //   double lower = entry.second[0];
  //   double upper = entry.second[1];
  //   llvm::errs() << valueToNodeMap.at(val)->symbol << ": [" << lower << ", "
  //                << upper << "]\n";
  // }

  // Sample `FPOptNumSamples` points from the hypercube. Store it in
  // `sampledPoints`.
  sampledPoints.clear();
  sampledPoints.resize(FPOptNumSamples);
  for (int i = 0; i < FPOptNumSamples; ++i) {
    MapVector<Value *, double> point;
    for (const auto &entry : hypercube) {
      Value *val = entry.first;
      double lower = entry.second[0];
      double upper = entry.second[1];
      double sample = dis(gen, decltype(dis)::param_type{lower, upper});
      point.insert({val, sample});
    }
    sampledPoints[i] = point;
    // llvm::errs() << "Sample " << i << ":\n";
    // for (const auto &entry : point) {
    //   llvm::errs() << valueToNodeMap.at(entry.first)->symbol << ": "
    //                << entry.second << "\n";
    // }
  }
}

void getSampledPoints(
    const std::string &expr,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<MapVector<Value *, double>, 4> &sampledPoints) {
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SmallVector<Value *, 4> inputs;
  for (const auto &argStr : argStrSet) {
    inputs.push_back(symbolToValueMap.at(argStr));
  }

  getSampledPoints(inputs, valueToNodeMap, symbolToValueMap, sampledPoints);
}

InstructionCost getCompCost(Function *F, const TargetTransformInfo &TTI) {
  std::unordered_map<BasicBlock *, InstructionCost> MaxCost;
  std::unordered_set<BasicBlock *> Visited;

  BasicBlock *EntryBB = &F->getEntryBlock();
  InstructionCost TotalCost = computeMaxCost(EntryBB, MaxCost, Visited, TTI);
  // llvm::errs() << "Total cost: " << TotalCost << "\n";
  return TotalCost;
}

// Sum up the cost of `output` and its FP operands recursively up to `inputs`
// (exclusive).
InstructionCost getCompCost(const SmallVector<Value *> &outputs,
                            const SetVector<Value *> &inputs,
                            const TargetTransformInfo &TTI) {
  assert(!outputs.empty());
  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;
  InstructionCost cost = 0;

  todo.insert(todo.end(), outputs.begin(), outputs.end());
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (inputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      // TODO: unfair to ignore branches when calculating cost
      auto instCost = getInstructionCompCost(I, TTI);

      // if (EnzymePrintFPOpt)
      //   llvm::errs() << "Cost of " << *I << " is: " << instCost << "\n";

      // Only add the cost of the instruction if it is not an input
      cost += instCost;

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
      for (auto &operand : operands) {
        todo.push_back(operand);
      }
    }
  }

  return cost;
}

void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
                      SmallPtrSetImpl<Instruction *> &exprInsts,
                      SmallPtrSetImpl<Value *> &visited) {
  if (!V || inputs.contains(V) || visited.contains(V)) {
    return;
  }

  visited.insert(V);

  if (auto *I = dyn_cast<Instruction>(V)) {
    exprInsts.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();

    for (auto &op : operands) {
      collectExprInsts(op, inputs, exprInsts, visited);
    }
  }
}

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