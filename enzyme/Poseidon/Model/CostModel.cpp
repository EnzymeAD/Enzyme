//=- CostModel.cpp - the measured cost model and cost walks ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CostModel.h"
#include "Targets/DeviceMath.h"
#include "Targets/MathFunctions.h"
#include "Targets/Target.h"
#include "Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <fstream>

using namespace llvm;

namespace poseidon {

static bool isScalarTypeName(StringRef T) {
  return is_contained({"bf16", "half", "float", "double", "fp80", "fp128"}, T);
}

// The row families poseidon-calibrate writes (tools/calibrate/README.md),
// read by a compile or not: the operations at a scalar type, the conversions
// by their source type, the tensor-core tiles by shape and by input and
// accumulator type, and one dimensionless family per matrix-product
// realization. A new family is added here with the calibrator arm that writes
// it.
static bool isCostModelRow(StringRef Opcode, StringRef Type) {
  if (is_contained({"ozaki_dispatch_rel", "ozaki_dispatch_rect_rel",
                    "tcec_dispatch_rel", "direct_dispatch_rel",
                    "wmma_inkernel_rel"},
                   Opcode))
    return !Type.empty();
  for (StringRef Tile : {"wmma_mma_", "wmma_load_a_", "wmma_load_b_",
                         "wmma_load_c_", "wmma_store_d_"})
    if (Opcode.consume_front(Tile)) {
      unsigned M, N, K;
      return Opcode.consume_front("m") && !Opcode.consumeInteger(10, M) &&
             Opcode.consume_front("n") && !Opcode.consumeInteger(10, N) &&
             Opcode.consume_front("k") && !Opcode.consumeInteger(10, K) &&
             Opcode.empty() && !Type.empty();
    }
  for (StringRef Cast : {"fpext_", "fptrunc_"})
    if (Opcode.consume_front(Cast)) {
      auto [From, To] = Opcode.split("_to_");
      return isScalarTypeName(From) && isScalarTypeName(To) && Type == From;
    }
  if (!isScalarTypeName(Type))
    return false;
  // rsqrt is measured on GPUs and priced by no compile yet.
  return is_contained({"fneg", "fadd", "fsub", "fmul", "fdiv", "fcmp",
                       "fmuladd", "maxnum", "minnum", "rsqrt"},
                      Opcode) ||
         any_of(mathFunctions(),
                [&](const MathFunction &Fn) { return Fn.Name == Opcode; });
}

void requireCostModel(CostModelState &CM, const Function &F) {
  StringRef targetCpu = Target::of(F).processor(F);
  if (Error E = requireCostModel(CM, targetCpu, F.getName(), isCostModelRow))
    reportUserError(F, F.getSubprogram(), toString(std::move(E)));
}

// A price the compile needs and the model does not have.
[[noreturn]] static void missingRow(CostModelState &CM, const Function &F,
                                    const DiagnosticLocation &Loc,
                                    StringRef Opcode, StringRef Type) {
  reportUserError(F, Loc,
                  "cost model " + Twine(costModelPath(CM)) + " has no row '" +
                      Opcode + "," + Type +
                      "'; measure it with poseidon-calibrate");
}

static double rowCost(CostModelState &CM, const Instruction &I,
                      StringRef Opcode, StringRef Type) {
  if (std::optional<double> cost = lookupCost(CM, Opcode, Type))
    return *cost;
  missingRow(CM, *I.getFunction(), I.getDebugLoc(), Opcode, Type);
}

const std::unordered_set<std::string> &getScalarTypes(CostModelState &CM,
                                                      const Function &F) {
  if (CM.ScalarTypes.empty())
    reportUserError(F, F.getSubprogram(),
                    "cost model " + Twine(costModelPath(CM)) +
                        " carries no '# scalar_types=' header, so the formats "
                        "the device has hardware for are unknown");
  return CM.ScalarTypes;
}

// An instruction no row family prices.
[[noreturn]] static void unpriced(const Instruction *I) {
  std::string Text;
  raw_string_ostream(Text) << *I;
  report_fatal_error("Poseidon: no cost-model row prices the instruction" +
                     Twine(Text));
}

double getInstructionCompCost(CostModelState &CM, const Instruction *I) {
#if LLVM_VERSION_MAJOR >= 20
  // O3 fuses sin(x) and cos(x) into llvm.sincos, whose struct result would
  // otherwise be priced as free.
  if (auto *II = dyn_cast<IntrinsicInst>(I))
    if (II->getIntrinsicID() == Intrinsic::sincos) {
      Type *ETy = cast<StructType>(I->getType())->getElementType(0);
      std::string P = ETy->isFloatTy()    ? "float"
                      : ETy->isDoubleTy() ? "double"
                      : ETy->isHalfTy()   ? "half"
                                          : "";
      if (P.empty())
        unpriced(I);
      return rowCost(CM, *I, "sin", P) + rowCost(CM, *I, "cos", P);
    }
#endif
  // A regime split's condition is an fcmp, whose i1 result is not an FP type
  // but runs on the FP pipe of its operands.
  if (!I->getType()->isFPOrFPVectorTy() && !isa<FCmpInst>(I))
    return 0.0;

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
  case Instruction::Select:
  case Instruction::Load:
  case Instruction::ExtractValue:
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
        case Intrinsic::asin:
          OpcodeName = "asin";
          break;
        case Intrinsic::acos:
          OpcodeName = "acos";
          break;
        case Intrinsic::atan:
          OpcodeName = "atan";
          break;
        case Intrinsic::atan2:
          OpcodeName = "atan2";
          break;
        case Intrinsic::sinh:
          OpcodeName = "sinh";
          break;
        case Intrinsic::cosh:
          OpcodeName = "cosh";
          break;
        case Intrinsic::tanh:
          OpcodeName = "tanh";
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
          // The square root precision tuning lowers an F32 one to.
          if (isApproxSqrtF32(CalledFunc->getIntrinsicID())) {
            OpcodeName = "sqrt";
            break;
          }
          unpriced(I);
        }
        }
      } else if (StringRef deviceMath = deviceMathName(CalledFunc->getName());
                 !deviceMath.empty()) {
        OpcodeName = deviceMath.str();
        // PT materializes fmax/fmin as llvm.maxnum/llvm.minnum
        // (lib/Precision.cpp), so the priced op and the emitted op are the
        // same instruction and the same measured row serves both.
        if (OpcodeName == "fmax")
          OpcodeName = "maxnum";
        else if (OpcodeName == "fmin")
          OpcodeName = "minnum";
      } else {
        std::string FuncName = CalledFunc->getName().str();
        if (!FuncName.empty() &&
            (FuncName.back() == 'f' || FuncName.back() == 'l'))
          FuncName.pop_back();

        if (!libmFuncs().count(FuncName))
          unpriced(I);
        OpcodeName = FuncName;
      }
    } else {
      unpriced(I);
    }
    break;
  }
  default:
    unpriced(I);
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
  else
    unpriced(I);

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
    else
      unpriced(I);

    OpcodeName += "_" + SrcPrecisionName + "_to_" + PrecisionName;
    PrecisionName = SrcPrecisionName;
  }

  return rowCost(CM, *I, OpcodeName, PrecisionName);
}

double profiledCost(CostModelState &CM, const Function &F,
                    const ProfileMap &profile,
                    function_ref<bool(const Instruction &, size_t &)> slotOf) {
  double cost = 0.0;
  for (const Instruction &I : instructions(F)) {
    size_t slot;
    if (!slotOf(I, slot))
      continue;
    auto it = profile.find(slot);
    if (it != profile.end())
      cost += (double)it->second.exec * getInstructionCompCost(CM, &I);
  }
  return cost;
}

const std::unordered_set<std::string> &getPTFuncs(CostModelState &CM,
                                                  const Function &F) {
  if (!CM.PTFuncs) {
    auto row = [&](StringRef Opcode, StringRef Type) {
      if (std::optional<double> cost = lookupCost(CM, Opcode, Type))
        return *cost;
      missingRow(CM, F, F.getSubprogram(), Opcode, Type);
    };
    std::unordered_set<std::string> funcs;
    for (const auto &func : libmFuncs()) {
      double costFP32 = row(func, "float");
      double costFP64 = row(func, "double");
      double costFPTrunc = row("fptrunc_double_to_float", "double");
      double costFPExt = row("fpext_float_to_double", "float");
      double costFPCast = costFPTrunc + costFPExt;
      if (costFP32 + costFPCast < costFP64)
        funcs.insert(func);
    }
    CM.PTFuncs = std::move(funcs);
  }
  return *CM.PTFuncs;
}

double computeMaxCost(CostModelState &CM, BasicBlock *BB,
                      std::unordered_map<BasicBlock *, double> &MaxCost,
                      std::unordered_set<BasicBlock *> &Visited) {
  if (MaxCost.find(BB) != MaxCost.end())
    return MaxCost[BB];

  if (!Visited.insert(BB).second)
    return 0;

  double BBCost = 0;
  for (const Instruction &I : *BB) {
    if (I.isTerminator())
      continue;

    auto instCost = getInstructionCompCost(CM, &I);

    BBCost += instCost;
  }

  double succCost = 0;

  if (!succ_empty(BB)) {
    double maxSuccCost = 0;
    for (BasicBlock *Succ : successors(BB)) {
      double succBBCost = computeMaxCost(CM, Succ, MaxCost, Visited);
      if (succBBCost > maxSuccCost)
        maxSuccCost = succBBCost;
    }
    succCost = maxSuccCost;
  }

  double totalCost = BBCost + succCost;
  MaxCost[BB] = totalCost;
  Visited.erase(BB);
  return totalCost;
}

double getCompCost(CostModelState &CM, Function *F) {
  std::unordered_map<BasicBlock *, double> MaxCost;
  std::unordered_set<BasicBlock *> Visited;

  BasicBlock *EntryBB = &F->getEntryBlock();
  double TotalCost = computeMaxCost(CM, EntryBB, MaxCost, Visited);
  return TotalCost;
}

double getCompCost(CostModelState &CM, const SmallVector<Value *> &outputs,
                   const SetVector<Value *> &inputs,
                   const DenseMap<const Instruction *, uint64_t> *opExec,
                   uint64_t normalizer) {
  assert(!outputs.empty());
  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;
  double cost = 0;

  todo.insert(todo.end(), outputs.begin(), outputs.end());
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (inputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      // TODO: unfair to ignore branches when calculating cost
      auto instCost = getInstructionCompCost(CM, I);
      if (opExec && normalizer) {
        auto it = opExec->find(I);
        if (it != opExec->end())
          instCost *= (double)it->second / (double)normalizer;
      }
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

} // namespace poseidon
