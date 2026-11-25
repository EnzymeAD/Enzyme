//===- CostModel.cpp - the measured cost model and cost walks -------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CostModel.h"
#include "FPCore.h"
#include "FPFormat.h"
#include "IR/Utils.h"
#include "Targets/MathLibrary.h"
#include "Targets/Target.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace llvm;

namespace poseidon {

void requireCostModel(CostModelState &CM, const Function &F) {
  const Target &T = Target::of(F);
  if (Error E = requireCostModel(CM, T.processor(F), F.getName(), T.isDevice()))
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

// The row `Opcode,Type`; a model without it is an error at `Loc` in `F`.
static double rowCost(CostModelState &CM, const Function &F,
                      const DiagnosticLocation &Loc, StringRef Opcode,
                      StringRef Type) {
  if (std::optional<double> cost = lookupCost(CM, Opcode, Type))
    return *cost;
  missingRow(CM, F, Loc, Opcode, Type);
}

static double rowCost(CostModelState &CM, const Instruction &At,
                      StringRef Opcode, StringRef Type) {
  return rowCost(CM, *At.getFunction(), At.getDebugLoc(), Opcode, Type);
}

// The cost-model type column of `T`, or empty when it has none.
static StringRef costType(Type *T) {
  const FPFormatInfo *F = formatOf(T);
  return F ? StringRef(F->CostType) : StringRef();
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

// An instruction of the program no row family prices.
[[noreturn]] static void unpriced(const Instruction *I) {
  std::string Text;
  raw_string_ostream(Text) << *I;
  reportUserError(*I, "no cost-model row family prices the instruction" +
                          Twine(Text));
}

// The row family pricing `Call`: the math function it computes, a device fmax
// or fmin priced as the intrinsic it lowers to.
static std::string callRowName(const CallInst &Call) {
  const Function *Callee = Call.getCalledFunction();
  if (!Callee)
    unpriced(&Call);
  if (Callee->isIntrinsic()) {
    // The approximate F32 square root is priced as sqrt.
    std::string Name = isApproxSqrtF32(Callee->getIntrinsicID())
                           ? "sqrt"
                           : intrinsicBaseName(*Callee).str();
    if (!isLibmFunction(Name) &&
        !is_contained({"fmuladd", "maxnum", "minnum"}, Name))
      unpriced(&Call);
    return Name;
  }
  if (StringRef DeviceMath = deviceMathName(Callee->getName());
      !DeviceMath.empty())
    return StringSwitch<std::string>(DeviceMath)
        .Case("fmax", "maxnum")
        .Case("fmin", "minnum")
        .Default(DeviceMath.str());
  StringRef Name = mathFunctionOf(Callee->getName());
  if (!isLibmFunction(Name))
    unpriced(&Call);
  return Name.str();
}

// The row family pricing `I`; none for an instruction that costs nothing.
static std::optional<std::string> rowName(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FNeg:
    return "fneg";
  case Instruction::FAdd:
    return "fadd";
  case Instruction::FSub:
    return "fsub";
  case Instruction::FMul:
    return "fmul";
  case Instruction::FDiv:
    return "fdiv";
  // frem computes C's fmod and is priced by its row.
  case Instruction::FRem:
    return "fmod";
  case Instruction::FCmp:
    return "fcmp";
  case Instruction::FPExt:
    return "fpext";
  case Instruction::FPTrunc:
    return "fptrunc";
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::Load:
  case Instruction::ExtractValue:
    return std::nullopt;
  case Instruction::Call:
    return callRowName(cast<CallInst>(I));
  default:
    unpriced(&I);
  }
}

double getInstructionCompCost(CostModelState &CM, const Instruction *I,
                              const Instruction *At) {
  const Instruction &RowAt = At ? *At : *I;
  // O3 fuses sin(x) and cos(x) into llvm.sincos, whose struct result would
  // otherwise be priced as free.
  if (auto *II = dyn_cast<IntrinsicInst>(I))
    if (II->getIntrinsicID() == Intrinsic::sincos) {
      StringRef P = costType(cast<StructType>(I->getType())->getElementType(0));
      if (P.empty())
        unpriced(I);
      return rowCost(CM, RowAt, "sin", P) + rowCost(CM, RowAt, "cos", P);
    }
  // A regime split's condition is an fcmp, whose i1 result is not an FP type
  // but runs on the FP pipe of its operands.
  if (!I->getType()->isFPOrFPVectorTy() && !isa<FCmpInst>(I))
    return 0.0;
  std::optional<std::string> OpcodeName = rowName(*I);
  if (!OpcodeName)
    return 0;

  Type *Ty = isa<FCmpInst>(I) ? I->getOperand(0)->getType() : I->getType();
  StringRef PrecisionName = costType(Ty);
  if (PrecisionName.empty())
    unpriced(I);
  // A conversion's row is named for both formats and typed by its source.
  if (isa<FPExtInst, FPTruncInst>(I)) {
    StringRef SrcPrecisionName = costType(I->getOperand(0)->getType());
    if (SrcPrecisionName.empty())
      unpriced(I);
    *OpcodeName += ("_" + SrcPrecisionName + "_to_" + PrecisionName).str();
    PrecisionName = SrcPrecisionName;
  }
  return rowCost(CM, RowAt, *OpcodeName, PrecisionName);
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
      return rowCost(CM, F, F.getSubprogram(), Opcode, Type);
    };
    std::unordered_set<std::string> funcs;
    for (const MathFunction &libm : mathFunctions()) {
      if (!libm.Libm)
        continue;
      StringRef func = libm.Name;
      double costFP32 = row(func, "float");
      double costFP64 = row(func, "double");
      double costFPTrunc = row("fptrunc_double_to_float", "double");
      double costFPExt = row("fpext_float_to_double", "float");
      double costFPCast = costFPTrunc + costFPExt;
      if (costFP32 + costFPCast < costFP64)
        funcs.insert(func.str());
    }
    CM.PTFuncs = std::move(funcs);
  }
  return *CM.PTFuncs;
}

static double computeMaxCost(CostModelState &CM, BasicBlock *BB,
                             DenseMap<BasicBlock *, double> &MaxCost,
                             SmallPtrSetImpl<BasicBlock *> &Visited,
                             InstWeight Weight, const Instruction *At) {
  if (auto It = MaxCost.find(BB); It != MaxCost.end())
    return It->second;

  if (!Visited.insert(BB).second)
    return 0;

  double BBCost = 0;
  for (const Instruction &I : *BB) {
    if (I.isTerminator())
      continue;

    double instCost = getInstructionCompCost(CM, &I, At);
    if (Weight)
      instCost *= Weight(I);

    BBCost += instCost;
  }

  double succCost = 0;

  if (!succ_empty(BB)) {
    double maxSuccCost = 0;
    for (BasicBlock *Succ : successors(BB)) {
      double succBBCost =
          computeMaxCost(CM, Succ, MaxCost, Visited, Weight, At);
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

double getCompCost(CostModelState &CM, Function *F, InstWeight Weight,
                   const Instruction *At) {
  DenseMap<BasicBlock *, double> MaxCost;
  SmallPtrSet<BasicBlock *, 16> Visited;

  BasicBlock *EntryBB = &F->getEntryBlock();
  double TotalCost = computeMaxCost(CM, EntryBB, MaxCost, Visited, Weight, At);
  return TotalCost;
}

double getCompCost(CostModelState &CM, ArrayRef<Value *> outputs,
                   const SetVector<Value *> &inputs,
                   const DenseMap<const Instruction *, uint64_t> *opExec,
                   uint64_t normalizer) {
  assert(!outputs.empty() && "a cost walk needs at least one output");
  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;
  double cost = 0;

  todo.insert(todo.end(), outputs.begin(), outputs.end());
  while (!todo.empty()) {
    Value *cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (inputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      double instCost = getInstructionCompCost(CM, I);
      if (opExec && normalizer) {
        auto it = opExec->find(I);
        if (it != opExec->end())
          instCost *= (double)it->second / (double)normalizer;
      }
      cost += instCost;

      for (Value *operand : fpOperands(*I))
        todo.push_back(operand);
    }
  }

  return cost;
}

} // namespace poseidon
