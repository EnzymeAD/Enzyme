//===- CostModel.cpp - the measured cost model and cost walks -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CostModel.h"
#include "Core/FPCore.h"
#include "Core/NumberFormat.h"
#include "Profile/ProfileReader.h"
#include "Support/Diagnostics.h"
#include "Support/Utils.h"
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

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>

using namespace llvm;

namespace poseidon {

Error requireCostModel(Session &S, const Function &F) {
  const Target &T = Target::of(F);
  if (Error E =
          requireCostModel(S.get<CostModelState>(), S.Opts.Model.CostModel,
                           T.processor(F), F.getName(), T.isDevice()))
    return userError(F, toString(std::move(E)));
  return Error::success();
}

Expected<double> rowCost(CostModelState &CM, const Function &F,
                         const DiagnosticLocation &Loc, StringRef Opcode,
                         StringRef Type) {
  if (std::optional<double> cost = lookupCost(CM, Opcode, Type))
    return *cost;
  return userError(F, Loc,
                   "cost model " + Twine(costModelPath(CM)) + " has no row '" +
                       Opcode + "," + Type +
                       "'; measure it with poseidon-calibrate");
}

static Expected<double> rowCost(CostModelState &CM, const Instruction &At,
                                StringRef Opcode, StringRef Type) {
  return rowCost(CM, *At.getFunction(), At.getDebugLoc(), Opcode, Type);
}

// The cost-model type column of `T`, or empty when it has none.
static StringRef costType(Type *T) {
  const NumberFormatInfo *F = formatOf(T);
  return F ? StringRef(F->CostType) : StringRef();
}

Expected<const std::unordered_set<std::string> *>
scalarTypes(CostModelState &CM, const Function &F) {
  if (CM.ScalarTypes.empty())
    return userError(F,
                     "cost model " + Twine(costModelPath(CM)) +
                         " carries no '# scalar_types=' header, so the formats "
                         "the device has hardware for are unknown");
  return &CM.ScalarTypes;
}

// An instruction of the program no row family prices.
static Error unpriced(const Instruction *I) {
  std::string Text;
  raw_string_ostream(Text) << *I;
  return userError(*I, "no cost-model row family prices the instruction" +
                           Twine(Text));
}

// The row family pricing `Call`: the math function it computes, a device fmax
// or fmin priced as the intrinsic it lowers to.
static Expected<std::string> callRowName(const CallInst &Call) {
  const Function *Callee = Call.getCalledFunction();
  if (!Callee)
    return unpriced(&Call);
  if (Callee->isIntrinsic()) {
    // The approximate F32 square root is priced as sqrt.
    std::string Name = isApproxSqrtF32(Callee->getIntrinsicID())
                           ? "sqrt"
                           : intrinsicBaseName(*Callee).str();
    if (!isLibmFunction(Name) &&
        !is_contained({"fmuladd", "maxnum", "minnum"}, Name))
      return unpriced(&Call);
    return Name;
  }
  if (std::optional<StringRef> DeviceMath = deviceMathName(Callee->getName()))
    return StringSwitch<std::string>(*DeviceMath)
        .Case("fmax", "maxnum")
        .Case("fmin", "minnum")
        .Default(DeviceMath->str());
  StringRef Name = mathFunctionOf(Callee->getName());
  if (!isLibmFunction(Name))
    return unpriced(&Call);
  return Name.str();
}

// The row family pricing `I`; none for an instruction that costs nothing.
static Expected<std::optional<std::string>> rowName(const Instruction &I) {
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
  case Instruction::Call: {
    Expected<std::string> Name = callRowName(cast<CallInst>(I));
    if (!Name)
      return Name.takeError();
    return std::optional<std::string>(std::move(*Name));
  }
  default:
    return unpriced(&I);
  }
}

Expected<double> instructionCost(CostModelState &CM, const Instruction *I,
                                 const Instruction *At) {
  const Instruction &RowAt = At ? *At : *I;
  // O3 fuses sin(x) and cos(x) into llvm.sincos, whose struct result would
  // otherwise be priced as free.
  if (auto *II = dyn_cast<IntrinsicInst>(I))
    if (II->getIntrinsicID() == Intrinsic::sincos) {
      StringRef P = costType(cast<StructType>(I->getType())->getElementType(0));
      if (P.empty())
        return unpriced(I);
      Expected<double> Sin = rowCost(CM, RowAt, "sin", P);
      if (!Sin)
        return Sin.takeError();
      Expected<double> Cos = rowCost(CM, RowAt, "cos", P);
      if (!Cos)
        return Cos.takeError();
      return *Sin + *Cos;
    }
  // A regime split's condition is an fcmp, whose i1 result is not an FP type
  // but runs on the FP pipe of its operands.
  if (!I->getType()->isFPOrFPVectorTy() && !isa<FCmpInst>(I))
    return 0.0;
  Expected<std::optional<std::string>> OpcodeName = rowName(*I);
  if (!OpcodeName)
    return OpcodeName.takeError();
  if (!*OpcodeName)
    return 0.0;

  Type *Ty = isa<FCmpInst>(I) ? I->getOperand(0)->getType() : I->getType();
  StringRef PrecisionName = costType(Ty);
  if (PrecisionName.empty())
    return unpriced(I);
  // A conversion's row is named for both formats and typed by its source.
  if (isa<FPExtInst, FPTruncInst>(I)) {
    StringRef SrcPrecisionName = costType(I->getOperand(0)->getType());
    if (SrcPrecisionName.empty())
      return unpriced(I);
    **OpcodeName += ("_" + SrcPrecisionName + "_to_" + PrecisionName).str();
    PrecisionName = SrcPrecisionName;
  }
  return rowCost(CM, RowAt, **OpcodeName, PrecisionName);
}

Expected<double>
profiledCost(CostModelState &CM, const Function &F, const ProfileMap &profile,
             function_ref<std::optional<size_t>(const Instruction *)> slotOf) {
  double cost = 0.0;
  for (const Instruction &I : instructions(F)) {
    std::optional<size_t> slot = slotOf(&I);
    if (!slot)
      continue;
    auto it = profile.find(*slot);
    if (it == profile.end())
      continue;
    Expected<double> price = instructionCost(CM, &I);
    if (!price)
      return price.takeError();
    cost += (double)it->second.exec * *price;
  }
  return cost;
}

Expected<std::optional<StreamPrices>> streamPrices(CostModelState &CM,
                                                   const Function &F) {
  static constexpr std::pair<StringLiteral, StringLiteral> Rows[] = {
      {"stream_load", "double"},   {"stream_store", "double"},
      {"stream_load", "float"},    {"stream_store", "float"},
      {"stream_combine", "pnorm"}, {"stream_combine", "band"}};
  if (none_of(Rows, [&](const auto &Row) {
        return lookupCost(CM, Row.first, Row.second).has_value();
      }))
    return std::nullopt;
  double Price[std::size(Rows)];
  for (auto [Cost, Row] : zip(Price, Rows)) {
    Expected<double> Found =
        rowCost(CM, F, F.getSubprogram(), Row.first, Row.second);
    if (!Found)
      return Found.takeError();
    Cost = *Found;
  }
  return StreamPrices{Price[0], Price[1], Price[2],
                      Price[3], Price[4], Price[5]};
}

double StreamPrices::perByte(bool Store, uint64_t Width) const {
  if (Width != 0 && Width <= 4)
    return (Store ? StoreFloat : LoadFloat) / 4;
  return (Store ? StoreDouble : LoadDouble) / 8;
}

double StreamPrices::combine(double Arith, double Memory) const {
  double Hi = std::max(Arith, Memory), Lo = std::min(Arith, Memory);
  if (Lo <= 0.0)
    return Hi;
  return Hi * std::pow(1.0 + std::pow(Lo / Hi, PNorm), 1.0 / PNorm);
}

double trafficCost(const StreamPrices &Prices,
                   ArrayRef<TrafficRecord> Records) {
  double Cost = 0.0;
  for (const TrafficRecord &R : Records)
    Cost += (double)R.LoadBytes * Prices.perByte(false, R.Width) +
            (double)R.StoreBytes * Prices.perByte(true, R.Width);
  return Cost;
}

static Expected<double> computeMaxCost(CostModelState &CM, BasicBlock *BB,
                                       DenseMap<BasicBlock *, double> &MaxCost,
                                       SmallPtrSetImpl<BasicBlock *> &Visited,
                                       InstWeight Weight,
                                       const Instruction *At) {
  if (auto It = MaxCost.find(BB); It != MaxCost.end())
    return It->second;

  if (!Visited.insert(BB).second)
    return 0.0;

  double BBCost = 0;
  for (const Instruction &I : *BB) {
    if (I.isTerminator())
      continue;

    Expected<double> instCost = instructionCost(CM, &I, At);
    if (!instCost)
      return instCost.takeError();
    if (Weight)
      *instCost *= Weight(I);

    BBCost += *instCost;
  }

  double succCost = 0;

  if (!succ_empty(BB)) {
    double maxSuccCost = 0;
    for (BasicBlock *Succ : successors(BB)) {
      Expected<double> succBBCost =
          computeMaxCost(CM, Succ, MaxCost, Visited, Weight, At);
      if (!succBBCost)
        return succBBCost.takeError();
      if (*succBBCost > maxSuccCost)
        maxSuccCost = *succBBCost;
    }
    succCost = maxSuccCost;
  }

  double totalCost = BBCost + succCost;
  MaxCost[BB] = totalCost;
  Visited.erase(BB);
  return totalCost;
}

Expected<double> compCost(CostModelState &CM, Function *F, InstWeight Weight,
                          const Instruction *At) {
  DenseMap<BasicBlock *, double> MaxCost;
  SmallPtrSet<BasicBlock *, 16> Visited;
  return computeMaxCost(CM, &F->getEntryBlock(), MaxCost, Visited, Weight, At);
}

SmallVector<Instruction *, 32>
backwardCone(ArrayRef<Value *> roots, function_ref<bool(Value *)> isInput) {
  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo(roots);
  SmallVector<Instruction *, 32> cone;
  while (!todo.empty()) {
    Value *cur = todo.pop_back_val();
    if (!seen.insert(cur).second || isInput(cur))
      continue;
    if (auto *I = dyn_cast<Instruction>(cur)) {
      cone.push_back(I);
      append_range(todo, fpOperands(*I));
    }
  }
  return cone;
}

Expected<double> compCost(CostModelState &CM, ArrayRef<Value *> outputs,
                          const SetVector<Value *> &inputs,
                          const DenseMap<const Instruction *, uint64_t> *opExec,
                          uint64_t normalizer) {
  assert(!outputs.empty() && "a cost walk needs at least one output");
  double cost = 0;
  for (Instruction *I :
       backwardCone(outputs, [&](Value *V) { return inputs.contains(V); })) {
    Expected<double> instCost = instructionCost(CM, I);
    if (!instCost)
      return instCost.takeError();
    if (opExec && normalizer) {
      auto it = opExec->find(I);
      if (it != opExec->end())
        *instCost *= (double)it->second / (double)normalizer;
    }
    cost += *instCost;
  }
  return cost;
}

} // namespace poseidon
