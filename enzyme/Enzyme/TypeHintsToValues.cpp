//=- TypeHintsToValues.cpp - Move type hints off promotable allocas =======//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// A type hint such as `__enzyme_integer(ptr, size)` stays an opaque call until
// Enzyme runs, so SROA treats the memory it names as escaped and keeps the
// whole alloca in memory. A by-value lambda capture with a hint on one of its
// members, for instance, is never promoted to registers.
//
// When the hinted pointer is based on an alloca that does not otherwise escape,
// every read of the hinted memory is in this function. This pass then wraps
// each value loaded from the hinted bytes in a call
//
//   %t = call T @__enzyme_typed_value.T(T %v), !enzyme_type !{...}
//
// carrying the type the hint gives those bytes, and erases the hint. The
// wrapper reads no memory, so SROA can promote the alloca, and Enzyme moves the
// annotation onto the wrapped value (lowerTypedValues) before it analyzes any
// function.
//
// An alloca keeps its hints if it is used by anything other than loads,
// stores to it, memory intrinsics, lifetime markers and its hints, if a load
// from it has a variable offset, or if a memory intrinsic copies out of the
// hinted bytes.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include <optional>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"

#include "TypeAnalysis/TypeAnalysis.h"
#include "TypeHintsToValues.h"
#include "Utils.h"

using namespace llvm;

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "type-hints-to-values"

static constexpr const char TypedValuePrefix[] = "__enzyme_typed_value";

namespace {

struct Hint {
  CallInst *Call;
  /// Offset of the hinted pointer from the alloca.
  int64_t Offset;
  /// Number of bytes the hint covers, or -1 for every offset.
  int64_t Extent;
  /// Types the hint gives the memory, indexed from the hinted pointer.
  TypeTree TT;

  bool overlaps(int64_t Start, int64_t Size) const {
    if (Start + Size <= Offset)
      return false;
    return Extent == -1 || Start < Offset + Extent;
  }
};

struct Access {
  LoadInst *Load;
  int64_t Offset;
};

/// Walk every use of AI, recording its loads with their offsets. Returns false
/// if AI escapes, if a load has a variable offset, or if a memory intrinsic
/// copies out of a hinted range, in which case its hints must stay.
bool collectLoads(AllocaInst *AI, const DataLayout &DL, ArrayRef<Hint> Hints,
                  SmallVectorImpl<Access> &Loads) {
  SmallPtrSet<CallInst *, 4> HintCalls;
  for (auto &H : Hints)
    HintCalls.insert(H.Call);

  unsigned IdxWidth = DL.getIndexTypeSizeInBits(AI->getType());
  // Each entry is a pointer derived from AI, and its offset from AI if that
  // is a constant.
  SmallVector<std::pair<Value *, std::optional<int64_t>>, 8> Todo;
  Todo.push_back({AI, 0});
  SmallPtrSet<Value *, 8> Seen;
  while (!Todo.empty()) {
    auto [V, Off] = Todo.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (Use &U : V->uses()) {
      auto *User = cast<Instruction>(U.getUser());
      if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
        APInt GEPOff(IdxWidth, 0);
        if (Off && GEP->accumulateConstantOffset(DL, GEPOff))
          Todo.push_back({GEP, *Off + GEPOff.getSExtValue()});
        else
          Todo.push_back({GEP, std::nullopt});
        continue;
      }
      if (isa<BitCastInst>(User)) {
        Todo.push_back({User, Off});
        continue;
      }
      if (auto *LI = dyn_cast<LoadInst>(User)) {
        if (!Off || LI->isVolatile())
          return false;
        Loads.push_back({LI, *Off});
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(User)) {
        if (SI->getValueOperand() == V)
          return false;
        continue;
      }
      if (auto *MTI = dyn_cast<MemTransferInst>(User)) {
        if (MTI->getRawSource() != V)
          continue;
        auto *Len = dyn_cast<ConstantInt>(MTI->getLength());
        if (!Off || !Len)
          return false;
        for (auto &H : Hints)
          if (H.overlaps(*Off, Len->getSExtValue()))
            return false;
        continue;
      }
      if (isa<MemSetInst>(User))
        continue;
      if (auto *II = dyn_cast<IntrinsicInst>(User)) {
        if (II->isLifetimeStartOrEnd() || II->isDroppable() ||
            isa<DbgInfoIntrinsic>(II))
          continue;
        return false;
      }
      if (auto *CI = dyn_cast<CallInst>(User))
        if (HintCalls.count(CI))
          continue;
      return false;
    }
  }
  return true;
}

/// The identity function that carries a type annotation for a value of type T.
Function *getTypedValueFn(Module &M, Type *T) {
  std::string Name = TypedValuePrefix;
  raw_string_ostream ss(Name);
  ss << "." << *T;
  if (auto *F = M.getFunction(Name))
    return F;
  auto *F = Function::Create(FunctionType::get(T, {T}, false),
                             GlobalValue::ExternalLinkage, Name, M);
#if LLVM_VERSION_MAJOR >= 16
  F->setDoesNotAccessMemory();
#else
  F->addFnAttr(Attribute::ReadNone);
#endif
  F->setDoesNotThrow();
  F->setWillReturn();
  F->addFnAttr(Attribute::NoSync);
  F->addFnAttr(Attribute::NoFree);
  F->addFnAttr(Attribute::Speculatable);
  return F;
}

bool typeHintsToValues(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();

  MapVector<AllocaInst *, SmallVector<Hint, 2>> HintsByAlloca;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      Function *Called = getFunctionFromCall(CI);
      // Leave malformed hints for type analysis to report.
      if (!Called || CI->arg_size() < 1 || CI->arg_size() > 2 ||
          !CI->getArgOperand(0)->getType()->isPointerTy() ||
          (CI->arg_size() == 2 && !isa<ConstantInt>(CI->getArgOperand(1))))
        continue;
      Hint H;
      H.Call = CI;
      if (!getTypeHintMemoryTree(*CI, Called->getName(), DL, H.TT, &H.Extent))
        continue;
      Value *Ptr = CI->getArgOperand(0);
      APInt Off(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
      auto *AI = dyn_cast<AllocaInst>(Ptr->stripAndAccumulateConstantOffsets(
          DL, Off, /*AllowNonInbounds*/ true));
      if (!AI)
        continue;
      H.Offset = Off.getSExtValue();
      HintsByAlloca[AI].push_back(H);
    }

  bool Changed = false;
  for (auto &[AI, Hints] : HintsByAlloca) {
    SmallVector<Access, 8> Loads;
    if (!collectLoads(AI, DL, Hints, Loads))
      continue;

    for (auto &A : Loads) {
      LoadInst *LI = A.Load;
      int64_t Size = (DL.getTypeSizeInBits(LI->getType()) + 7) / 8;
      TypeTree ValueTT;
      for (auto &H : Hints) {
        if (!H.overlaps(A.Offset, Size))
          continue;
        // The hinted types within the loaded bytes, indexed from the load.
        int64_t Rel = A.Offset - H.Offset;
        TypeTree Mem = Rel >= 0 ? H.TT.ShiftIndices(DL, Rel, Size, 0)
                                : H.TT.ShiftIndices(DL, 0, Size + Rel, -Rel);
        ValueTT |= Mem.Only(-1, nullptr).Lookup(Size, DL);
      }
      if (!ValueTT.isKnown())
        continue;

      IRBuilder<> B(LI->getNextNode());
      auto *Typed =
          B.CreateCall(getTypedValueFn(*F.getParent(), LI->getType()), {LI});
      Typed->setMetadata("enzyme_type", ValueTT.toMD(LI->getContext()));
      LI->replaceUsesWithIf(Typed,
                            [&](Use &U) { return U.getUser() != Typed; });
    }

    for (auto &H : Hints)
      H.Call->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

class TypeHintsToValues final : public FunctionPass {
public:
  static char ID;
  TypeHintsToValues() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override { return typeHintsToValues(F); }
};

} // namespace

bool lowerTypedValues(Module &M) {
  bool Changed = false;
  for (Function &F : make_early_inc_range(M)) {
    if (!startsWith(F.getName(), TypedValuePrefix))
      continue;
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &F)
        continue;
      Value *V = CI->getArgOperand(0);
      if (auto *I = dyn_cast<Instruction>(V)) {
        TypeTree TT;
        if (auto *MD = CI->getMetadata("enzyme_type"))
          TT = TypeTree::fromMD(MD);
        if (auto *MD = I->getMetadata("enzyme_type")) {
          // Keep the existing annotation where the two disagree.
          TypeTree Merged = TypeTree::fromMD(MD);
          bool Legal = true;
          Merged.checkedOrIn(TT, /*PointerIntSame*/ false, Legal);
          if (Legal)
            TT = Merged;
          else
            TT = TypeTree::fromMD(MD);
        }
        I->setMetadata("enzyme_type", TT.toMD(I->getContext()));
      }
      CI->replaceAllUsesWith(V);
      CI->eraseFromParent();
      Changed = true;
    }
    if (F.use_empty())
      F.eraseFromParent();
  }
  return Changed;
}

char TypeHintsToValues::ID = 0;

static RegisterPass<TypeHintsToValues>
    X("type-hints-to-values",
      "Move Enzyme type hints off allocas that do not otherwise escape");

TypeHintsToValuesNewPM::Result
TypeHintsToValuesNewPM::run(Function &F, FunctionAnalysisManager &FAM) {
  if (!typeHintsToValues(F))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

llvm::AnalysisKey TypeHintsToValuesNewPM::Key;
