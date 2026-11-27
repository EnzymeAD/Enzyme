//===- Recognize.cpp - what both product recognizers use ------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/NumberFormat.h"
#include "MatmulInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

namespace {
// A multiply-add: llvm.fma / llvm.fmuladd (`fmul` null), or a contractable
// fadd of a contractable fmul; `acc` is the addend.
struct FMAMatch {
  Instruction *fma;
  Instruction *fmul;
  Value *a, *b, *acc;
};
} // namespace

static std::optional<FMAMatch> matchFMA(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return std::nullopt;
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    Intrinsic::ID id = II->getIntrinsicID();
    if (id == Intrinsic::fma || id == Intrinsic::fmuladd)
      return FMAMatch{II, nullptr, II->getArgOperand(0), II->getArgOperand(1),
                      II->getArgOperand(2)};
  }
  auto *fadd = dyn_cast<BinaryOperator>(I);
  if (!fadd || fadd->getOpcode() != Instruction::FAdd)
    return std::nullopt;
  if (!fadd->getFastMathFlags().allowContract())
    return std::nullopt;
  for (unsigned side = 0; side < 2; ++side) {
    auto *fmulI = dyn_cast<BinaryOperator>(fadd->getOperand(side));
    if (!fmulI || fmulI->getOpcode() != Instruction::FMul)
      continue;
    if (!fmulI->getFastMathFlags().allowContract())
      continue;
    return FMAMatch{fadd, fmulI, fmulI->getOperand(0), fmulI->getOperand(1),
                    fadd->getOperand(1 - side)};
  }
  return std::nullopt;
}

// Whether every user of `v` inside `L` is `expected`.
static bool hasOnlyInLoopUser(const Loop *L, Value *v, Instruction *expected) {
  for (User *U : v->users()) {
    auto *uI = dyn_cast<Instruction>(U);
    if (!uI || !L->contains(uI))
      continue;
    if (uI != expected)
      return false;
  }
  return true;
}

static bool isClosedFmaPhiCycle(const Loop *L, Instruction *fma, PHINode *phi) {
  return hasOnlyInLoopUser(L, fma, phi) && hasOnlyInLoopUser(L, phi, fma);
}

Expected<ReductionCycle> matchReductionCycle(const Loop *L, PHINode &phi,
                                             BasicBlock *latch) {
  auto fail = [](const Twine &reason) {
    return createStringError(inconvertibleErrorCode(), reason);
  };
  Value *backedgeVal = phi.getIncomingValueForBlock(latch);
  if (!backedgeVal)
    return fail("no backedge value");
  std::optional<FMAMatch> fm = matchFMA(backedgeVal);
  if (!fm || fm->acc != &phi)
    return fail("backedge value is not an accumulating fma");
  // Otherwise a per-iteration partial sum is live and the loop is not a pure
  // reduction.
  if (!isClosedFmaPhiCycle(L, fm->fma, &phi))
    return fail("fma/phi cycle is not closed");
  auto *aLoad = dyn_cast<LoadInst>(fm->a);
  auto *bLoad = dyn_cast<LoadInst>(fm->b);
  if (!aLoad || !bLoad)
    return fail("an fma multiplicand is not a load");
  if (aLoad->getType() != phi.getType() || bLoad->getType() != phi.getType())
    return fail("mixed-precision operands");
  return ReductionCycle{fm->fma, fm->fmul, aLoad, bLoad};
}

bool isPositiveZeroSeed(const Value *V) {
  SmallVector<const Value *, 4> work{V};
  SmallPtrSet<const Value *, 4> seen;
  while (!work.empty()) {
    const Value *cur = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    if (const auto *C = dyn_cast<ConstantFP>(cur)) {
      if (!C->isZero() || C->isNegative())
        return false;
      continue;
    }
    const auto *P = dyn_cast<PHINode>(cur);
    if (!P)
      return false;
    append_range(work, P->incoming_values());
  }
  return true;
}

std::optional<StringRef> reductionLoopShapeRefusal(const Loop &L) {
  BasicBlock *latch = L.getLoopLatch();
  if (!latch)
    return "the loop has no unique latch";
  if (L.getExitingBlock() != latch)
    return "the loop exits from a block other than its latch";
  if (!L.getExitBlock())
    return "the loop has no unique exit block";
  if (!L.getLoopPreheader())
    return "the loop has no preheader";
  return std::nullopt;
}

void forEachInnermostLoop(LoopInfo &LI, function_ref<void(Loop *)> fn) {
  SmallVector<Loop *, 8> worklist(LI.begin(), LI.end());
  while (!worklist.empty()) {
    Loop *L = worklist.pop_back_val();
    if (L->getSubLoops().empty())
      fn(L);
    else
      append_range(worklist, L->getSubLoops());
  }
}

bool hasEffect(const Instruction &I) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&I);
      II && II->isAssumeLikeIntrinsic())
    return false;
  return I.mayHaveSideEffects();
}

std::optional<std::string> libraryElementRefusal(NumberFormat elem) {
  if (elem == NumberFormat::FP64)
    return std::nullopt;
  return "the product's elements are " + std::string(dtypeName(elem)) +
         ", and every library-call entry is a GEMM of doubles";
}

} // namespace poseidon
