//===- RecognizeScalarLoop.cpp - products in constant-trip reductions -----===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/Options.h"
#include "MatmulInternal.h"
#include "Model/FPFormat.h"
#include "Profile/ProfileRead.h"
#include "Sites/Canonicalize.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <array>
#include <map>
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

static int64_t outerGEPElementSize(const DataLayout &DL, Value *ptr) {
  GEPOperator *outermost = nullptr;
  for (Value *cur = ptr;;) {
    auto *gep = dyn_cast<GEPOperator>(cur);
    if (!gep)
      break;
    outermost = gep;
    cur = gep->getPointerOperand();
  }
  if (!outermost)
    return 0;
  return (int64_t)DL.getTypeAllocSize(outermost->getSourceElementType())
      .getFixedValue();
}

static int64_t extractLeadingDimByte(const DataLayout &DL, LoadInst *load,
                                     int64_t fallbackByte) {
  int64_t scalarBytes =
      (int64_t)DL.getTypeAllocSize(load->getType()).getFixedValue();
  int64_t outer = outerGEPElementSize(DL, load->getPointerOperand());
  return (outer > scalarBytes) ? outer : fallbackByte;
}

static void extractTidCoefficients(const SCEV *S, int64_t mult,
                                   std::array<int64_t, 3> &out) {
  while (auto *cast = dyn_cast<SCEVCastExpr>(S))
    S = cast->getOperand();
  if (auto *add = dyn_cast<SCEVAddExpr>(S)) {
    for (const SCEV *Op : add->operands())
      extractTidCoefficients(Op, mult, out);
    return;
  }
  if (auto *rec = dyn_cast<SCEVAddRecExpr>(S)) {
    extractTidCoefficients(rec->getStart(), mult, out);
    return;
  }
  if (auto *mul = dyn_cast<SCEVMulExpr>(S)) {
    if (mul->getNumOperands() == 2) {
      if (auto *cst = dyn_cast<SCEVConstant>(mul->getOperand(0))) {
        int64_t c = cst->getAPInt().getSExtValue();
        extractTidCoefficients(mul->getOperand(1), mult * c, out);
      }
    }
    return;
  }
  auto *unk = dyn_cast<SCEVUnknown>(S);
  if (!unk)
    return;
  LaunchGeometry g = launchGeometryOf(unk->getValue());
  if (g.Kind != LaunchValue::ThreadIndex)
    return;
  out[g.Axis] += mult;
}

std::array<int64_t, 3> threadIndexCoefficients(const SCEV *S) {
  std::array<int64_t, 3> coef = {0, 0, 0};
  extractTidCoefficients(S, 1, coef);
  return coef;
}

// `blockIdx.a * blockDim.a + threadIdx.a` on one axis, modulo casts and operand
// order; an intra-block index would bound the tile, not the matrix.
static bool isGlobalThreadIndex(const Value *v, int &axis) {
  while (const auto *CI = dyn_cast<CastInst>(v))
    v = CI->getOperand(0);
  const auto *add = dyn_cast<BinaryOperator>(v);
  if (!add || add->getOpcode() != Instruction::Add)
    return false;
  for (unsigned i = 0; i < 2; ++i) {
    LaunchGeometry t = launchGeometryOf(add->getOperand(i));
    if (t.Kind != LaunchValue::ThreadIndex)
      continue;
    const auto *mul = dyn_cast<BinaryOperator>(add->getOperand(1 - i));
    if (!mul || mul->getOpcode() != Instruction::Mul)
      continue;
    for (unsigned j = 0; j < 2; ++j) {
      LaunchGeometry c = launchGeometryOf(mul->getOperand(j));
      LaunchGeometry n = launchGeometryOf(mul->getOperand(1 - j));
      if (c.Kind == LaunchValue::BlockIndex &&
          n.Kind == LaunchValue::BlockDim && c.Axis == t.Axis &&
          n.Axis == t.Axis) {
        axis = t.Axis;
        return true;
      }
    }
  }
  return false;
}

// Tightens `bound` on the axis of `idx` to `ub` when `idx` is a global thread
// index.
static bool boundGlobalThreadIndex(const Value *idx, uint64_t ub,
                                   ThreadIndexBounds &bound) {
  int axis = -1;
  if (!isGlobalThreadIndex(idx, axis))
    return false;
  bound.tighten(axis, ub);
  return true;
}

static bool collectIndexBounds(const Value *cond, bool holds,
                               ThreadIndexBounds &bound, unsigned depth) {
  if (depth > 8)
    return false;
  if (const auto *BO = dyn_cast<BinaryOperator>(cond)) {
    if ((BO->getOpcode() == Instruction::Or && !holds) ||
        (BO->getOpcode() == Instruction::And && holds))
      return collectIndexBounds(BO->getOperand(0), holds, bound, depth + 1) &&
             collectIndexBounds(BO->getOperand(1), holds, bound, depth + 1);
    return false;
  }
  const auto *IC = dyn_cast<ICmpInst>(cond);
  if (!IC)
    return false;
  CmpInst::Predicate P = holds ? IC->getPredicate() : IC->getInversePredicate();
  const Value *idx = IC->getOperand(0);
  const auto *cst = dyn_cast<ConstantInt>(IC->getOperand(1));
  if (!cst) {
    cst = dyn_cast<ConstantInt>(IC->getOperand(0));
    if (!cst)
      return false;
    idx = IC->getOperand(1);
    P = CmpInst::getSwappedPredicate(P);
  }
  if (cst->getValue().isNegative() || cst->getValue().getActiveBits() > 31)
    return false;
  uint64_t ub;
  switch (P) {
  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_ULT:
    ub = cst->getZExtValue();
    break;
  case CmpInst::ICMP_SLE:
  case CmpInst::ICMP_ULE:
    ub = cst->getZExtValue() + 1;
    break;
  default:
    return false;
  }
  if (ub == 0)
    return false;
  // (a | b) u< 2^n is how (a u< 2^n) && (b u< 2^n) folds.
  if (const auto *Or = dyn_cast<BinaryOperator>(idx);
      Or && Or->getOpcode() == Instruction::Or && CmpInst::isUnsigned(P) &&
      isPowerOf2_64(ub))
    return boundGlobalThreadIndex(Or->getOperand(0), ub, bound) &&
           boundGlobalThreadIndex(Or->getOperand(1), ub, bound);
  return boundGlobalThreadIndex(idx, ub, bound);
}

bool collectThreadIndexBounds(const Value *cond, bool holds,
                              ThreadIndexBounds &bound) {
  return collectIndexBounds(cond, holds, bound, 0);
}

// The matrix index whose threads' byte offsets are `S`: one thread axis, or
// exactly two, byteOffset = unit * tid[fast] + unit * mult * tid[slow]
// (sum-factorized FEM contractions). Sets `unitByte` to the byte step of one
// unit of the index (0 when it is not positive). An index of no axis, three
// axes, a non-divisible ratio or equal coefficients has fast axis Other.
static MatrixIndex classifyTidIndex(const SCEV *S, int64_t &unitByte) {
  unitByte = 0;
  MatrixIndex idx;
  std::array<int64_t, 3> coef = threadIndexCoefficients(S);
  unsigned n = (coef[0] != 0) + (coef[1] != 0) + (coef[2] != 0);
  static const TidAxis kAxis[3] = {TidAxis::TidX, TidAxis::TidY, TidAxis::TidZ};
  if (n == 1) {
    int i = coef[0] ? 0 : coef[1] ? 1 : 2;
    if (coef[i] > 0)
      unitByte = coef[i];
    idx.fast = kAxis[i];
    return idx;
  }
  idx.fast = TidAxis::Other;
  if (n != 2)
    return idx;

  int lo = -1, hi = -1;
  for (int i = 0; i < 3; ++i) {
    if (coef[i] == 0)
      continue;
    if (coef[i] < 0)
      return idx; // negative index coefficient: not a tensor index
    if (lo < 0 || coef[i] < coef[lo])
      lo = i;
    if (hi < 0 || coef[i] > coef[hi])
      hi = i;
  }
  if (lo < 0 || hi < 0 || lo == hi)
    return idx; // equal coefficients: extents are not separable
  if (coef[hi] % coef[lo] != 0)
    return idx;

  idx.fast = kAxis[lo];
  idx.slow = kAxis[hi];
  idx.fuseMult = (unsigned)(coef[hi] / coef[lo]);
  unitByte = coef[lo];
  return idx;
}

// The extension of the integer `v` to `wide`, distributed over the no-wrap
// arithmetic `v` is computed by: sext over nsw add, sub, mul and shl by a
// constant, zext over the nuw ones. Contract: `v` reaches the address of an
// executed load through an inbounds GEP. A wrap of any of these operations
// makes that address poison and the load undefined, so in every defined
// execution none wraps and the extension of the result equals the arithmetic
// on the extended operands. This makes the address of A[row * K + k] with int
// indices an affine function of k, which ScalarEvolution alone does not see.
static const SCEV *extendedIndex(ScalarEvolution &SE, Value *v, Type *wide,
                                 bool isSigned, unsigned depth = 0) {
  auto leaf = [&] {
    return isSigned ? SE.getSignExtendExpr(SE.getSCEV(v), wide)
                    : SE.getZeroExtendExpr(SE.getSCEV(v), wide);
  };
  auto *BO = dyn_cast<OverflowingBinaryOperator>(v);
  if (!BO || depth > 8 ||
      !(isSigned ? BO->hasNoSignedWrap() : BO->hasNoUnsignedWrap()))
    return leaf();
  auto operand = [&](unsigned i) {
    return extendedIndex(SE, BO->getOperand(i), wide, isSigned, depth + 1);
  };
  switch (BO->getOpcode()) {
  case Instruction::Add:
    return SE.getAddExpr(operand(0), operand(1));
  case Instruction::Sub:
    return SE.getMinusSCEV(operand(0), operand(1));
  case Instruction::Mul:
    return SE.getMulExpr(operand(0), operand(1));
  case Instruction::Shl:
    if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
        C && C->getValue().ult(wide->getIntegerBitWidth()))
      return SE.getMulExpr(operand(0),
                           SE.getConstant(APInt::getOneBitSet(
                               wide->getIntegerBitWidth(), C->getZExtValue())));
    return leaf();
  default:
    return leaf();
  }
}

// The address `ptr` of an executed load with the extended index of an
// inbounds GEP taken by extendedIndex.
static const SCEV *extendedAddress(ScalarEvolution &SE, const DataLayout &DL,
                                   Value *ptr, unsigned depth = 0) {
  auto *GEP = dyn_cast<GetElementPtrInst>(ptr);
  if (!GEP || depth > 8 || !GEP->isInBounds() || GEP->getNumIndices() != 1)
    return SE.getSCEV(ptr);
  auto *ext = dyn_cast<CastInst>(GEP->getOperand(1));
  if (!ext || !(isa<SExtInst>(ext) || isa<ZExtInst>(ext)) ||
      ext->getType() != DL.getIndexType(GEP->getType()))
    return SE.getSCEV(ptr);
  Type *idxTy = ext->getType();
  const SCEV *idx =
      extendedIndex(SE, ext->getOperand(0), idxTy, isa<SExtInst>(ext));
  const SCEV *size = SE.getConstant(
      idxTy, DL.getTypeAllocSize(GEP->getSourceElementType()).getFixedValue());
  return SE.getAddExpr(
      extendedAddress(SE, DL, GEP->getPointerOperand(), depth + 1),
      SE.getMulExpr(size, idx));
}

// The handle of the reduction `rc` of `L` when its operands index a matrix
// product: A steps by one element and B by whole elements along the
// reduction, and A's row and B's column are thread indices. An error saying
// why otherwise.
static Expected<ScalarLoopHandle>
scalarLoopHandle(Loop *L, const ReductionCycle &rc, ScalarEvolution &SE,
                 const DataLayout &DL, unsigned trip) {
  Type *phiTy = rc.aLoad->getType();
  LoadInst *aLoad = rc.aLoad;
  LoadInst *bLoad = rc.bLoad;

  // An address ScalarEvolution does not see as a recurrence of L may still
  // be one once its index's extension is distributed.
  auto address = [&](LoadInst *load) {
    const SCEV *S = SE.getSCEV(load->getPointerOperand());
    if (auto *rec = dyn_cast<SCEVAddRecExpr>(S); rec && rec->getLoop() == L)
      return S;
    return extendedAddress(SE, DL, load->getPointerOperand());
  };
  const SCEV *aAddr = address(aLoad);
  const SCEV *bAddr = address(bLoad);
  auto *aRec = dyn_cast<SCEVAddRecExpr>(aAddr);
  auto *bRec = dyn_cast<SCEVAddRecExpr>(bAddr);
  auto refuse = [](const Twine &reason) {
    return createStringError(inconvertibleErrorCode(), reason);
  };
  if (!aRec || !bRec || aRec->getLoop() != L || bRec->getLoop() != L)
    return refuse("an operand's address is not an affine function of the "
                  "loop's counter");
  auto *aStepC = dyn_cast<SCEVConstant>(aRec->getStepRecurrence(SE));
  auto *bStepC = dyn_cast<SCEVConstant>(bRec->getStepRecurrence(SE));
  if (!aStepC || !bStepC)
    return refuse("an operand's address does not step by a constant");
  int64_t aStrideByte = aStepC->getAPInt().getSExtValue();
  int64_t bStrideByte = bStepC->getAPInt().getSExtValue();

  // matmul A invariant
  int64_t srcByteSize = (int64_t)DL.getTypeAllocSize(phiTy).getFixedValue();
  if (aStrideByte != srcByteSize)
    return refuse("A does not step by one element along the reduction");
  // matmul B invariant
  if (bStrideByte <= 0 || bStrideByte % srcByteSize != 0)
    return refuse("B does not step by whole elements along the reduction");

  int64_t aRowUnit = 0, bColUnit = 0;
  MatrixIndex aRow = classifyTidIndex(aRec->getStart(), aRowUnit);
  MatrixIndex bCol = classifyTidIndex(bRec->getStart(), bColUnit);
  if (!tidAxisIndex(aRow.fast) || !tidAxisIndex(bCol.fast))
    return refuse("A's row or B's column is not a thread index of one axis, "
                  "or of two");
  // Axis disjointness is required only once a fused index is involved: sharing
  // a thread axis between the two matrix indices makes the fused numbering
  // ambiguous.
  if (aRow.fused() || bCol.fused()) {
    if (aRow.fast == bCol.fast ||
        (aRow.fused() && (aRow.slow == bCol.fast || aRow.slow == bCol.slow)) ||
        (bCol.fused() && bCol.slow == aRow.fast))
      return refuse("a fused index shares a thread axis with the other "
                    "index");
  }

  // Leading dimensions: prefer the byte step SCEV reports per unit of the
  // row/column index, which is the leading dimension for any pointer shape;
  // the fallback srcByteSize * trip assumes an unpadded row of K elements.
  int64_t aLdByte = (aRowUnit > 0)
                        ? aRowUnit
                        : extractLeadingDimByte(DL, aLoad, srcByteSize * trip);
  int64_t bLdByte = extractLeadingDimByte(DL, bLoad, srcByteSize * trip);

  // A fused index steps the operand by `unitByte` per unit of the fused index;
  // that must be the stride the single-axis path uses, or the block-base
  // back-out (start - idx*stride) lands off the tile.
  if (aRow.fused() && aRowUnit != aLdByte)
    return refuse("A's fused row index does not step by A's leading "
                  "dimension");
  // B's per-column step is the element size when B is K-strided (row-major)
  // and its leading dimension when B is K-contiguous (col-major).
  bool bIsRowMajor = (bStrideByte != srcByteSize);
  if (!bIsRowMajor && bColUnit > 0)
    bLdByte = bColUnit;
  if (bCol.fused() && bColUnit != (bIsRowMajor ? srcByteSize : bLdByte))
    return refuse("B's fused column index does not step by B's column "
                  "stride");

  ScalarLoopHandle h;
  h.preheader = L->getLoopPreheader();
  h.exitBB = L->getExitBlock();
  for (BasicBlock *BB : L->blocks())
    h.blocks.insert(BB);
  h.fma = rc.fma;
  h.fmul = rc.fmul;
  h.aPtr = aLoad->getPointerOperand();
  h.bPtr = bLoad->getPointerOperand();
  h.aStartSCEV = aRec->getStart();
  h.bStartSCEV = bRec->getStart();
  h.SE = &SE;
  h.aRow = aRow;
  h.bCol = bCol;
  h.aStrideByte = aStrideByte;
  h.bStrideByte = bStrideByte;
  h.aLeadingDimByte = aLdByte;
  h.bLeadingDimByte = bLdByte;
  return h;
}

static unsigned dimForAxis(const uint32_t (&dims)[3], TidAxis axis) {
  if (std::optional<unsigned> i = tidAxisIndex(axis))
    return dims[*i];
  return 0;
}

// Extent of a (possibly fused) matrix index: dim[fast], or dim[fast]*dim[slow]
// for a fused pair. The source's fusion multiplier must equal the profiled
// extent of the fast axis, otherwise the index is not a dense numbering of the
// block's threads; return 0 (unknown) in that case so the caller skips the
// site.
static unsigned extentForIndex(const Options &O, const uint32_t (&dims)[3],
                               const MatrixIndex &idx, StringRef what,
                               StringRef fname) {
  unsigned fastDim = dimForAxis(dims, idx.fast);
  if (!idx.fused())
    return fastDim;
  unsigned slowDim = dimForAxis(dims, idx.slow);
  if (fastDim == 0 || slowDim == 0)
    return 0;
  if (idx.fuseMult != fastDim) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << fname << ": fused " << what
                   << " index multiplier " << idx.fuseMult
                   << " does not match the profiled block extent " << fastDim
                   << " of its fast thread axis; the fused index is not a "
                      "dense numbering of the block's threads, so the "
                      "product is skipped rather than given a guessed "
                      "extent\n";
    return 0;
  }
  return fastDim * slowDim;
}

// Walk F's innermost scalar loops, invoking `cb` for each matmul-shaped
// reduction phi. The profile header is not consulted, so this also runs at
// profile generation.
static void forEachScalarLoopReduction(
    const Options &O, Function &F, ScalarEvolution &SE, LoopInfo &LI,
    function_ref<void(Loop *, ScalarLoopHandle &, FPKind, unsigned)> cb) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto reject = [&](StringRef why) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] reject loop in " << F.getName() << ": " << why
                   << '\n';
  };
  forEachInnermostLoop(LI, [&](Loop *L) {
    BasicBlock *latch = L->getLoopLatch();
    if (!latch) {
      reject("the loop has no unique latch");
      return;
    }
    // Each header phi is tested independently, so a loop with several
    // reductions yields one matmul each; the loop's refusals are reported for
    // the phis that are products.
    std::optional<StringRef> shape = reductionLoopShapeRefusal(*L);
    unsigned trip = SE.getSmallConstantTripCount(L);
    for (PHINode &phi : L->getHeader()->phis()) {
      if (fpKindFromType(phi.getType()) == FPKind::Invalid)
        continue;
      Expected<ReductionCycle> rc = matchReductionCycle(L, phi, latch);
      if (!rc) {
        consumeError(rc.takeError());
        continue;
      }
      Expected<ScalarLoopHandle> h = scalarLoopHandle(L, *rc, SE, DL, trip);
      if (!h) {
        reject(toString(h.takeError()));
        continue;
      }
      if (shape) {
        reject(*shape);
        continue;
      }
      if (trip == 0) {
        reject("trip count is not a known constant");
        continue;
      }
      if (!isPositiveZeroSeed(
              phi.getIncomingValueForBlock(L->getLoopPreheader()))) {
        reject("the accumulator does not start at +0.0");
        continue;
      }
      cb(L, *h, fpKindFromType(phi.getType()), trip);
    }
  });
}

void collectScalarLoopReductionTrips(
    const Options &O, Function &F,
    SmallVectorImpl<std::pair<size_t, unsigned>> &trips) {
  Module &M = *F.getParent();
  if (!O.Candidates.RaiseWMMA || !Target::of(M).hasInKernelMatrixRaise())
    return;
  if (F.isDeclaration())
    return;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  TargetLibraryInfo TLI(TLII, &F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  forEachScalarLoopReduction(
      O, F, SE, LI, [&](Loop *, ScalarLoopHandle &h, FPKind, unsigned trip) {
        if (std::optional<size_t> slot = slotOf(h.fma))
          trips.emplace_back(*slot, trip);
      });
  if (O.Diagnostics.Print && !trips.empty())
    llvm::errs() << "[poseidon] recorded " << trips.size()
                 << " reduction trip(s) for " << F.getName() << '\n';
}

// The guards a thread passes on its way to `preheader`, which it reaches at
// most once: each block from the entry that `preheader` does not post-dominate
// branches on a thread-index bound, and only the edge where the bound holds
// leads there. Every thread that passes them reaches `preheader`. None when
// some thread skips it for another reason.
static std::optional<ThreadIndexBounds>
guardsBefore(Function &F, BasicBlock *preheader, const DominatorTree &DT,
             const PostDominatorTree &PDT, const LoopInfo &LI) {
  if (LI.getLoopFor(preheader))
    return std::nullopt;
  ThreadIndexBounds bound;
  BasicBlock *X = &F.getEntryBlock();
  while (!PDT.dominates(preheader, X)) {
    Value *cond = nullptr;
    BasicBlock *onTrue = nullptr, *onFalse = nullptr;
    if (LI.getLoopFor(X) ||
        !PatternMatch::match(
            X->getTerminator(),
            PatternMatch::m_Br(PatternMatch::m_Value(cond),
                               PatternMatch::m_BasicBlock(onTrue),
                               PatternMatch::m_BasicBlock(onFalse))))
      return std::nullopt;
    const bool viaTrue = DT.dominates(BasicBlockEdge(X, onTrue), preheader);
    const bool viaFalse = DT.dominates(BasicBlockEdge(X, onFalse), preheader);
    if (viaTrue == viaFalse ||
        !collectThreadIndexBounds(cond, /*holds=*/viaTrue, bound))
      return std::nullopt;
    X = viaTrue ? onTrue : onFalse;
  }
  return bound;
}

void findScalarLoopMatmuls(const Options &O, Function &F, ScalarEvolution &SE,
                           LoopInfo &LI,
                           const FunctionProfileHeader &profileHeader,
                           SmallVectorImpl<AbstractMatmul> &out) {
  assert(O.Candidates.RaiseWMMA && "-poseidon-raise-wmma must be enabled");
  assert(Target::of(F).hasInKernelMatrixRaise() &&
         "findScalarLoopMatmuls expects GPU (NVPTX) target");

  // Without a profile header the launch geometry (and thus M, N) is unknown;
  // launchCount == 0 means the function was never exercised, so refuse to
  // guess.
  if (profileHeader.launchCount == 0) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << F.getName()
                   << ": no profile header (launchCount=0); skipping all "
                      "scalar-loop matmul candidates\n";
    return;
  }

  unsigned nextId = static_cast<unsigned>(out.size());

  // Profile-scale reduction lengths recorded at profile generation, keyed by
  // the fma's probe index.
  const std::map<size_t, unsigned> &redTrip = profileHeader.redTrip;
  DominatorTree DT(F);
  PostDominatorTree PDT(F);

  forEachScalarLoopReduction(
      O, F, SE, LI,
      [&](Loop *L, ScalarLoopHandle &h, FPKind eltKind, unsigned trip) {
        unsigned profM = extentForIndex(O, profileHeader.maxBlockDim, h.aRow,
                                        "A-row", F.getName());
        unsigned profN = extentForIndex(O, profileHeader.maxBlockDim, h.bCol,
                                        "B-col", F.getName());
        if (profM == 0 || profN == 0) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[poseidon] " << F.getName()
                         << ": profile header reports zero extent for matmul's "
                            "row/col tid axis; skipping\n";
          return;
        }

        // Profile-confirmed 2D block: the materializer skips tid.z / ntid.z.
        h.is2DBlock = (profileHeader.maxBlockDim[2] == 1);
        for (unsigned a = 0; a < 3; ++a)
          h.profiledBlock[a] = profileHeader.maxBlockDim[a];
        // Threads per CTA: the materializer needs it to tell whether the last
        // warp is partially populated (wmma is undefined there).
        h.blockThreads = profileHeader.maxBlockDim[0] *
                         profileHeader.maxBlockDim[1] *
                         profileHeader.maxBlockDim[2];
        // The in-kernel raise puts block barriers and warp-wide tensor-core
        // calls in the preheader, so every thread of a block that raises has
        // to execute it exactly once. A thread that passes the loop's guards
        // does, and a block raises only when all of its threads pass them.
        std::optional<ThreadIndexBounds> guard =
            guardsBefore(F, h.preheader, DT, PDT, LI);
        const bool wholeBlock = guard.has_value();
        if (wholeBlock)
          h.loopGuard = *guard;
        else if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] " << F.getName()
                       << ": not every thread of the block reaches the "
                          "reduction loop once, so it has no in-kernel raise\n";
        // A launch of another block shape than the profiled one runs the
        // loop, so the raise joins the loop's exit, which has to take nothing
        // from the loop but the product.
        const bool exitTakesProduct =
            h.exitBB->getSinglePredecessor() == L->getLoopLatch() &&
            llvm::all_of(h.exitBB->phis(), [&](PHINode &phi) {
              return phi.getIncomingValue(0) == h.fma;
            });
        if (wholeBlock && !exitTakesProduct && O.Diagnostics.Print)
          llvm::errs() << "[poseidon] " << F.getName()
                       << ": the reduction loop's exit takes more from it than "
                          "the product, so it has no in-kernel raise\n";
        h.inKernelRaisable = wholeBlock && exitTakesProduct;

        AbstractMatmul am;
        am.id = nextId++;
        am.M = profM;
        am.N = profN;
        am.K = trip;
        // gridDim x blockDim along the row/col axes is the full GEMM output,
        // the product of gridDim the CTA count; 0 when the profile has no
        // grid dimensions.
        {
          unsigned gM = dimForAxis(profileHeader.maxGridDim, h.aRow.fast);
          unsigned gN = dimForAxis(profileHeader.maxGridDim, h.bCol.fast);
          am.globalM = gM * profM;
          am.globalN = gN * profN;
          // A fused index lives entirely inside the CTA, so a grid axis sharing
          // its fast thread axis is a batch dimension, not an extension of this
          // index; report the global shape as unknown (0) rather than fabricate
          // one.
          if (h.aRow.fused() || h.bCol.fused()) {
            am.globalM = 0;
            am.globalN = 0;
          }
          uint32_t gx = profileHeader.maxGridDim[0],
                   gy = profileHeader.maxGridDim[1],
                   gz = profileHeader.maxGridDim[2];
          am.gridCTAs = gx ? gx * (gy ? gy : 1u) * (gz ? gz : 1u) : 0u;
        }
        am.elemType = eltKind;
        am.origin = AbstractMatmul::Origin::ScalarLoopReduction;
        am.scalarLoop = h;
        am.outputValue = h.fma;
        // Profile-scale reduction length; 0 when the header has none.
        {
          if (std::optional<size_t> slot = slotOf(h.fma)) {
            auto it = redTrip.find(*slot);
            am.globalK = it == redTrip.end() ? 0u : it->second;
          }
        }
        for (BasicBlock *BB : L->blocks())
          for (Instruction &I : *BB)
            am.footprint.insert(&I);

        out.push_back(std::move(am));
      });
}

} // namespace poseidon
