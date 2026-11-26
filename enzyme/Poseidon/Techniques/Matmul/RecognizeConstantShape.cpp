//===- RecognizeConstantShape.cpp - products in constant-trip reductions --===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Internal.h"
#include "Model/FPFormat.h"
#include "Profile/ProfileReader.h"
#include "Sites/Canonicalize.h"
#include "Sites/LaunchManifest.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Support/Utils.h"
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
#include "llvm/IR/InstIterator.h"
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
using namespace llvm::PatternMatch;

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

  // A fused index steps the operand by `unitByte` per unit of the fused index;
  // that must be the stride the single-axis path uses, or the block-base
  // back-out (start - idx*stride) lands off the tile.
  if (aRow.fused() && aRowUnit != aLdByte)
    return refuse("A's fused row index does not step by A's leading "
                  "dimension");
  // B's leading dimension is its step along the reduction when B is
  // K-strided (row-major), whatever row type its address is computed in, and
  // its per-column step when B is K-contiguous (col-major).
  bool bIsRowMajor = (bStrideByte != srcByteSize);
  int64_t bLdByte = bIsRowMajor ? bStrideByte
                    : bColUnit > 0
                        ? bColUnit
                        : extractLeadingDimByte(DL, bLoad, srcByteSize * trip);
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
  if (!O.Candidates.EnableMatmul || !Target::of(M).hasInKernelMatrixRaise())
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

// Strip LCSSA / single-value phis and casts from a stored value, testing
// whether it is `target`.
static bool valueReaches(const Value *v, const Value *target) {
  for (int guard = 0; v && guard < 16; ++guard) {
    if (v == target)
      return true;
    if (auto *P = dyn_cast<PHINode>(v)) {
      const Value *same = nullptr;
      bool ok = true;
      for (const Value *in : P->incoming_values()) {
        if (in == P)
          continue;
        if (!same)
          same = in;
        else if (same != in) {
          ok = false;
          break;
        }
      }
      if (ok && same) {
        v = same;
        continue;
      }
      return false;
    }
    if (auto *C = dyn_cast<CastInst>(v)) {
      v = C->getOperand(0);
      continue;
    }
    return false;
  }
  return false;
}

// The exclusive upper bound that the conditions gating the C store (outside the
// reduction loop) put on the global thread index of each axis, 0 for an axis
// they leave unbounded; none when one of them is not such a bound.
static std::optional<ThreadIndexBounds>
storeGuardBounds(Function &F, const RecognizedMatmul &m,
                 const StoreInst *cStore) {
  DominatorTree DT(F);
  const BasicBlock *SB = cStore->getParent();
  ThreadIndexBounds bound;
  for (BasicBlock &BB : F) {
    if (m.scalarLoop.blocks.count(&BB))
      continue;
    Instruction *T = BB.getTerminator();
    if (T->getNumSuccessors() < 2)
      continue;
    for (unsigned i = 0, e = T->getNumSuccessors(); i < e; ++i) {
      // The edge query rejects a duplicated edge on its own, so a two-way
      // branch to one block dominates nothing and contributes no fact.
      if (!DT.dominates(BasicBlockEdge(&BB, T->getSuccessor(i)), SB))
        continue;
      Value *Cond = nullptr;
      if (!match(T, m_Br(m_Value(Cond), m_BasicBlock(), m_BasicBlock())) ||
          !collectThreadIndexBounds(Cond, /*holds=*/i == 0, bound))
        return std::nullopt;
    }
  }
  return bound;
}

// Whether `addr` is its base pointer when every thread and block index is 0.
static bool startsAtBase(ScalarEvolution &SE, const SCEV *addr) {
  const SCEV *offset = SE.getMinusSCEV(addr, SE.getPointerBase(addr));
  struct LaunchIndices {
    explicit LaunchIndices(ScalarEvolution &SE) : SE(SE) {}
    ScalarEvolution &SE;
    ValueToSCEVMapTy atZero;
    bool follow(const SCEV *S) {
      if (const auto *U = dyn_cast<SCEVUnknown>(S)) {
        LaunchValue kind = launchGeometryOf(U->getValue()).Kind;
        if (kind == LaunchValue::ThreadIndex || kind == LaunchValue::BlockIndex)
          atZero[U->getValue()] = SE.getZero(U->getType());
      }
      return true;
    }
    bool isDone() const { return false; }
  } indices(SE);
  visitAll(offset, indices);
  return SCEVParameterRewriter::rewrite(offset, SE, indices.atZero)->isZero();
}

StoreInst *productStore(Function &F, const RecognizedMatmul &m) {
  for (Instruction &I : instructions(F))
    if (auto *S = dyn_cast<StoreInst>(&I);
        S && tracePointerRoot(S->getPointerOperand()).param >= 0 &&
        valueReaches(S->getValueOperand(), m.outputValue))
      return S;
  return nullptr;
}

std::optional<OutputExtents>
constantOutputExtents(const RecognizedMatmul &m, const ThreadIndexBounds &guard,
                      unsigned axM, unsigned axN) {
  if (!m.globalM || !m.globalN)
    return std::nullopt;
  const uint64_t K = m.K, profileK = m.globalK;
  if (!profileK)
    return OutputExtents{m.globalM, m.globalN, false};
  // Rescaling by K / globalK is sound only for an extent that tracks matrix
  // order (equal to globalK at profile scale): rescaling any other fabricates
  // an extent, and a library call would read past its operand.
  if (K == profileK || (m.globalM == profileK && m.globalN == profileK))
    return OutputExtents{(unsigned)(m.globalM * K / profileK),
                         (unsigned)(m.globalN * K / profileK), false};
  if (guard.ub[axM] && guard.ub[axN])
    return OutputExtents{(unsigned)guard.ub[axM], (unsigned)guard.ub[axN],
                         true};
  return std::nullopt;
}

bool hasEffect(const Instruction &I) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&I);
      II && II->isAssumeLikeIntrinsic())
    return false;
  return I.mayHaveSideEffects();
}

std::optional<std::string> libraryElementRefusal(FPKind elem) {
  if (elem == FPKind::FP64)
    return std::nullopt;
  return "the product's elements are " + std::string(fpKindName(elem)) +
         ", and every library-call entry is a GEMM of doubles";
}

Expected<std::optional<GemmBodyNote>>
constantShapeLibraryNote(const Options &O, Function &F,
                         const RecognizedMatmul &m) {
  assert(m.origin == RecognizedMatmul::Origin::ConstantShape &&
         "a constant-shape product");
  auto refuse = [&](const Twine &why) -> std::optional<GemmBodyNote> {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] " << F.getName() << ": " << why << '\n';
    return std::nullopt;
  };
  GemmBodyNote note;

  // The raiser reports m.M/m.N as the per-block tile and m.K as the K-loop
  // trip, which for a full-reduction square GEMM is the global dim, so the
  // dispatch dimension is m.K. Require FP64, A contiguous along K, and B either
  // row-major K-strided by N or col-major (K-contiguous).
  const ScalarLoopHandle &h = m.scalarLoop;
  if (std::optional<std::string> why = libraryElementRefusal(m.elemType))
    return refuse(*why);
  int64_t K = (int64_t)m.K;
  if (m.K == ~0u || K <= 0)
    return refuse("the reduction length is not known");
  if (h.aStrideByte != 8)
    return refuse("A is not a row-major matrix of doubles");
  bool bColMajor;
  if (h.bStrideByte == 8)
    bColMajor = true;
  else if (h.bStrideByte == K * 8)
    bColMajor = false;
  else
    return refuse("B steps by " + Twine(h.bStrideByte) +
                  " bytes along the reduction, neither one double nor a row "
                  "of " +
                  Twine(K));
  std::optional<unsigned> axM = tidAxisIndex(h.aRow.fast);
  std::optional<unsigned> axN = tidAxisIndex(h.bCol.fast);
  if (!axM || !axN || *axM == *axN)
    return refuse("the product's rows and columns do not run along two thread "
                  "axes");

  LaunchRef aRoot = tracePointerRoot(m.scalarLoop.aPtr);
  LaunchRef bRoot = tracePointerRoot(m.scalarLoop.bPtr);
  note.aParam = aRoot.param;
  note.aOffset = aRoot.offset;
  note.bParam = bRoot.param;
  note.bOffset = bRoot.offset;
  if (note.aParam < 0 || note.bParam < 0)
    return refuse("an operand is not read through a pointer parameter");

  int storeCount = 0;
  StoreInst *lastArgStore = nullptr;
  for (Instruction &I : instructions(F))
    if (auto *S = dyn_cast<StoreInst>(&I);
        S && tracePointerRoot(S->getPointerOperand()).param >= 0) {
      storeCount++;
      lastArgStore = S;
    }
  StoreInst *cStore = productStore(F, m);
  // A fused body's stored value reaches the matmul result only through the
  // loop's lcssa accumulator, which valueReaches may miss; with exactly one
  // argument-rooted store, that store is the C store.
  if (!cStore && storeCount == 1)
    cStore = lastArgStore;
  if (!cStore)
    return refuse("no store of the product into a pointer parameter");
  LaunchRef cRoot = tracePointerRoot(cStore->getPointerOperand());
  note.cParam = cRoot.param;
  note.cOffset = cRoot.offset;
  if (note.cParam < 0)
    return refuse("the product is not stored through a pointer parameter");

  // Library call supports only a pure GEMM body: a self-overwriting library
  // dispatch cannot represent a fused epilogue that may read its own output, so
  // the stored value (modulo lcssa) must be an instruction inside the loop.
  {
    const Value *sv = cStore->getValueOperand();
    while (const auto *phi = dyn_cast<PHINode>(sv)) {
      if (phi->getNumIncomingValues() != 1)
        break;
      sv = phi->getIncomingValue(0);
    }
    const auto *svI = dyn_cast<Instruction>(sv);
    if (!svI || !m.scalarLoop.blocks.count(svI->getParent()))
      return refuse("the value stored to C is not the product alone but an "
                    "expression of it computed after the loop");
  }

  std::optional<ThreadIndexBounds> guard = storeGuardBounds(F, m, cStore);
  if (!guard)
    return refuse("a condition gating the GEMM store is not a thread-index "
                  "bound");

  // Every dim must be deploy scale. Leading dims come from the handle (bytes
  // to elements).
  if (!m.globalM || !m.globalN)
    return refuse("the profile does not give the extent of the product's "
                  "rows and columns");
  std::optional<OutputExtents> extents =
      constantOutputExtents(m, *guard, *axM, *axN);
  if (!extents) {
    // A warning: the solver then picks among in-kernel candidates that may
    // be materially less accurate.
    diagnose(F,
             "no library call for " + F.getName() +
                 ": a non-square product at a changed deploy scale (profile "
                 "M/N/K = " +
                 Twine(m.globalM) + "/" + Twine(m.globalN) + "/" +
                 Twine(m.globalK) + ", deploy K = " + Twine(K) +
                 ") has no recoverable geometry; profile at the deploy shape",
             DS_Warning);
    if (O.Model.StrictMode)
      return userError(
          F, "strict mode: the library-call geometry of " + F.getName() +
                 " is unrecoverable for a non-square GEMM at a changed deploy "
                 "scale, so its candidate set would be reduced; profile at the "
                 "deploy shape, or leave out -poseidon-strict-mode");
    return std::nullopt;
  }
  const unsigned gM = extents->M, gNcols = extents->Ncols;
  if (extents->fromGuard) {
    // The host stub intersects the guard with the launch geometry,
    // min(crop, gridDim * blockDim).
    note.launchCrop = true;
    note.cropMAxis = (int)*axM;
    note.cropNAxis = (int)*axN;
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] recovered the deploy geometry of "
                   << F.getName() << ": M=" << gM << " Ncols=" << gNcols
                   << " K=" << K << " from the index guard on thread axes "
                   << *axM << '/' << *axN << '\n';
  }
  if (!note.launchCrop) {
    if (guard->ub[*axM] && guard->ub[*axM] < gM)
      return refuse("the GEMM store is gated by row < " +
                    Twine(guard->ub[*axM]) + ", which drops rows of the " +
                    Twine(gM) + " x " + Twine(gNcols) +
                    " product the library call writes");
    if (guard->ub[*axN] && guard->ub[*axN] < gNcols)
      return refuse("the GEMM store is gated by column < " +
                    Twine(guard->ub[*axN]) + ", which drops columns of the " +
                    Twine(gM) + " x " + Twine(gNcols) +
                    " product the library call writes");
  }

  // The call takes each operand from the start of its pointer parameter.
  const SCEV *cAddr = h.SE->getSCEV(cStore->getPointerOperand());
  if (!startsAtBase(*h.SE, h.aStartSCEV) ||
      !startsAtBase(*h.SE, h.bStartSCEV) || !startsAtBase(*h.SE, cAddr))
    return refuse("an operand of the product does not start where its "
                  "pointer parameter points");

  // C's layout and leading dimension from the byte step of its store address
  // along the row and column thread axes: one element along the column axis is
  // row-major, along the row axis column-major.
  const std::array<int64_t, 3> cStep = threadIndexCoefficients(cAddr);
  const unsigned axOther = 3 - *axM - *axN;
  constexpr int64_t elemBytes = 8;
  int64_t ldc = 0;
  if (cStep[axOther] == 0 && cStep[*axN] == elemBytes &&
      cStep[*axM] % elemBytes == 0 && cStep[*axM] / elemBytes >= gNcols) {
    note.cColMajor = false;
    ldc = cStep[*axM] / elemBytes;
  } else if (cStep[axOther] == 0 && cStep[*axM] == elemBytes &&
             cStep[*axN] % elemBytes == 0 && cStep[*axN] / elemBytes >= gM) {
    note.cColMajor = true;
    ldc = cStep[*axN] / elemBytes;
  } else {
    return refuse("the product's store address is not a row-major or "
                  "column-major element of C");
  }
  note.aColMajor = false;
  note.bColMajor = bColMajor;
  note.dims[DimM] = RuntimeDim::constant(gM);
  note.dims[DimNcols] = RuntimeDim::constant(gNcols);
  note.dims[DimK] = RuntimeDim::constant(K);
  note.dims[DimLda] =
      RuntimeDim::constant(h.aLeadingDimByte ? h.aLeadingDimByte / 8 : K);
  note.dims[DimLdb] =
      RuntimeDim::constant(h.bLeadingDimByte ? h.bLeadingDimByte / 8 : K);
  note.dims[DimLdc] = RuntimeDim::constant(ldc);
  if (llvm::any_of(note.dims,
                   [](const RuntimeDim &d) { return d.add > kMaxDimConstant; }))
    return refuse("a dimension of the product is out of the range of the "
                  "library call's int dimensions");
  // A library call takes the place of the launch of a body whose one effect
  // is the product's store; any other body's kernel is launched after it.
  note.standalone = llvm::all_of(instructions(F), [&](const Instruction &I) {
    return &I == cStore || !hasEffect(I);
  });
  // The kept launch runs after the call has written C, so it may not read C.
  if (!note.standalone && llvm::any_of(instructions(F), [&](Instruction &I) {
        auto *L = dyn_cast<LoadInst>(&I);
        return L &&
               tracePointerRoot(L->getPointerOperand()).param == note.cParam;
      }))
    return refuse("the body reads C, which the library call writes before "
                  "the kernel that does the body's other work runs");
  return note;
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

// What the loop `L` does besides accumulating the product `fma`: an effect, or
// a value other than the product that is used after the loop. None when it
// does nothing else.
static std::optional<StringRef> workBesidesProduct(const Loop &L,
                                                   const Instruction &fma) {
  for (const BasicBlock *BB : L.blocks())
    for (const Instruction &I : *BB) {
      if (hasEffect(I))
        return "the reduction loop has an effect besides the product";
      if (&I != &fma && llvm::any_of(I.users(), [&](const User *U) {
            return !L.contains(cast<Instruction>(U));
          }))
        return "the reduction loop computes a value besides the product that "
               "is used after it";
    }
  return std::nullopt;
}

void findConstantShapeMatmuls(const Options &O, Function &F,
                              ScalarEvolution &SE, LoopInfo &LI,
                              const SiteProfileHeader &profileHeader,
                              SmallVectorImpl<RecognizedMatmul> &out) {
  assert(O.Candidates.EnableMatmul &&
         "-poseidon-enable-matmul must be enabled");
  assert(Target::of(F).hasInKernelMatrixRaise() &&
         "findConstantShapeMatmuls expects GPU (NVPTX) target");

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

        for (unsigned a = 0; a < 3; ++a)
          h.profiledBlock[a] = profileHeader.maxBlockDim[a];
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
        // loop, so the raise joins the loop's exit, and a block that raises
        // runs nothing of the loop.
        const bool exitFromLatch =
            h.exitBB->getSinglePredecessor() == L->getLoopLatch();
        if (std::optional<StringRef> work = workBesidesProduct(*L, *h.fma))
          h.otherWork = work->str();
        if (wholeBlock && O.Diagnostics.Print) {
          if (!exitFromLatch)
            llvm::errs() << "[poseidon] " << F.getName()
                         << ": the reduction loop's exit is entered from "
                            "outside the loop, so it has no in-kernel raise\n";
          else if (!h.otherWork.empty())
            llvm::errs() << "[poseidon] " << F.getName() << ": " << h.otherWork
                         << ", so it has no in-kernel raise\n";
        }
        h.inKernelRaisable = wholeBlock && exitFromLatch && h.otherWork.empty();

        RecognizedMatmul am;
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
        am.origin = RecognizedMatmul::Origin::ConstantShape;
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
