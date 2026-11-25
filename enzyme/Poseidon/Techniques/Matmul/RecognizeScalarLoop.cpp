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
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <array>
#include <map>
#include <optional>

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

std::optional<ReductionCycle> matchReductionCycle(const Loop *L, PHINode &phi,
                                                  BasicBlock *latch,
                                                  StringRef *why) {
  auto fail = [&](StringRef reason) -> std::optional<ReductionCycle> {
    if (why)
      *why = reason;
    return std::nullopt;
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

StringRef reductionLoopShapeRefusal(const Loop &L) {
  BasicBlock *latch = L.getLoopLatch();
  if (!latch)
    return "the loop has no unique latch";
  if (L.getExitingBlock() != latch)
    return "the loop exits from a block other than its latch";
  if (!L.getExitBlock())
    return "the loop has no unique exit block";
  if (!L.getLoopPreheader())
    return "the loop has no preheader";
  return {};
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

// Fused-index classification: also accepts a matrix index built from exactly
// two thread axes, byteOffset = unit * tid[fast] + unit * mult * tid[slow]
// (sum-factorized FEM contractions). Returns the fast axis and, on a fused
// match, sets `slow`, `mult` and `unitByte`; rejects three axes, a
// non-divisible ratio, or equal coefficients.
static TidAxis classifyTidIndex(const SCEV *S, TidAxis &slow, unsigned &mult,
                                int64_t &unitByte) {
  slow = TidAxis::Unknown;
  mult = 0;
  unitByte = 0;

  std::array<int64_t, 3> coef = threadIndexCoefficients(S);
  unsigned n = (coef[0] != 0) + (coef[1] != 0) + (coef[2] != 0);
  static const TidAxis kAxis[3] = {TidAxis::TidX, TidAxis::TidY, TidAxis::TidZ};
  if (n == 1) {
    // Single axis: still report the per-index byte step, which is the operand's
    // leading dimension.
    int i = coef[0] ? 0 : coef[1] ? 1 : 2;
    if (coef[i] > 0)
      unitByte = coef[i];
    return kAxis[i];
  }
  if (n != 2)
    return TidAxis::Other;

  int lo = -1, hi = -1;
  for (int i = 0; i < 3; ++i) {
    if (coef[i] == 0)
      continue;
    if (coef[i] < 0)
      return TidAxis::Other; // negative index coefficient: not a tensor index
    if (lo < 0 || coef[i] < coef[lo])
      lo = i;
    if (hi < 0 || coef[i] > coef[hi])
      hi = i;
  }
  if (lo < 0 || hi < 0 || lo == hi)
    return TidAxis::Other; // equal coefficients: extents are not separable
  if (coef[hi] % coef[lo] != 0)
    return TidAxis::Other;

  slow = kAxis[hi];
  mult = (unsigned)(coef[hi] / coef[lo]);
  unitByte = coef[lo];
  return kAxis[lo];
}

// Extract a matmul handle from one header phi of an innermost loop; true if
// the phi+FMA chain is matmul-shaped (A K-stride, B K-stride, tid-axis
// invariants), false silently otherwise.
static bool tryAnalyzeReductionPhi(Loop *L, PHINode &phi, ScalarEvolution &SE,
                                   const DataLayout &DL, unsigned trip,
                                   BasicBlock *latch,
                                   ScalarLoopHandle &outHandle,
                                   FPKind &eltKind) {
  Type *phiTy = phi.getType();
  FPKind srcPrec = fpKindFromType(phiTy);
  if (srcPrec == FPKind::Invalid)
    return false;
  std::optional<ReductionCycle> rc = matchReductionCycle(L, phi, latch);
  if (!rc)
    return false;
  LoadInst *aLoad = rc->aLoad;
  LoadInst *bLoad = rc->bLoad;

  const SCEV *aAddr = SE.getSCEV(aLoad->getPointerOperand());
  const SCEV *bAddr = SE.getSCEV(bLoad->getPointerOperand());
  auto *aRec = dyn_cast<SCEVAddRecExpr>(aAddr);
  auto *bRec = dyn_cast<SCEVAddRecExpr>(bAddr);
  if (!aRec || !bRec)
    return false;
  if (aRec->getLoop() != L || bRec->getLoop() != L)
    return false;
  auto *aStepC = dyn_cast<SCEVConstant>(aRec->getStepRecurrence(SE));
  auto *bStepC = dyn_cast<SCEVConstant>(bRec->getStepRecurrence(SE));
  if (!aStepC || !bStepC)
    return false;
  int64_t aStrideByte = aStepC->getAPInt().getSExtValue();
  int64_t bStrideByte = bStepC->getAPInt().getSExtValue();

  // matmul A invariant
  int64_t srcByteSize = (int64_t)DL.getTypeAllocSize(phiTy).getFixedValue();
  if (aStrideByte != srcByteSize)
    return false;
  // matmul B invariant
  if (bStrideByte <= 0 || bStrideByte % srcByteSize != 0)
    return false;

  TidAxis aRowSlow = TidAxis::Unknown, bColSlow = TidAxis::Unknown;
  unsigned aRowMult = 0, bColMult = 0;
  int64_t aRowUnit = 0, bColUnit = 0;
  TidAxis aRowAxis =
      classifyTidIndex(aRec->getStart(), aRowSlow, aRowMult, aRowUnit);
  TidAxis bColAxis =
      classifyTidIndex(bRec->getStart(), bColSlow, bColMult, bColUnit);
  auto axisOk = [](TidAxis a) {
    return a == TidAxis::TidX || a == TidAxis::TidY || a == TidAxis::TidZ;
  };
  if (!axisOk(aRowAxis) || !axisOk(bColAxis))
    return false;
  // Axis disjointness is required only once a fused index is involved: sharing
  // a thread axis between the two matrix indices makes the fused numbering
  // ambiguous.
  if (aRowSlow != TidAxis::Unknown || bColSlow != TidAxis::Unknown) {
    if (aRowAxis == bColAxis)
      return false;
    if (aRowSlow != TidAxis::Unknown &&
        (!axisOk(aRowSlow) || aRowSlow == bColAxis || aRowSlow == bColSlow))
      return false;
    if (bColSlow != TidAxis::Unknown &&
        (!axisOk(bColSlow) || bColSlow == aRowAxis))
      return false;
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
  if (aRowSlow != TidAxis::Unknown && aRowUnit != aLdByte)
    return false;
  // B's per-column step is the element size when B is K-strided (row-major)
  // and its leading dimension when B is K-contiguous (col-major).
  bool bIsRowMajor = (bStrideByte != srcByteSize);
  if (!bIsRowMajor && bColUnit > 0)
    bLdByte = bColUnit;
  if (bColSlow != TidAxis::Unknown &&
      bColUnit != (bIsRowMajor ? srcByteSize : bLdByte))
    return false;

  outHandle.preheader = L->getLoopPreheader();
  outHandle.exitBB = L->getExitBlock();
  outHandle.blocks.clear();
  for (BasicBlock *BB : L->blocks())
    outHandle.blocks.insert(BB);
  outHandle.fma = rc->fma;
  outHandle.fmul = rc->fmul;
  outHandle.aPtr = aLoad->getPointerOperand();
  outHandle.bPtr = bLoad->getPointerOperand();
  outHandle.aStartSCEV = aRec->getStart();
  outHandle.bStartSCEV = bRec->getStart();
  outHandle.SE = &SE;
  outHandle.aRowAxis = aRowAxis;
  outHandle.bColAxis = bColAxis;
  outHandle.aRowSlowAxis = aRowSlow;
  outHandle.bColSlowAxis = bColSlow;
  outHandle.aRowFuseMult = aRowMult;
  outHandle.bColFuseMult = bColMult;
  outHandle.aRowFuseUnitByte = aRowUnit;
  outHandle.bColFuseUnitByte = bColUnit;
  outHandle.aStrideByte = aStrideByte;
  outHandle.bStrideByte = bStrideByte;
  outHandle.aLeadingDimByte = aLdByte;
  outHandle.bLeadingDimByte = bLdByte;
  eltKind = srcPrec;
  return true;
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
                               TidAxis fast, TidAxis slow, unsigned mult,
                               StringRef what, StringRef fname) {
  unsigned fastDim = dimForAxis(dims, fast);
  if (slow == TidAxis::Unknown)
    return fastDim;
  unsigned slowDim = dimForAxis(dims, slow);
  if (fastDim == 0 || slowDim == 0)
    return 0;
  if (mult != fastDim) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[raise-wmma] " << fname << ": fused " << what
                   << " index multiplier " << mult
                   << " does not match the profiled block extent " << fastDim
                   << " of its fast thread axis; the fused index is not a "
                      "dense numbering of the block's threads. Skipping "
                      "(refusing to guess a matmul dimension).\n";
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
      llvm::errs() << "[raise-wmma] reject loop in " << F.getName() << ": "
                   << why << '\n';
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
    StringRef shape = reductionLoopShapeRefusal(*L);
    unsigned trip = SE.getSmallConstantTripCount(L);
    for (PHINode &phi : L->getHeader()->phis()) {
      ScalarLoopHandle h;
      FPKind eltKind = FPKind::Invalid;
      if (!tryAnalyzeReductionPhi(L, phi, SE, DL, trip, latch, h, eltKind))
        continue;
      if (!shape.empty()) {
        reject(shape);
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
      cb(L, h, eltKind, trip);
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
        size_t idx;
        if (tryReadProfIdxMetadata(h.fma, idx))
          trips.emplace_back(idx, trip);
      });
  if (O.Diagnostics.Print && !trips.empty())
    llvm::errs() << "[raise-wmma] recorded " << trips.size()
                 << " reduction trip(s) for " << F.getName() << '\n';
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
      llvm::errs() << "[raise-wmma] " << F.getName()
                   << ": no profile header (launchCount=0); skipping all "
                      "scalar-loop matmul candidates\n";
    return;
  }

  unsigned nextId = static_cast<unsigned>(out.size());

  // Profile-scale reduction lengths recorded at profile generation, keyed by
  // the fma's probe index.
  const std::map<size_t, unsigned> &redTrip = profileHeader.redTrip;
  PostDominatorTree PDT(F);

  forEachScalarLoopReduction(
      O, F, SE, LI,
      [&](Loop *L, ScalarLoopHandle &h, FPKind eltKind, unsigned trip) {
        unsigned profM = extentForIndex(O, profileHeader.maxBlockDim,
                                        h.aRowAxis, h.aRowSlowAxis,
                                        h.aRowFuseMult, "A-row", F.getName());
        unsigned profN = extentForIndex(O, profileHeader.maxBlockDim,
                                        h.bColAxis, h.bColSlowAxis,
                                        h.bColFuseMult, "B-col", F.getName());
        if (profM == 0 || profN == 0) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[raise-wmma] " << F.getName()
                         << ": profile header reports zero extent for matmul's "
                            "row/col tid axis; skipping\n";
          return;
        }

        // Profile-confirmed 2D block: the materializer skips tid.z / ntid.z.
        h.is2DBlock = (profileHeader.maxBlockDim[2] == 1);
        // Threads per CTA: the materializer needs it to tell whether the last
        // warp is partially populated (wmma is undefined there).
        h.blockThreads = profileHeader.maxBlockDim[0] *
                         profileHeader.maxBlockDim[1] *
                         profileHeader.maxBlockDim[2];
        // The in-kernel raise puts block barriers and warp-wide tensor-core
        // calls in the preheader, so every thread of the block that runs F
        // has to execute it exactly once: it does when the preheader
        // post-dominates the entry and lies in no loop.
        h.reachedByWholeBlock = !LI.getLoopFor(h.preheader) &&
                                PDT.dominates(h.preheader, &F.getEntryBlock());
        if (!h.reachedByWholeBlock && O.Diagnostics.Print)
          llvm::errs() << "[raise-wmma] " << F.getName()
                       << ": not every thread of the block reaches the "
                          "reduction loop once\n";

        AbstractMatmul am;
        am.id = nextId++;
        am.M = profM;
        am.N = profN;
        am.K = trip;
        // gridDim x blockDim along the row/col axes is the full GEMM output,
        // the product of gridDim the CTA count; 0 when the profile has no
        // grid dimensions.
        {
          unsigned gM = dimForAxis(profileHeader.maxGridDim, h.aRowAxis);
          unsigned gN = dimForAxis(profileHeader.maxGridDim, h.bColAxis);
          am.globalM = gM * profM;
          am.globalN = gN * profN;
          // A fused index lives entirely inside the CTA, so a grid axis sharing
          // its fast thread axis is a batch dimension, not an extension of this
          // index; report the global shape as unknown (0) rather than fabricate
          // one.
          if (h.aRowSlowAxis != TidAxis::Unknown ||
              h.bColSlowAxis != TidAxis::Unknown) {
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
          size_t sidx;
          if (tryReadProfIdxMetadata(h.fma, sidx)) {
            auto it = redTrip.find(sidx);
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
