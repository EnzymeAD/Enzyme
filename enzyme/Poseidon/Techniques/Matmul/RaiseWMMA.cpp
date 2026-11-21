//=- RaiseWMMA.cpp - Scalar-loop to WMMA raising for Poseidon -------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RaiseWMMA.h"
#include "Core/Options.h"
#include "IR/Utils.h"
#include "MatmulInternal.h"
#include "Model/FPFormat.h"
#include "Targets/NVPTXTensorCore.h"
#include "Targets/Target.h"
#include "WmmaUtils.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <map>
#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

std::optional<FMAMatch> matchFMA(Value *V) {
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

bool isClosedFmaPhiCycle(const Loop *L, Instruction *fma, PHINode *phi) {
  return hasOnlyInLoopUser(L, fma, phi) && hasOnlyInLoopUser(L, phi, fma);
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

static Value *gepRootBase(Value *ptr) {
  // TODO: interleaving
  while (auto *gep = dyn_cast<GEPOperator>(ptr))
    ptr = gep->getPointerOperand();
  while (auto *asc = dyn_cast<AddrSpaceCastOperator>(ptr))
    ptr = asc->getPointerOperand();
  return ptr;
}

struct TidLinearCoefs {
  int64_t coef[3] = {0, 0, 0};
};

static void extractTidCoefficients(const SCEV *S, int64_t mult,
                                   TidLinearCoefs &out) {
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
  out.coef[g.Axis] += mult;
}

static TidAxis classifyTidAxis(const SCEV *S) {
  TidLinearCoefs coefs;
  extractTidCoefficients(S, 1, coefs);
  unsigned n =
      (coefs.coef[0] != 0) + (coefs.coef[1] != 0) + (coefs.coef[2] != 0);
  if (n != 1)
    return TidAxis::Other;
  if (coefs.coef[0] != 0)
    return TidAxis::TidX;
  if (coefs.coef[1] != 0)
    return TidAxis::TidY;
  if (coefs.coef[2] != 0)
    return TidAxis::TidZ;
  llvm_unreachable("classifyTidAxis: n == 1 but all coefs are zero");
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

  TidLinearCoefs coefs;
  extractTidCoefficients(S, 1, coefs);
  unsigned n =
      (coefs.coef[0] != 0) + (coefs.coef[1] != 0) + (coefs.coef[2] != 0);
  static const TidAxis kAxis[3] = {TidAxis::TidX, TidAxis::TidY, TidAxis::TidZ};
  if (n <= 1) {
    // Single axis: still report the per-index byte step, which is the operand's
    // leading dimension.
    for (int i = 0; i < 3; ++i)
      if (coefs.coef[i] != 0 && coefs.coef[i] > 0)
        unitByte = coefs.coef[i];
    return classifyTidAxis(S);
  }
  if (n != 2)
    return TidAxis::Other;

  int lo = -1, hi = -1;
  for (int i = 0; i < 3; ++i) {
    if (coefs.coef[i] == 0)
      continue;
    if (coefs.coef[i] < 0)
      return TidAxis::Other; // negative index coefficient: not a tensor index
    if (lo < 0 || coefs.coef[i] < coefs.coef[lo])
      lo = i;
    if (hi < 0 || coefs.coef[i] > coefs.coef[hi])
      hi = i;
  }
  if (lo < 0 || hi < 0 || lo == hi)
    return TidAxis::Other; // equal coefficients: extents are not separable
  if (coefs.coef[hi] % coefs.coef[lo] != 0)
    return TidAxis::Other;

  slow = kAxis[hi];
  mult = (unsigned)(coefs.coef[hi] / coefs.coef[lo]);
  unitByte = coefs.coef[lo];
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

  Value *backedgeVal = phi.getIncomingValueForBlock(latch);
  if (!backedgeVal)
    return false;
  std::optional<FMAMatch> fm = matchFMA(backedgeVal);
  if (!fm || fm->acc != &phi)
    return false;

  // The fma/phi pair must form a closed in-loop cycle; otherwise the
  // per-iteration intermediate is live and the chain can't collapse to a
  // single wmma tile.
  if (!isClosedFmaPhiCycle(L, fm->fma, &phi))
    return false;

  LoadInst *aLoad = dyn_cast<LoadInst>(fm->a);
  LoadInst *bLoad = dyn_cast<LoadInst>(fm->b);
  if (!aLoad || !bLoad)
    return false;

  // TODO: handle mixed precision
  if (aLoad->getType() != phiTy || bLoad->getType() != phiTy)
    return false;

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
  // matmul B invariant (TODO incomplete)
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
  outHandle.fma = fm->fma;
  outHandle.fmul = fm->fmul;
  outHandle.aBase = gepRootBase(aLoad->getPointerOperand());
  outHandle.bBase = gepRootBase(bLoad->getPointerOperand());
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
  outHandle.tripCount = trip;
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
  SmallVector<Loop *, 8> worklist;
  for (Loop *L : LI)
    worklist.push_back(L);
  while (!worklist.empty()) {
    Loop *L = worklist.pop_back_val();
    for (Loop *Sub : L->getSubLoops())
      worklist.push_back(Sub);
    if (!L->getSubLoops().empty())
      continue;

    // Loop-level preconditions once; then each header phi is tested
    // independently so loops with several reductions yield one matmul each.
    BasicBlock *header = L->getHeader();
    BasicBlock *latch = L->getLoopLatch();
    if (!latch) {
      if (O.Diagnostics.Print)
        llvm::errs() << "[raise-wmma] reject loop @ "
                     << header->getParent()->getName()
                     << ": no unique latch (multi-latch / irreducible)\n";
      continue;
    }
    unsigned trip = SE.getSmallConstantMaxTripCount(L);
    if (trip == 0) {
      if (O.Diagnostics.Print)
        llvm::errs() << "[raise-wmma] reject loop in "
                     << header->getParent()->getName()
                     << ": trip count is not a known constant\n";
      continue;
    }

    for (PHINode &phi : header->phis()) {
      ScalarLoopHandle h;
      FPKind eltKind = FPKind::Invalid;
      if (!tryAnalyzeReductionPhi(L, phi, SE, DL, trip, latch, h, eltKind))
        continue;
      cb(L, h, eltKind, trip);
    }
  }
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

  forEachScalarLoopReduction(
      O, F, SE, LI,
      [&](Loop *L, ScalarLoopHandle &h, FPKind eltKind, unsigned) {
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

        AbstractMatmul am;
        am.id = nextId++;
        am.M = profM;
        am.N = profN;
        am.K = h.tripCount;
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
        am.aType = eltKind; // TODO: handle mixed precision
        am.bType = eltKind;
        am.accType = eltKind;
        am.dType = eltKind;
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
            am.footprint.insert(&I); // TODO: ponder over this

        out.push_back(std::move(am));
      });
}

namespace {
class ScalarLoopRaiseMaterializer {
public:
  ScalarLoopRaiseMaterializer(const AbstractMatmul &m,
                              const MatmulUnit::Option &t)
      : m(m), t(t) {}
  void run();

private:
  const AbstractMatmul &m;
  const MatmulUnit::Option &t;
};
} // namespace

void ScalarLoopRaiseMaterializer::run() {
  if (t.realization != Realization::InKernel ||
      t.variant.scheme != Scheme::Direct)
    report_fatal_error(
        "ScalarLoopRaiseMaterializer: only the in-kernel direct scheme is "
        "realized here");
  if (t.tileM == 0 || t.tileN == 0 || t.tileK == 0 || t.mChain == 0 ||
      t.nChain == 0 || t.kChain == 0)
    report_fatal_error("ScalarLoopRaiseMaterializer: zero tile/chain dim");
  if (t.mChain * t.tileM != m.M + t.padM ||
      t.nChain * t.tileN != m.N + t.padN || t.kChain * t.tileK != m.K + t.padK)
    report_fatal_error(
        "ScalarLoopRaiseMaterializer: chain*tile doesn't match source+padding");

  const ScalarLoopHandle &h = m.scalarLoop;
  BasicBlock *preheader = h.preheader;
  BasicBlock *exitBB = h.exitBB;
  if (!preheader)
    report_fatal_error(
        "ScalarLoopRaiseMaterializer: loop has no unique preheader");
  if (!exitBB)
    report_fatal_error(
        "ScalarLoopRaiseMaterializer: loop has no unique exit block");

  Function *F = preheader->getParent();
  Module *mod = F->getParent();
  LLVMContext &ctx = mod->getContext();
  const DataLayout &DL = mod->getDataLayout();
  Type *i32Ty = Type::getInt32Ty(ctx);
  PointerType *genPtrTy = PointerType::get(ctx, /*addrspace=*/0);

  Type *srcInputTy = llvmTypeForFPKind(ctx, m.aType);
  Type *srcAccTy = llvmTypeForFPKind(ctx, m.accType);
  Type *tgtInputTy = llvmTypeForFPKind(ctx, t.variant.operand);
  Type *tgtAccTy = llvmTypeForFPKind(ctx, t.variant.accumulator);
  int64_t srcInputByte =
      (int64_t)DL.getTypeAllocSize(srcInputTy).getFixedValue();
  int64_t tgtInputByte =
      (int64_t)DL.getTypeAllocSize(tgtInputTy).getFixedValue();
  int64_t tgtAccByte = (int64_t)DL.getTypeAllocSize(tgtAccTy).getFixedValue();

  // B layout from the captured reduction-axis stride: a one-element K stride
  // means K-contiguous (col-major), otherwise K-strided (row-major).
  // Classifying by contiguity handles arbitrary global leading dims.
  if (h.bStrideByte % srcInputByte != 0)
    report_fatal_error("ScalarLoopRaiseMaterializer: B reduction-axis stride " +
                       Twine(h.bStrideByte) +
                       " is not a multiple of the element size " +
                       Twine(srcInputByte));
  bool bRowMajor = (h.bStrideByte != srcInputByte);

  const bool needAccCvt = (m.accType != t.variant.accumulator);
  const unsigned mPad = t.mChain * t.tileM;
  const unsigned nPad = t.nChain * t.tileN;
  // A and B always route through scratch, which also handles a K-strided B;
  // for F64/F64 the fill is an identity copy. B is laid out row-major there.

  const std::string inputPrecStr = fpKindName(t.variant.operand);
  const std::string accPrecStr = fpKindName(t.variant.accumulator);
  switch (t.variant.operand) {
  case FPKind::F16:
  case FPKind::BF16:
  case FPKind::TF32:
  case FPKind::F64:
    break;
  default:
    report_fatal_error(
        "ScalarLoopRaiseMaterializer: unsupported input precision");
  }

  const nvptx::Wmma wmma = nvptx::declareWmma(mod, t.tileM, t.tileN, t.tileK,
                                              inputPrecStr, accPrecStr,
                                              /*bRowMajor=*/true);
  Function *loadAFn = wmma.LoadA;
  Function *loadBFn = wmma.LoadB;
  Function *mmaFn = wmma.Mma;
  Function *storeDFn = wmma.StoreD;

  Type *aFragRetTy = loadAFn->getFunctionType()->getReturnType();
  Type *bFragRetTy = loadBFn->getFunctionType()->getReturnType();
  FunctionType *mmaFTy = mmaFn->getFunctionType();
  Type *dFragRetTy = mmaFTy->getReturnType();

  auto fragSize = [](Type *T) -> unsigned {
    if (auto *ST = dyn_cast<StructType>(T))
      return ST->getNumElements();
    return 1;
  };
  unsigned aFragSize = fragSize(aFragRetTy);
  unsigned bFragSize = fragSize(bFragRetTy);
  unsigned dFragSize = fragSize(dFragRetTy);
  unsigned cFragSize = mmaFTy->getNumParams() - aFragSize - bFragSize;
  if (cFragSize != dFragSize)
    report_fatal_error("ScalarLoopRaiseMaterializer: mma C/D fragment shapes "
                       "differ; chained accumulation requires C precision == "
                       "D precision");
  Type *cFragElemTy = mmaFTy->getParamType(aFragSize + bFragSize);

  // Scratch keyed by (function, op, element type, extents) and not by matmul
  // id, so raises in one function that need an identically shaped buffer share
  // one (per-raise sets would exceed the 48 KB static cap). Safe because the
  // raised regions are sequential and bracketed by block barriers; the extents
  // in the key prevent reuse at another size. Names are materialized to
  // std::string because a Twine over temporaries would dangle.
  auto scratchName = [&](StringRef op, StringRef prec, uint64_t rows,
                         uint64_t cols) -> std::string {
    return ("__poseidon_raise_" + F->getName() + "_" + op + "_" + prec + "_" +
            Twine(rows) + "x" + Twine(cols))
        .str();
  };

  GlobalVariable *dScratch =
      getOrCreateSharedScratch(mod, scratchName("d", accPrecStr, mPad, nPad),
                               tgtAccTy, (uint64_t)mPad * nPad);
  // K-streaming: scratch holds one K-tile chunk of A and B, refilled per outer
  // iteration, so shared memory stays bounded regardless of K.
  GlobalVariable *aScratch = getOrCreateSharedScratch(
      mod, scratchName("a", inputPrecStr, mPad, t.tileK), tgtInputTy,
      (uint64_t)mPad * t.tileK);
  GlobalVariable *bScratch = getOrCreateSharedScratch(
      mod, scratchName("b", inputPrecStr, t.tileK, nPad), tgtInputTy,
      (uint64_t)t.tileK * nPad);

  IRBuilder<> B(preheader->getTerminator());

  const nvptx::Barrier barrier = nvptx::declareBarrier(mod);

  // Generic views of the source bases. Nothing reads them, since the fill
  // addresses through the expanded per-thread starts, but they are emitted.
  if (h.aBase->getType() != genPtrTy)
    B.CreateAddrSpaceCast(h.aBase, genPtrTy);
  if (h.bBase->getType() != genPtrTy)
    B.CreateAddrSpaceCast(h.bBase, genPtrTy);

  Type *i64Ty = Type::getInt64Ty(ctx);

  int64_t bKStrideByte = bRowMajor ? h.bStrideByte : srcInputByte;

  // TODO: clamp the cvt writes to (rowTid < M && colTid < N) for blocks larger
  // than M x N; for now we assume runtime block dims match the profile.
  Value *aBlockBase = nullptr, *bBlockBase = nullptr;
  {
    if (!h.aStartSCEV || !h.bStartSCEV || !h.SE)
      report_fatal_error("ScalarLoopRaiseMaterializer: missing aStartSCEV / "
                         "bStartSCEV / SE; analyzer didn't populate them");
#if LLVM_VERSION_MAJOR >= 22 && !defined(ENZYME_LLVM22_PRERELEASE)
    SCEVExpander sex(*h.SE, "poseidon.raise");
#else
    SCEVExpander sex(*h.SE, h.SE->getDataLayout(), "poseidon.raise");
#endif

    Value *rowTid =
        emitTidIndex(B, mod, h.aRowAxis, h.aRowSlowAxis, h.aRowFuseMult);
    Value *colTid =
        emitTidIndex(B, mod, h.bColAxis, h.bColSlowAxis, h.bColFuseMult);
    Value *rowTidI64 = B.CreateZExt(rowTid, i64Ty);
    Value *colTidI64 = B.CreateZExt(colTid, i64Ty);

    Value *aPerThreadStart =
        sex.expandCodeFor(h.aStartSCEV, genPtrTy, preheader->getTerminator());
    Value *bPerThreadStart =
        sex.expandCodeFor(h.bStartSCEV, genPtrTy, preheader->getTerminator());
    // A fused index must step the operand by exactly the stride the block-base
    // back-out assumes; the recognizer already checked this, so a mismatch
    // means the two derivations disagree. Abort rather than emit a wrong
    // address.
    if (h.aRowSlowAxis != TidAxis::Unknown &&
        h.aRowFuseUnitByte != h.aLeadingDimByte)
      report_fatal_error(
          "ScalarLoopRaiseMaterializer: fused A-row unit stride " +
          Twine(h.aRowFuseUnitByte) + " != A leading dim " +
          Twine(h.aLeadingDimByte));
    Value *aRowThreadOff =
        B.CreateMul(rowTidI64, ConstantInt::get(i64Ty, h.aLeadingDimByte));
    aBlockBase = B.CreateGEP(Type::getInt8Ty(ctx), aPerThreadStart,
                             B.CreateNeg(aRowThreadOff));
    int64_t bColThreadStrideByte = bRowMajor ? srcInputByte : h.bLeadingDimByte;
    if (h.bColSlowAxis != TidAxis::Unknown &&
        h.bColFuseUnitByte != bColThreadStrideByte)
      report_fatal_error(
          "ScalarLoopRaiseMaterializer: fused B-col unit stride " +
          Twine(h.bColFuseUnitByte) + " != B column stride " +
          Twine(bColThreadStrideByte));
    Value *bColThreadOff =
        B.CreateMul(colTidI64, ConstantInt::get(i64Ty, bColThreadStrideByte));
    bBlockBase = B.CreateGEP(Type::getInt8Ty(ctx), bPerThreadStart,
                             B.CreateNeg(bColThreadOff));
  }

  Value *aSrcBase = B.CreateAddrSpaceCast(aScratch, genPtrTy);
  Value *bSrcBase = B.CreateAddrSpaceCast(bScratch, genPtrTy);
  Value *dSrcBase = B.CreateAddrSpaceCast(dScratch, genPtrTy);

  // wmma load strides: the scratch tiles are tight-packed at (mPad x tileK) /
  // (tileK x nPad).
  int64_t aSrcRowByte = (int64_t)t.tileK * tgtInputByte;
  int64_t bSrcKStepByte = (int64_t)nPad * tgtInputByte;
  int64_t bSrcColStepByte = tgtInputByte;
  int64_t ldaElts = aSrcRowByte / tgtInputByte;
  int64_t ldbElts = bSrcKStepByte / tgtInputByte;
  int64_t ldcElts = nPad;
  Value *ldaV = ConstantInt::get(i32Ty, ldaElts);
  Value *ldbV = ConstantInt::get(i32Ty, ldbElts);
  Value *ldcV = ConstantInt::get(i32Ty, ldcElts);

  Constant *cZeroElt = Constant::getNullValue(cFragElemTy);
  Type *i8Ty = Type::getInt8Ty(ctx);

  // K-streaming outer loop: per k_outer, cooperatively refill the A/B scratch
  // with the next K-tile chunk, barrier, one wmma per (mt, nt) sub-tile with
  // the accumulator carried in PHIs, barrier; after the loop the accumulators
  // are stored to D-scratch. Padding is handled by the padded fill (exact zeros
  // outside the real extent); guard-masking the MMA would be wrong (the
  // fragments are warp-collective) and an uninitialized pad feeds 0 * Inf =
  // NaN.
  const bool padded = (t.padM > 0 || t.padN > 0 || t.padK > 0);

  BasicBlock *preheaderBB = B.GetInsertBlock();
  Instruction *postLoopInst = preheader->getTerminator();
  BasicBlock *afterLoopBB = preheaderBB->splitBasicBlock(
      postLoopInst->getIterator(), "wmma_kstream.after");
  preheaderBB->getTerminator()->eraseFromParent();

  BasicBlock *loopHdrBB =
      BasicBlock::Create(ctx, "wmma_kstream.hdr", F, afterLoopBB);
  BasicBlock *loopBodyBB =
      BasicBlock::Create(ctx, "wmma_kstream.body", F, afterLoopBB);

  IRBuilder<>(preheaderBB).CreateBr(loopHdrBB);

  // Header PHIs: k_outer plus one PHI per (mt, nt, frag-element) accumulator.
  IRBuilder<> hdrB(loopHdrBB);
  PHINode *kOuterPhi = hdrB.CreatePHI(i32Ty, 2, "k_outer");
  kOuterPhi->addIncoming(ConstantInt::get(i32Ty, 0), preheaderBB);

  unsigned numCFragPhis = t.mChain * t.nChain * cFragSize;
  SmallVector<PHINode *, 32> cFragPhis(numCFragPhis);
  for (unsigned i = 0; i < numCFragPhis; ++i) {
    cFragPhis[i] = hdrB.CreatePHI(cFragElemTy, 2, "cFrag");
    cFragPhis[i]->addIncoming(cZeroElt, preheaderBB);
  }

  Value *kLoopBound = ConstantInt::get(i32Ty, (int32_t)(t.kChain * t.tileK));
  Value *cond = hdrB.CreateICmpULT(kOuterPhi, kLoopBound);
  hdrB.CreateCondBr(cond, loopBodyBB, afterLoopBB);

  // Pre-install a placeholder branch back to the header so the fill helpers'
  // splitBasicBlock has a terminator to anchor; it migrates to the latch BB.
  IRBuilder<> bodyB(loopBodyBB);
  Instruction *placeholderBr = bodyB.CreateBr(loopHdrBB);
  bodyB.SetInsertPoint(placeholderBr);
  Value *kOuter64 = bodyB.CreateZExt(kOuterPhi, i64Ty);

  {
    auto [tlin, bsz] = emitThreadLinAndBlockSize(bodyB, mod, h.is2DBlock);

    // The per-iter fill covers the padded tile when padding is present, so the
    // single-shot form is only valid when that element count fits the block
    // (which the profile guarantees spans m.M x m.N).
    const uint64_t blockArea = (uint64_t)m.M * (uint64_t)m.N;
    const uint64_t aFillRows = padded ? (uint64_t)mPad : (uint64_t)m.M;
    const uint64_t aFillCols = (uint64_t)t.tileK;
    const uint64_t bFillRows = (uint64_t)t.tileK;
    const uint64_t bFillCols = padded ? (uint64_t)nPad : (uint64_t)m.N;
    const bool aFillSingleShot = (aFillRows * aFillCols) <= blockArea;
    const bool bFillSingleShot = (bFillRows * bFillCols) <= blockArea;

    // K-streaming makes the last K-tile short whenever padK > 0, so the valid
    // K count is the runtime min(tileK, K - k_outer).
    Value *validK = ConstantInt::get(i32Ty, (int32_t)t.tileK);
    if (t.padK > 0)
      validK = bodyB.CreateBinaryIntrinsic(
          Intrinsic::umin,
          bodyB.CreateSub(ConstantInt::get(i32Ty, (int32_t)m.K), kOuterPhi),
          ConstantInt::get(i32Ty, (int32_t)t.tileK));
    Value *validM = ConstantInt::get(i32Ty, (int32_t)m.M);
    Value *validN = ConstantInt::get(i32Ty, (int32_t)m.N);

    Value *aScratchGen = bodyB.CreateAddrSpaceCast(aScratch, genPtrTy);
    Value *aSrcForIter = bodyB.CreateGEP(
        i8Ty, aBlockBase,
        bodyB.CreateMul(kOuter64, ConstantInt::get(i64Ty, h.aStrideByte)));
    if (padded)
      emitCooperativeScratchFillPadded(
          bodyB, aScratchGen, tgtInputTy, t.variant.operand, aSrcForIter,
          srcInputTy, /*fillRows=*/aFillRows, /*fillCols=*/aFillCols,
          /*validRows=*/validM, /*validCols=*/validK,
          /*srcRowStrideByte=*/h.aLeadingDimByte,
          /*srcColStrideByte=*/h.aStrideByte,
          /*dstRowStrideByte=*/(int64_t)t.tileK * tgtInputByte,
          /*dstColStrideByte=*/tgtInputByte, tlin, bsz,
          /*assumeBlockCoversAll=*/aFillSingleShot);
    else
      emitCooperativeScratchFill(bodyB, aScratchGen, tgtInputTy,
                                 t.variant.operand, aSrcForIter, srcInputTy,
                                 /*numRows=*/m.M, /*numCols=*/(uint64_t)t.tileK,
                                 /*srcRowStrideByte=*/h.aLeadingDimByte,
                                 /*srcColStrideByte=*/h.aStrideByte,
                                 /*dstRowStrideByte=*/(int64_t)t.tileK *
                                     tgtInputByte,
                                 /*dstColStrideByte=*/tgtInputByte, tlin, bsz,
                                 /*assumeBlockCoversAll=*/aFillSingleShot);

    Value *bScratchGen = bodyB.CreateAddrSpaceCast(bScratch, genPtrTy);
    Value *bSrcForIter = bodyB.CreateGEP(
        i8Ty, bBlockBase,
        bodyB.CreateMul(kOuter64, ConstantInt::get(i64Ty, bKStrideByte)));
    int64_t bNStrideByte = bRowMajor ? srcInputByte : h.bLeadingDimByte;
    if (padded)
      emitCooperativeScratchFillPadded(
          bodyB, bScratchGen, tgtInputTy, t.variant.operand, bSrcForIter,
          srcInputTy, /*fillRows=*/bFillRows, /*fillCols=*/bFillCols,
          /*validRows=*/validK, /*validCols=*/validN,
          /*srcRowStrideByte=*/bKStrideByte,
          /*srcColStrideByte=*/bNStrideByte,
          /*dstRowStrideByte=*/(int64_t)nPad * tgtInputByte,
          /*dstColStrideByte=*/tgtInputByte, tlin, bsz,
          /*assumeBlockCoversAll=*/bFillSingleShot);
    else
      emitCooperativeScratchFill(bodyB, bScratchGen, tgtInputTy,
                                 t.variant.operand, bSrcForIter, srcInputTy,
                                 /*numRows=*/(uint64_t)t.tileK, /*numCols=*/m.N,
                                 /*srcRowStrideByte=*/bKStrideByte,
                                 /*srcColStrideByte=*/bNStrideByte,
                                 /*dstRowStrideByte=*/(int64_t)nPad *
                                     tgtInputByte,
                                 /*dstColStrideByte=*/tgtInputByte, tlin, bsz,
                                 /*assumeBlockCoversAll=*/bFillSingleShot);
    bodyB.CreateCall(barrier.Fn, barrier.Args);
  }

  // Warp-gate the WMMA chain when a single warp covers the per-block work
  // (shorter fragment live ranges in the other warps), and always when the
  // block's thread count is not a multiple of 32: warp-collective wmma ops in a
  // partially populated warp are undefined.
  const bool partialWarpBlock =
      (h.blockThreads != 0) && (h.blockThreads % nvptx::kWarpSize != 0);
  const bool useWarpGate = (t.mChain * t.nChain == 1) || partialWarpBlock;
  BasicBlock *wmmaMergeBB = nullptr;
  BasicBlock *bodyPreGateBB = nullptr;
  if (useWarpGate) {
    Value *bodyTlin = emitThreadLinAndBlockSize(bodyB, mod, h.is2DBlock).first;
    Value *warpId = bodyB.CreateLShr(
        bodyTlin, ConstantInt::get(i32Ty, nvptx::kLog2WarpSize), "warp_id");
    Value *isWarp0 =
        bodyB.CreateICmpEQ(warpId, ConstantInt::get(i32Ty, 0), "is_warp0");

    bodyPreGateBB = bodyB.GetInsertBlock();
    Instruction *splitPt = &*bodyB.GetInsertPoint();
    wmmaMergeBB = bodyPreGateBB->splitBasicBlock(splitPt->getIterator(),
                                                 "wmma_gate.merge");
    bodyPreGateBB->getTerminator()->eraseFromParent();
    BasicBlock *wmmaDoBB =
        BasicBlock::Create(ctx, "wmma_gate.do", F, wmmaMergeBB);
    IRBuilder<>(bodyPreGateBB).CreateCondBr(isWarp0, wmmaDoBB, wmmaMergeBB);
    bodyB.SetInsertPoint(wmmaDoBB);
  }

  SmallVector<Value *, 32> cFragLoop(numCFragPhis);
  for (unsigned mt = 0; mt < t.mChain; ++mt) {
    for (unsigned nt = 0; nt < t.nChain; ++nt) {
      Value *aRowBase = bodyB.CreatePtrAdd(
          aSrcBase,
          ConstantInt::get(i64Ty, (int64_t)mt * t.tileM * aSrcRowByte));
      Value *bColBase = bodyB.CreatePtrAdd(
          bSrcBase,
          ConstantInt::get(i64Ty, (int64_t)nt * t.tileN * bSrcColStepByte));

      // Scratch is refilled at offset 0 per iteration, so the wmma loads read
      // from the scratch base with no K offset.
      Value *aFragV = bodyB.CreateCall(loadAFn, {aRowBase, ldaV});
      Value *bFragV = bodyB.CreateCall(loadBFn, {bColBase, ldbV});

      SmallVector<Value *, 32> mmaArgs;
      mmaArgs.reserve(aFragSize + bFragSize + cFragSize);
      for (unsigned i = 0; i < aFragSize; ++i)
        mmaArgs.push_back(aFragSize == 1 ? aFragV
                                         : bodyB.CreateExtractValue(aFragV, i));
      for (unsigned i = 0; i < bFragSize; ++i)
        mmaArgs.push_back(bFragSize == 1 ? bFragV
                                         : bodyB.CreateExtractValue(bFragV, i));
      unsigned phiBase = (mt * t.nChain + nt) * cFragSize;
      for (unsigned i = 0; i < cFragSize; ++i)
        mmaArgs.push_back(cFragPhis[phiBase + i]);

      Value *dFragV = bodyB.CreateCall(mmaFn, mmaArgs);
      for (unsigned i = 0; i < dFragSize; ++i)
        cFragLoop[phiBase + i] =
            (dFragSize == 1) ? dFragV : bodyB.CreateExtractValue(dFragV, i);
    }
  }

  // Close the warp gate: merge the wmma result (warp 0) with the unchanged
  // accumulator (other warps) so the back edge uses the merged value.
  if (useWarpGate) {
    BasicBlock *wmmaEndBB = bodyB.GetInsertBlock();
    bodyB.CreateBr(wmmaMergeBB);

    IRBuilder<> mergeB =
        builderAt(wmmaMergeBB, wmmaMergeBB->getFirstInsertionPt());
    for (unsigned i = 0; i < numCFragPhis; ++i) {
      PHINode *phi = mergeB.CreatePHI(cFragElemTy, 2, "cFrag.merge");
      phi->addIncoming(cFragLoop[i], wmmaEndBB);
      phi->addIncoming(cFragPhis[i], bodyPreGateBB);
      cFragLoop[i] = phi;
    }
    setInsertPoint(bodyB, wmmaMergeBB, wmmaMergeBB->getFirstInsertionPt());
  }

  // Barrier after the wmma reads so the next iter's fill doesn't race them.
  bodyB.CreateCall(barrier.Fn, barrier.Args);

  // Latch: k_next = k_outer + tileK. The placeholder branch is already the BB's
  // terminator; the latch is whatever BB it migrated into.
  Value *kNext =
      bodyB.CreateAdd(kOuterPhi, ConstantInt::get(i32Ty, (int32_t)t.tileK));
  BasicBlock *latchBB = placeholderBr->getParent();

  kOuterPhi->addIncoming(kNext, latchBB);
  for (unsigned i = 0; i < numCFragPhis; ++i)
    cFragPhis[i]->addIncoming(cFragLoop[i], latchBB);

  // Store the final accumulators to D-scratch, warp-gated too: only warp 0's
  // PHIs hold the result, and the barrier after makes the store visible to the
  // readback in exitBB.
  IRBuilder<> afterB = builderAt(afterLoopBB, afterLoopBB->getFirstNonPHIIt());
  BasicBlock *afterStoreMergeBB = nullptr;
  if (useWarpGate) {
    Value *afterTlin =
        emitThreadLinAndBlockSize(afterB, mod, h.is2DBlock).first;
    Value *afterWarpId = afterB.CreateLShr(
        afterTlin, ConstantInt::get(i32Ty, nvptx::kLog2WarpSize), "warp_id");
    Value *afterIsWarp0 = afterB.CreateICmpEQ(
        afterWarpId, ConstantInt::get(i32Ty, 0), "is_warp0");
    Instruction *splitPt = &*afterB.GetInsertPoint();
    BasicBlock *curr = afterB.GetInsertBlock();
    afterStoreMergeBB =
        curr->splitBasicBlock(splitPt->getIterator(), "wmma_store_gate.merge");
    curr->getTerminator()->eraseFromParent();
    BasicBlock *storeDoBB =
        BasicBlock::Create(ctx, "wmma_store_gate.do", F, afterStoreMergeBB);
    IRBuilder<>(curr).CreateCondBr(afterIsWarp0, storeDoBB, afterStoreMergeBB);
    afterB.SetInsertPoint(storeDoBB);
  }

  for (unsigned mt = 0; mt < t.mChain; ++mt) {
    for (unsigned nt = 0; nt < t.nChain; ++nt) {
      Value *dTileByte =
          ConstantInt::get(i64Ty, ((int64_t)mt * t.tileM * (int64_t)nPad +
                                   (int64_t)nt * t.tileN) *
                                      tgtAccByte);
      Value *dTilePtr = afterB.CreatePtrAdd(dSrcBase, dTileByte);

      SmallVector<Value *, 16> storeArgs;
      storeArgs.reserve(2 + dFragSize);
      storeArgs.push_back(dTilePtr);
      unsigned phiBase = (mt * t.nChain + nt) * cFragSize;
      for (unsigned i = 0; i < dFragSize; ++i)
        storeArgs.push_back(cFragPhis[phiBase + i]);
      storeArgs.push_back(ldcV);
      afterB.CreateCall(storeDFn, storeArgs);
    }
  }

  if (useWarpGate) {
    afterB.CreateBr(afterStoreMergeBB);
    setInsertPoint(afterB, afterStoreMergeBB,
                   afterStoreMergeBB->getFirstInsertionPt());
  }
  afterB.CreateCall(barrier.Fn, barrier.Args);

  B.SetInsertPoint(&*afterLoopBB->getFirstInsertionPt());

  auto insertIt = exitBB->getFirstNonPHIIt();
  IRBuilder<> RB = builderAt(exitBB, insertIt);

  // TODO: clamp readback to (tid_row < M && tid_col < N) for blocks larger than
  // the M x N output tile. The dScratch row stride is N_pad.
  Value *rowTid =
      emitTidIndex(RB, mod, h.aRowAxis, h.aRowSlowAxis, h.aRowFuseMult);
  Value *colTid =
      emitTidIndex(RB, mod, h.bColAxis, h.bColSlowAxis, h.bColFuseMult);
  Value *rowOff = RB.CreateMul(rowTid, ConstantInt::get(i32Ty, nPad));
  Value *cellIdx = RB.CreateAdd(rowOff, colTid);
  Value *dScratchGen = RB.CreateAddrSpaceCast(dScratch, genPtrTy);
  Value *cellPtr = RB.CreateGEP(tgtAccTy, dScratchGen, cellIdx);
  LoadInst *rawReadback = RB.CreateLoad(tgtAccTy, cellPtr);
  rawReadback->setAlignment(Align(tgtAccByte));
  Value *readback =
      needAccCvt ? emitFPCast(RB, rawReadback, srcAccTy) : (Value *)rawReadback;

  SmallVector<Use *, 8> usesToRewrite;
  for (Use &U : h.fma->uses()) {
    auto *userI = dyn_cast<Instruction>(U.getUser());
    if (!userI || h.blocks.contains(userI->getParent()))
      continue;
    usesToRewrite.push_back(&U);
  }
  SmallVector<PHINode *, 4> phisToErase;
  for (Use *U : usesToRewrite) {
    auto *userI = cast<Instruction>(U->getUser());
    if (auto *phi = dyn_cast<PHINode>(userI)) {
      // LCSSA-style phi in the exit block: setting its incoming value to
      // readback (defined inside exitBB) would violate dominance, so collapse
      // the phi instead.
      if (phi->getParent() == exitBB) {
        phi->replaceAllUsesWith(readback);
        phisToErase.push_back(phi);
        continue;
      }
    }
    U->set(readback);
  }
  for (PHINode *phi : phisToErase)
    phi->eraseFromParent();
  // TODO: delete the now-dead reduction loop. Its body remains as dead code;
  // downstream DCE may clean it up but we don't drive that here (would need
  // DT/LI/SE plumbed through).
}

void materializeScalarLoopRaise(const AbstractMatmul &m,
                                const MatmulUnit::Option &opt) {
  ScalarLoopRaiseMaterializer(m, opt).run();
}

} // namespace poseidon
