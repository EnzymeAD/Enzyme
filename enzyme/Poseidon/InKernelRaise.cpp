//=- InKernelRaise.cpp - Ozaki Scheme I matmul candidates for Poseidon ----=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ozaki Scheme I N-slice materialization: a cooperative Veltkamp-split fill
// into per-slice scratch, a K-streamed loop with an N(N+1)/2-mma chain per
// K-tile, and a readback combining sum_k d[k] / SCALE^k.
//===----------------------------------------------------------------------===//

#include "Flags.h"
#include "Optimize.h"

#include "InKernelRaise.h"
#include "WmmaUtils.h"
#include <functional>

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <cmath>

using namespace llvm;

namespace poseidon {

namespace {

enum class OzakiSliceKind { F16, BF16, TF32 };

OzakiSliceKind sliceFromFPKind(FPKind k) {
  switch (k) {
  case FPKind::F16:
    return OzakiSliceKind::F16;
  case FPKind::BF16:
    return OzakiSliceKind::BF16;
  case FPKind::TF32:
    return OzakiSliceKind::TF32;
  default:
    report_fatal_error(
        "OzakiOzaki: unsupported slice FPKind (must be F16/BF16/TF32)");
  }
}

const char *sliceIntrinTag(OzakiSliceKind k) {
  switch (k) {
  case OzakiSliceKind::F16:
    return "f16";
  case OzakiSliceKind::BF16:
    return "bf16";
  case OzakiSliceKind::TF32:
    return "tf32";
  }
  llvm_unreachable("unknown OzakiSliceKind");
}

Type *sliceScratchTy(LLVMContext &ctx, OzakiSliceKind k) {
  // The scratch element type matches the wmma fragment's element view; TF32
  // wmma loads bit-cast F32 storage into i32 fragments, so F32 is stored after
  // explicit f2tf32 quantization.
  switch (k) {
  case OzakiSliceKind::F16:
    return Type::getHalfTy(ctx);
  case OzakiSliceKind::BF16:
    return Type::getBFloatTy(ctx);
  case OzakiSliceKind::TF32:
    return Type::getFloatTy(ctx);
  }
  llvm_unreachable("unknown OzakiSliceKind");
}

// Veltkamp scale: residual after split rounds is bounded by ulp(hi)/2 ≈
// |hi|·2^-(mant+1), so scaling by 2^(mant+1) brings it into slice range. F16
// and TF32 both have 10 mantissa bits (2^11); BF16 uses 2^7 (matches the TCEC
// reference, not 2^8).
double veltkampScale(OzakiSliceKind k) {
  switch (k) {
  case OzakiSliceKind::F16:
    return 2048.0;
  case OzakiSliceKind::BF16:
    return 128.0;
  case OzakiSliceKind::TF32:
    return 2048.0;
  }
  llvm_unreachable("unknown OzakiSliceKind");
}

struct OzakiVariant {
  unsigned N;           // slice count
  OzakiSliceKind slice; // F16 / BF16 / TF32
  FPKind accumKind;     // F32 (mma accumulator precision)
  unsigned tileM, tileN, tileK;
};

OzakiVariant deriveOzakiVariant(const CandidateMatmul::Option &opt) {
  OzakiVariant v;
  v.N = opt.strategyParam;
  v.slice = sliceFromFPKind(opt.inputPrec);
  v.accumKind = opt.accPrec;
  v.tileM = opt.tileM;
  v.tileN = opt.tileN;
  v.tileK = opt.tileK;
  return v;
}

// Quantize an F32 value to slice precision (half for F16/BF16; float for TF32,
// bit-identical to F32 with the bottom 13 mantissa bits zeroed).
Value *emitSliceQuantize(IRBuilder<> &B, Module *mod, Value *vs_f32,
                         OzakiSliceKind slice) {
  LLVMContext &ctx = B.getContext();
  Type *f32Ty = Type::getFloatTy(ctx);
  Type *sliceTy = sliceScratchTy(ctx, slice);
  if (slice == OzakiSliceKind::F16 || slice == OzakiSliceKind::BF16)
    return B.CreateFPTrunc(vs_f32, sliceTy);
  // nvvm.f2tf32.rna gives an i32 bit pattern; bitcast to float for the shmem
  // store, which wmma load.a.tf32 reads as i32 again.
  Function *f2tf32 =
      Intrinsic::getOrInsertDeclaration(mod, Intrinsic::nvvm_f2tf32_rna);
  Value *q_i32 = B.CreateCall(f2tf32, {vs_f32});
  return B.CreateBitCast(q_i32, f32Ty);
}

// Slice value back to F32 for the next residual; TF32 is already F32 storage.
Value *emitSliceToF32(IRBuilder<> &B, Value *slice, OzakiSliceKind kind) {
  Type *f32Ty = Type::getFloatTy(B.getContext());
  if (kind == OzakiSliceKind::TF32)
    return slice; // bit-pattern already in float
  return B.CreateFPExt(slice, f32Ty);
}

// N-slice Veltkamp split:
//   s[0]  = quantize(v)
//   r[k]  = (r[k-1] - s[k-1]_f32) * SCALE ; s[k] = quantize(r[k])
// Reconstruction: v ≈ Σ_k s[k] / SCALE^k. Each slice captures ~mant_bits of v
// (10 for F16/TF32, 7 for BF16).
SmallVector<Value *, 8> emitNwayVeltkampSplit(IRBuilder<> &B, Module *mod,
                                              Value *vs_f32, unsigned N,
                                              OzakiSliceKind slice) {
  Type *f32Ty = Type::getFloatTy(B.getContext());
  Value *scale = ConstantFP::get(f32Ty, veltkampScale(slice));
  SmallVector<Value *, 8> slices;
  slices.reserve(N);
  Value *cur = vs_f32;
  for (unsigned k = 0; k < N; ++k) {
    Value *s_k = emitSliceQuantize(B, mod, cur, slice);
    slices.push_back(s_k);
    if (k + 1 < N) {
      Value *s_k_f32 = emitSliceToF32(B, s_k, slice);
      Value *resid = B.CreateFSub(cur, s_k_f32);
      cur = B.CreateFMul(resid, scale);
    }
  }
  return slices;
}

struct OzakiWmmaFns {
  Function *loadA;
  Function *loadB;
  Function *mma;
  Function *storeD;
};

OzakiWmmaFns resolveOzakiWmmaFns(Module *mod, const OzakiVariant &v,
                                 PointerType *genPtrTy) {
  std::string shape =
      ("m" + Twine(v.tileM) + "n" + Twine(v.tileN) + "k" + Twine(v.tileK))
          .str();
  const char *tag = sliceIntrinTag(v.slice);
  Intrinsic::ID loadAId = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                               ".load.a.row.stride." + tag);
  Intrinsic::ID loadBId = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                               ".load.b.row.stride." + tag);
  // NVPTX mma suffix encoding: F16 inputs name a C/D precision pair ("f32.f32"
  // for the F32 accumulator chain); BF16 and TF32 imply F32 and use one suffix.
  std::string mmaName;
  switch (v.slice) {
  case OzakiSliceKind::F16:
    mmaName = "llvm.nvvm.wmma." + shape + ".mma.row.row.f32.f32";
    break;
  case OzakiSliceKind::BF16:
    mmaName = "llvm.nvvm.wmma." + shape + ".mma.row.row.bf16";
    break;
  case OzakiSliceKind::TF32:
    mmaName = "llvm.nvvm.wmma." + shape + ".mma.row.row.tf32";
    break;
  }
  Intrinsic::ID mmaId = resolveWmmaIntrinsic(mmaName);
  Intrinsic::ID storeDId = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                                ".store.d.row.stride.f32");
  return {
      Intrinsic::getOrInsertDeclaration(mod, loadAId, {genPtrTy}),
      Intrinsic::getOrInsertDeclaration(mod, loadBId, {genPtrTy}),
      Intrinsic::getOrInsertDeclaration(mod, mmaId),
      Intrinsic::getOrInsertDeclaration(mod, storeDId, {genPtrTy}),
  };
}

class OzakiIRaiseMaterializer {
public:
  OzakiIRaiseMaterializer(const AbstractMatmul &m,
                          const CandidateMatmul::Option &t)
      : m(m), t(t) {}
  void run();

private:
  // FP (tcec) slice path: F16/BF16/TF32 Veltkamp split, F32 accumulator.
  void runTcec();

  const AbstractMatmul &m;
  const CandidateMatmul::Option &t;
};

} // namespace

void OzakiIRaiseMaterializer::run() {
  if (t.strategy != CandidateMatmul::Option::Strategy::OzakiI)
    report_fatal_error(
        "OzakiIRaiseMaterializer: only OzakiI strategy supported");
  if (t.strategyParam < 2 || t.strategyParam > 8)
    report_fatal_error("OzakiIRaiseMaterializer: N (strategyParam) must be in "
                       "[2, 8] (got " +
                       Twine(t.strategyParam) + ")");
  runTcec();
}

void OzakiIRaiseMaterializer::runTcec() {
  if (t.inputPrec != FPKind::F16 && t.inputPrec != FPKind::BF16 &&
      t.inputPrec != FPKind::TF32)
    report_fatal_error("OzakiIRaiseMaterializer: input slice must be F16, "
                       "BF16, or TF32 (got " +
                       Twine(fpKindName(t.inputPrec)) + ")");
  if (t.accPrec != FPKind::F32)
    report_fatal_error(
        "OzakiIRaiseMaterializer: accumulator must be F32 (got " +
        Twine(fpKindName(t.accPrec)) + ")");
  if (t.tileM == 0 || t.tileN == 0 || t.tileK == 0 || t.mChain == 0 ||
      t.nChain == 0 || t.kChain == 0)
    report_fatal_error("OzakiIRaiseMaterializer: zero tile/chain dim");
  if (t.mChain * t.tileM != m.M + t.padM ||
      t.nChain * t.tileN != m.N + t.padN || t.kChain * t.tileK != m.K + t.padK)
    report_fatal_error(
        "OzakiIRaiseMaterializer: chain*tile doesn't match source+padding");
  // padM/padN/padK are all supported: the fill writes the full padded tile
  // with an exact zero outside the real extent, and a zero slice contributes
  // exactly 0 to every cross product. Guard-masking the mma would be wrong
  // (the fragments are warp-collective), and an uninitialized pad feeds
  // 0 * Inf = NaN into real outputs.

  const OzakiVariant variant = deriveOzakiVariant(t);

  const ScalarLoopHandle &h = m.scalarLoop;
  BasicBlock *preheader = h.preheader;
  BasicBlock *exitBB = h.exitBB;
  if (!preheader)
    report_fatal_error("OzakiIRaiseMaterializer: loop has no unique preheader");
  if (!exitBB)
    report_fatal_error(
        "OzakiIRaiseMaterializer: loop has no unique exit block");

  Function *F = preheader->getParent();
  Module *mod = F->getParent();
  LLVMContext &ctx = mod->getContext();
  const DataLayout &DL = mod->getDataLayout();
  Type *i32Ty = Type::getInt32Ty(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *f32Ty = Type::getFloatTy(ctx);
  PointerType *genPtrTy = PointerType::get(ctx, /*addrspace=*/0);

  Type *srcInputTy = llvmTypeForFPKind(ctx, m.aType);
  Type *srcAccTy = llvmTypeForFPKind(ctx, m.accType);
  int64_t srcInputByte =
      (int64_t)DL.getTypeAllocSize(srcInputTy).getFixedValue();

  Type *sliceTy = sliceScratchTy(ctx, variant.slice);
  int64_t sliceByte = (int64_t)DL.getTypeAllocSize(sliceTy).getFixedValue();
  int64_t f32Byte = (int64_t)DL.getTypeAllocSize(f32Ty).getFixedValue();

  // B layout from the captured reduction-axis stride: a one-element K stride
  // means K-contiguous (col-major), otherwise K-strided (row-major). Ozaki
  // always routes B through scratch, so this only sets the fill's read strides.
  if (h.bStrideByte % srcInputByte != 0)
    report_fatal_error("OzakiIRaiseMaterializer: B reduction-axis stride " +
                       Twine(h.bStrideByte) +
                       " is not a multiple of element size " +
                       Twine(srcInputByte));
  bool bRowMajor = (h.bStrideByte != srcInputByte);

  // Padded extents (chain * tile); the A/B scratch K-dim is one tileK chunk
  // (K-streamed), the output scratch is the full mPad x nPad. No per-row/col
  // scaling: an unscaled Veltkamp split delivers F32-class accuracy on
  // F16-safe inputs.
  const unsigned mPad = t.mChain * t.tileM;
  const unsigned nPad = t.nChain * t.tileN;

  OzakiWmmaFns fns = resolveOzakiWmmaFns(mod, variant, genPtrTy);
  Type *aFragRetTy = fns.loadA->getFunctionType()->getReturnType();
  Type *bFragRetTy = fns.loadB->getFunctionType()->getReturnType();
  FunctionType *mmaFTy = fns.mma->getFunctionType();
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
    report_fatal_error("OzakiIRaiseMaterializer: mma C/D fragment shapes "
                       "differ — chained accumulation requires C precision == "
                       "D precision");
  Type *cFragElemTy = mmaFTy->getParamType(aFragSize + bFragSize);

  // Scratch: aSlice[k] (slice-typed, M_pad x tileK), bSlice[k] (tileK x N_pad),
  // dSlice[k] (F32, M_pad x N_pad) for k = 0..N-1; K-streamed sizing keeps
  // shared memory bounded regardless of K.
  const unsigned N = variant.N;
  // Keyed by (function, op, slice index, element kind, extents) and not by
  // matmul id, so raises in one function that need an identically shaped and
  // typed buffer share it (per-raise buffers blow the 48 KB static cap). Safe
  // because the raised regions are sequential and bracketed by block barriers;
  // the extents in the key prevent reuse at another size.
  const std::string sliceTag = sliceIntrinTag(variant.slice);
  auto scratchName = [&](const char *op, unsigned k, const std::string &prec,
                         uint64_t rows, uint64_t cols) -> std::string {
    return ("__poseidon_ozaki_" + F->getName() + "_" + op + Twine(k) + "_" +
            prec + "_" + Twine(rows) + "x" + Twine(cols))
        .str();
  };
  SmallVector<GlobalVariable *, 8> aScratch(N), bScratch(N), dScratch(N);
  for (unsigned k = 0; k < N; ++k) {
    aScratch[k] = getOrCreateSharedScratch(
        mod, scratchName("a", k, sliceTag, mPad, t.tileK), sliceTy,
        (uint64_t)mPad * t.tileK);
    bScratch[k] = getOrCreateSharedScratch(
        mod, scratchName("b", k, sliceTag, t.tileK, nPad), sliceTy,
        (uint64_t)t.tileK * nPad);
    dScratch[k] =
        getOrCreateSharedScratch(mod, scratchName("d", k, "f32", mPad, nPad),
                                 f32Ty, (uint64_t)mPad * nPad);
  }
  IRBuilder<> B(preheader->getTerminator());

#if LLVM_VERSION_MAJOR > 20
  Function *barFn = Intrinsic::getOrInsertDeclaration(
      mod, Intrinsic::nvvm_barrier_cta_sync_aligned_all);
  SmallVector<Value *, 1> barArgs = {ConstantInt::get(i32Ty, 0)};
#else
  Function *barFn =
      Intrinsic::getOrInsertDeclaration(mod, Intrinsic::nvvm_barrier0);
  SmallVector<Value *, 1> barArgs;
#endif
  Type *i8Ty = Type::getInt8Ty(ctx);

  if (!h.aStartSCEV || !h.bStartSCEV || !h.SE)
    report_fatal_error("OzakiIRaiseMaterializer: missing aStartSCEV / "
                       "bStartSCEV / SE — analyzer didn't populate them");
#if LLVM_VERSION_MAJOR >= 22
  SCEVExpander sex(*h.SE, "poseidon.ozaki");
#else
  SCEVExpander sex(*h.SE, h.SE->getDataLayout(), "poseidon.ozaki");
#endif

  // Fused output indices: the per-thread iter-0 start the block-base back-out
  // subtracts must use the same (possibly fused) index the recognizer used, not
  // the bare fast axis; same helper and consistency check as the Direct raiser.
  Value *rowTid =
      emitTidIndex(B, mod, h.aRowAxis, h.aRowSlowAxis, h.aRowFuseMult);
  Value *colTid =
      emitTidIndex(B, mod, h.bColAxis, h.bColSlowAxis, h.bColFuseMult);
  Value *rowTidI64 = B.CreateZExt(rowTid, i64Ty);
  Value *colTidI64 = B.CreateZExt(colTid, i64Ty);

  // Per-block source bases (per-thread iter-0 start with threadIdx contribution
  // removed). Cooperative fill needs uniform bases across the block.
  Value *aPerThreadStart =
      sex.expandCodeFor(h.aStartSCEV, genPtrTy, preheader->getTerminator());
  Value *bPerThreadStart =
      sex.expandCodeFor(h.bStartSCEV, genPtrTy, preheader->getTerminator());
  if (h.aRowSlowAxis != TidAxis::Unknown &&
      h.aRowFuseUnitByte != h.aLeadingDimByte)
    report_fatal_error("OzakiIRaiseMaterializer: fused A-row unit stride " +
                       Twine(h.aRowFuseUnitByte) + " != A leading dim " +
                       Twine(h.aLeadingDimByte));
  Value *aRowThreadOff =
      B.CreateMul(rowTidI64, ConstantInt::get(i64Ty, h.aLeadingDimByte));
  Value *aBlockBase =
      B.CreateGEP(i8Ty, aPerThreadStart, B.CreateNeg(aRowThreadOff));
  int64_t bColThreadStrideByte = bRowMajor ? srcInputByte : h.bLeadingDimByte;
  if (h.bColSlowAxis != TidAxis::Unknown &&
      h.bColFuseUnitByte != bColThreadStrideByte)
    report_fatal_error("OzakiIRaiseMaterializer: fused B-col unit stride " +
                       Twine(h.bColFuseUnitByte) + " != B column stride " +
                       Twine(bColThreadStrideByte));
  Value *bColThreadOff =
      B.CreateMul(colTidI64, ConstantInt::get(i64Ty, bColThreadStrideByte));
  Value *bBlockBase =
      B.CreateGEP(i8Ty, bPerThreadStart, B.CreateNeg(bColThreadOff));
  (void)rowTidI64;
  (void)colTidI64;

  int64_t bSrcKStepByte = bRowMajor ? h.bStrideByte : srcInputByte;
  int64_t bSrcNStrideByte = bRowMajor ? srcInputByte : h.bLeadingDimByte;

  // Capture the preheader terminator BEFORE the K-stream setup splits the BB.
  Instruction *origTerm = preheader->getTerminator();

  // K-streamed main loop, structurally the Direct WMMA loop with a Veltkamp
  // split + N-way stores in the fill, an N(N+1)/2-mma chain per K-tile and N
  // PHI accumulator groups.
  Constant *cZeroElt = Constant::getNullValue(cFragElemTy);
  int64_t ldaElts = t.tileK;
  int64_t ldbElts = nPad;
  int64_t ldcElts = nPad;
  Value *ldaV = ConstantInt::get(i32Ty, ldaElts);
  Value *ldbV = ConstantInt::get(i32Ty, ldbElts);
  Value *ldcV = ConstantInt::get(i32Ty, ldcElts);

  int64_t aScratchRowByte = (int64_t)t.tileK * sliceByte;
  int64_t bScratchKStepByte = (int64_t)nPad * sliceByte;

  SmallVector<Value *, 8> aSliceGen(N), bSliceGen(N), dSliceGen(N);
  for (unsigned k = 0; k < N; ++k) {
    aSliceGen[k] = B.CreateAddrSpaceCast(aScratch[k], genPtrTy);
    bSliceGen[k] = B.CreateAddrSpaceCast(bScratch[k], genPtrTy);
    dSliceGen[k] = B.CreateAddrSpaceCast(dScratch[k], genPtrTy);
  }

  BasicBlock *preheaderBB = B.GetInsertBlock();
  Instruction *postLoopInst = origTerm;
  BasicBlock *afterLoopBB = preheaderBB->splitBasicBlock(
      postLoopInst->getIterator(), "ozaki_kstream.after");
  preheaderBB->getTerminator()->eraseFromParent();
  BasicBlock *loopHdrBB =
      BasicBlock::Create(ctx, "ozaki_kstream.hdr", F, afterLoopBB);
  BasicBlock *loopBodyBB =
      BasicBlock::Create(ctx, "ozaki_kstream.body", F, afterLoopBB);
  IRBuilder<>(preheaderBB).CreateBr(loopHdrBB);

  IRBuilder<> hdrB(loopHdrBB);
  PHINode *kOuterPhi = hdrB.CreatePHI(i32Ty, 2, "k_outer");
  kOuterPhi->addIncoming(ConstantInt::get(i32Ty, 0), preheaderBB);
  // N PHI accumulator groups: cAccPhis[k][i] is the i-th frag element of the
  // k-th Ozaki order accumulator (k = 0..N-1), at this K-stream iteration.
  unsigned numPhisPerGroup = t.mChain * t.nChain * cFragSize;
  SmallVector<SmallVector<PHINode *, 16>, 8> cAccPhis(N);
  for (unsigned k = 0; k < N; ++k) {
    cAccPhis[k].resize(numPhisPerGroup);
    for (unsigned i = 0; i < numPhisPerGroup; ++i) {
      cAccPhis[k][i] =
          hdrB.CreatePHI(cFragElemTy, 2, ("cAcc" + Twine(k)).str());
      cAccPhis[k][i]->addIncoming(cZeroElt, preheaderBB);
    }
  }
  Value *kLoopBound = ConstantInt::get(i32Ty, (int32_t)(t.kChain * t.tileK));
  Value *kCond = hdrB.CreateICmpULT(kOuterPhi, kLoopBound);
  hdrB.CreateCondBr(kCond, loopBodyBB, afterLoopBB);

  IRBuilder<> bodyB(loopBodyBB);
  Instruction *placeholderBr = bodyB.CreateBr(loopHdrBB);
  bodyB.SetInsertPoint(placeholderBr);
  Value *kOuter64 = bodyB.CreateZExt(kOuterPhi, i64Ty);

  std::pair<Value *, Value *> tlinBszPair =
      emitThreadLinAndBlockSize(bodyB, mod, h.is2DBlock);
  Value *tlin = tlinBszPair.first;
  Value *bsz = tlinBszPair.second;

  Value *aSrcForIter = bodyB.CreateGEP(
      i8Ty, aBlockBase,
      bodyB.CreateMul(kOuter64, ConstantInt::get(i64Ty, h.aStrideByte)));
  Value *bSrcForIter = bodyB.CreateGEP(
      i8Ty, bBlockBase,
      bodyB.CreateMul(kOuter64, ConstantInt::get(i64Ty, bSrcKStepByte)));

  // Cooperative N-way Veltkamp-split fill, padded: the fill covers the full
  // padded tile, an out-of-range element stores an exact zero to all N slices,
  // the load+split is branched rather than selected (a pad lane spends no split
  // arithmetic), and the last K-tile's valid extent is the runtime
  // min(tileK, K - k_outer). The single-shot if-form is only correct when the
  // block covers the padded element count; otherwise a strided loop is emitted.
  Constant *sliceZero = Constant::getNullValue(sliceTy);
  auto emitVeltkampFillPadded = [&](Value *srcBase, int64_t srcRowStride,
                                    int64_t srcColStride,
                                    ArrayRef<Value *> dstSlices,
                                    int64_t dstRowStride, uint64_t fillRows,
                                    uint64_t fillCols, Value *validRows,
                                    Value *validCols) {
    uint64_t numElts = fillRows * fillCols;
    if (numElts == 0)
      return;

    // One element's work at index `counter`; leaves the builder on the merge
    // block so the caller can continue (or loop) from there.
    auto emitElement = [&](IRBuilder<> &eb, Value *counter) {
      Value *fillColsV = ConstantInt::get(i32Ty, (int32_t)fillCols);
      Value *r32 = eb.CreateUDiv(counter, fillColsV, "ozaki_pfill.r");
      Value *c32 = eb.CreateURem(counter, fillColsV, "ozaki_pfill.c");
      Value *inRange =
          eb.CreateAnd(eb.CreateICmpULT(r32, validRows),
                       eb.CreateICmpULT(c32, validCols), "ozaki_pfill.in");
      BasicBlock *predBB = eb.GetInsertBlock();
      BasicBlock *loadBB = BasicBlock::Create(ctx, "ozaki_pfill.load", F);
      BasicBlock *mergeBB = BasicBlock::Create(ctx, "ozaki_pfill.merge", F);
      eb.CreateCondBr(inRange, loadBB, mergeBB);

      IRBuilder<> lB(loadBB);
      Value *srcOff =
          lB.CreateAdd(lB.CreateMul(lB.CreateZExt(r32, i64Ty),
                                    ConstantInt::get(i64Ty, srcRowStride)),
                       lB.CreateMul(lB.CreateZExt(c32, i64Ty),
                                    ConstantInt::get(i64Ty, srcColStride)));
      Value *srcPtr = lB.CreateGEP(i8Ty, srcBase, srcOff);
      Value *srcVal = lB.CreateLoad(srcInputTy, srcPtr);
      Value *valF32 =
          srcInputTy == f32Ty ? srcVal : emitFPCast(lB, srcVal, f32Ty);
      SmallVector<Value *, 8> slices =
          emitNwayVeltkampSplit(lB, mod, valF32, N, variant.slice);
      BasicBlock *loadEndBB = lB.GetInsertBlock();
      lB.CreateBr(mergeBB);

      // All N merge PHIs must come first: a PHI may not follow a non-PHI in a
      // basic block, so the address arithmetic is emitted after them.
      IRBuilder<> mB(mergeBB);
      SmallVector<PHINode *, 8> merged(N);
      for (unsigned k = 0; k < N; ++k) {
        merged[k] = mB.CreatePHI(sliceTy, 2, "ozaki_pfill.v");
        merged[k]->addIncoming(slices[k], loadEndBB);
        merged[k]->addIncoming(sliceZero, predBB);
      }
      Value *r64 = mB.CreateZExt(r32, i64Ty);
      Value *c64 = mB.CreateZExt(c32, i64Ty);
      Value *dstOff =
          mB.CreateAdd(mB.CreateMul(r64, ConstantInt::get(i64Ty, dstRowStride)),
                       mB.CreateMul(c64, ConstantInt::get(i64Ty, sliceByte)));
      for (unsigned k = 0; k < N; ++k)
        mB.CreateStore(merged[k], mB.CreateGEP(i8Ty, dstSlices[k], dstOff));
      eb.SetInsertPoint(mergeBB);
    };

    BasicBlock *origBB = bodyB.GetInsertBlock();
    Instruction *splitInst = &*bodyB.GetInsertPoint();
    BasicBlock *afterBB =
        origBB->splitBasicBlock(splitInst->getIterator(), "ozaki_pfill.after");
    origBB->getTerminator()->eraseFromParent();

    if (numElts <= (uint64_t)m.M * (uint64_t)m.N) {
      // Block covers the padded tile: at most one element per thread. The
      // guard compare must be built in origBB, since bodyB's insertion point
      // has already moved past the split.
      BasicBlock *workBB =
          BasicBlock::Create(ctx, "ozaki_pfill.do", F, afterBB);
      IRBuilder<> gB(origBB);
      gB.CreateCondBr(
          gB.CreateICmpULT(tlin, ConstantInt::get(i32Ty, (int32_t)numElts)),
          workBB, afterBB);
      IRBuilder<> wbB(workBB);
      emitElement(wbB, tlin);
      wbB.CreateBr(afterBB);
    } else {
      BasicBlock *hdrBB =
          BasicBlock::Create(ctx, "ozaki_pfill.hdr", F, afterBB);
      BasicBlock *bodyFillBB =
          BasicBlock::Create(ctx, "ozaki_pfill.body", F, afterBB);
      IRBuilder<>(origBB).CreateBr(hdrBB); // no values built here
      IRBuilder<> hB(hdrBB);
      PHINode *ctr = hB.CreatePHI(i32Ty, 2, "ozaki_pfill.i");
      ctr->addIncoming(tlin, origBB);
      hB.CreateCondBr(
          hB.CreateICmpULT(ctr, ConstantInt::get(i32Ty, (int32_t)numElts)),
          bodyFillBB, afterBB);
      IRBuilder<> fB(bodyFillBB);
      emitElement(fB, ctr);
      Value *next = fB.CreateAdd(ctr, bsz, "ozaki_pfill.next");
      ctr->addIncoming(next, fB.GetInsertBlock());
      fB.CreateBr(hdrBB);
    }
    bodyB.SetInsertPoint(&*afterBB->getFirstInsertionPt());
  };

  // Valid source extents. K-streaming makes the last K-tile short whenever
  // padK > 0, so the valid K count is a RUNTIME min(tileK, K - k_outer).
  Value *validK = ConstantInt::get(i32Ty, (int32_t)t.tileK);
  if (t.padK > 0)
    validK = bodyB.CreateBinaryIntrinsic(
        Intrinsic::umin,
        bodyB.CreateSub(ConstantInt::get(i32Ty, (int32_t)m.K), kOuterPhi),
        ConstantInt::get(i32Ty, (int32_t)t.tileK));
  Value *validM = ConstantInt::get(i32Ty, (int32_t)m.M);
  Value *validN = ConstantInt::get(i32Ty, (int32_t)m.N);

  emitVeltkampFillPadded(aSrcForIter, h.aLeadingDimByte, h.aStrideByte,
                         aSliceGen, aScratchRowByte, (uint64_t)mPad,
                         (uint64_t)t.tileK, validM, validK);
  emitVeltkampFillPadded(bSrcForIter, bSrcKStepByte, bSrcNStrideByte, bSliceGen,
                         bScratchKStepByte, (uint64_t)t.tileK, (uint64_t)nPad,
                         validK, validN);

  bodyB.CreateCall(barFn, barArgs);
  (void)bsz;

  // Warp-gate the mma chain when there is a single output tile per block, and
  // always when the block's thread count is not a multiple of 32: its last
  // warp is partially populated, and warp-collective wmma ops in a partial
  // warp are undefined (the ungated chain corrupts the shared output tile).
  const bool partialWarpBlock =
      (h.blockThreads != 0) && (h.blockThreads % 32 != 0);
  const bool useWarpGate = (t.mChain * t.nChain == 1) || partialWarpBlock;
  BasicBlock *gatePreBB = nullptr;
  BasicBlock *gateMergeBB = nullptr;
  if (useWarpGate) {
    Value *warpId =
        bodyB.CreateLShr(tlin, ConstantInt::get(i32Ty, 5), "warp_id");
    Value *isWarp0 =
        bodyB.CreateICmpEQ(warpId, ConstantInt::get(i32Ty, 0), "is_warp0");
    gatePreBB = bodyB.GetInsertBlock();
    Instruction *splitPt = &*bodyB.GetInsertPoint();
    gateMergeBB =
        gatePreBB->splitBasicBlock(splitPt->getIterator(), "ozaki_gate.merge");
    gatePreBB->getTerminator()->eraseFromParent();
    BasicBlock *gateDoBB =
        BasicBlock::Create(ctx, "ozaki_gate.do", F, gateMergeBB);
    IRBuilder<>(gatePreBB).CreateCondBr(isWarp0, gateDoBB, gateMergeBB);
    bodyB.SetInsertPoint(gateDoBB);
  }

  // cAccLoop[k][i] = back-edge value of k-th accumulator's i-th frag elt.
  SmallVector<SmallVector<Value *, 16>, 8> cAccLoop(N);
  for (unsigned k = 0; k < N; ++k)
    cAccLoop[k].resize(numPhisPerGroup);

  auto callMma = [&](Value *aTile, Value *bTile,
                     ArrayRef<Value *> cIn) -> Value * {
    Value *aFragV = bodyB.CreateCall(fns.loadA, {aTile, ldaV});
    Value *bFragV = bodyB.CreateCall(fns.loadB, {bTile, ldbV});
    SmallVector<Value *, 32> args;
    args.reserve(aFragSize + bFragSize + cFragSize);
    for (unsigned i = 0; i < aFragSize; ++i)
      args.push_back(aFragSize == 1 ? aFragV
                                    : bodyB.CreateExtractValue(aFragV, i));
    for (unsigned i = 0; i < bFragSize; ++i)
      args.push_back(bFragSize == 1 ? bFragV
                                    : bodyB.CreateExtractValue(bFragV, i));
    for (unsigned i = 0; i < cFragSize; ++i)
      args.push_back(cIn[i]);
    return bodyB.CreateCall(fns.mma, args);
  };

  for (unsigned mt = 0; mt < t.mChain; ++mt) {
    for (unsigned nt = 0; nt < t.nChain; ++nt) {
      SmallVector<Value *, 8> aRowBases(N), bColBases(N);
      for (unsigned k = 0; k < N; ++k) {
        aRowBases[k] = bodyB.CreatePtrAdd(
            aSliceGen[k],
            ConstantInt::get(i64Ty, (int64_t)mt * t.tileM * aScratchRowByte));
        bColBases[k] = bodyB.CreatePtrAdd(
            bSliceGen[k],
            ConstantInt::get(i64Ty, (int64_t)nt * t.tileN * sliceByte));
      }
      unsigned phiBase = (mt * t.nChain + nt) * cFragSize;

      SmallVector<SmallVector<Value *, 16>, 8> cur(N);
      for (unsigned k = 0; k < N; ++k) {
        cur[k].resize(cFragSize);
        for (unsigned i = 0; i < cFragSize; ++i)
          cur[k][i] = cAccPhis[k][phiBase + i];
      }

      // For each (i, j) with i + j < N accumulate a[i]*b[j] into group c[i+j];
      // products with i + j >= N are below the captured resolution. The
      // dominant c[0] uses fresh acc + add rather than a chained mma: the
      // K-stream reduction stays outside the wmma's internal K-reduction, and
      // the order matters in the F32 sum.
      SmallVector<Value *, 16> zeroAcc(cFragSize, cZeroElt);
      for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j + i < N; ++j) {
          unsigned kg = i + j;
          if (i == 0 && j == 0) {
            Value *tmpV = callMma(aRowBases[0], bColBases[0], zeroAcc);
            for (unsigned e = 0; e < cFragSize; ++e) {
              Value *tmpElt =
                  (cFragSize == 1) ? tmpV : bodyB.CreateExtractValue(tmpV, e);
              cur[0][e] = bodyB.CreateFAdd(cur[0][e], tmpElt);
            }
            continue;
          }
          Value *dV = callMma(aRowBases[i], bColBases[j], cur[kg]);
          for (unsigned e = 0; e < cFragSize; ++e)
            cur[kg][e] =
                (cFragSize == 1) ? dV : bodyB.CreateExtractValue(dV, e);
        }
      }

      for (unsigned k = 0; k < N; ++k)
        for (unsigned i = 0; i < cFragSize; ++i)
          cAccLoop[k][phiBase + i] = cur[k][i];
    }
  }

  if (useWarpGate) {
    BasicBlock *gateEndBB = bodyB.GetInsertBlock();
    bodyB.CreateBr(gateMergeBB);
    IRBuilder<> mergeB(gateMergeBB, gateMergeBB->getFirstInsertionPt());
    for (unsigned k = 0; k < N; ++k) {
      for (unsigned i = 0; i < numPhisPerGroup; ++i) {
        PHINode *p = mergeB.CreatePHI(cFragElemTy, 2,
                                      ("cAcc" + Twine(k) + ".merge").str());
        p->addIncoming(cAccLoop[k][i], gateEndBB);
        p->addIncoming(cAccPhis[k][i], gatePreBB);
        cAccLoop[k][i] = p;
      }
    }
    bodyB.SetInsertPoint(gateMergeBB, gateMergeBB->getFirstInsertionPt());
  }

  bodyB.CreateCall(barFn, barArgs);

  Value *kNext =
      bodyB.CreateAdd(kOuterPhi, ConstantInt::get(i32Ty, (int32_t)t.tileK));
  BasicBlock *latchBB = placeholderBr->getParent();
  kOuterPhi->addIncoming(kNext, latchBB);
  for (unsigned k = 0; k < N; ++k)
    for (unsigned i = 0; i < numPhisPerGroup; ++i)
      cAccPhis[k][i]->addIncoming(cAccLoop[k][i], latchBB);

  IRBuilder<> afterB(afterLoopBB, afterLoopBB->getFirstNonPHIIt());
  BasicBlock *storeMergeBB = nullptr;
  if (useWarpGate) {
    auto [aTlin, aBsz] = emitThreadLinAndBlockSize(afterB, mod, h.is2DBlock);
    (void)aBsz;
    Value *aWarpId = afterB.CreateLShr(aTlin, ConstantInt::get(i32Ty, 5));
    Value *aIsWarp0 = afterB.CreateICmpEQ(aWarpId, ConstantInt::get(i32Ty, 0));
    Instruction *splitPt = &*afterB.GetInsertPoint();
    BasicBlock *curr = afterB.GetInsertBlock();
    storeMergeBB =
        curr->splitBasicBlock(splitPt->getIterator(), "ozaki_store_gate.merge");
    curr->getTerminator()->eraseFromParent();
    BasicBlock *storeDoBB =
        BasicBlock::Create(ctx, "ozaki_store_gate.do", F, storeMergeBB);
    IRBuilder<>(curr).CreateCondBr(aIsWarp0, storeDoBB, storeMergeBB);
    afterB.SetInsertPoint(storeDoBB);
  }

  for (unsigned mt = 0; mt < t.mChain; ++mt) {
    for (unsigned nt = 0; nt < t.nChain; ++nt) {
      Value *dTileByte =
          ConstantInt::get(i64Ty, ((int64_t)mt * t.tileM * (int64_t)nPad +
                                   (int64_t)nt * t.tileN) *
                                      f32Byte);
      unsigned phiBase = (mt * t.nChain + nt) * cFragSize;
      for (unsigned k = 0; k < N; ++k) {
        Value *dTilePtr = afterB.CreatePtrAdd(dSliceGen[k], dTileByte);
        SmallVector<Value *, 16> args;
        args.reserve(2 + cFragSize);
        args.push_back(dTilePtr);
        for (unsigned i = 0; i < cFragSize; ++i)
          args.push_back(cAccPhis[k][phiBase + i]);
        args.push_back(ldcV);
        afterB.CreateCall(fns.storeD, args);
      }
    }
  }
  if (useWarpGate) {
    afterB.CreateBr(storeMergeBB);
    afterB.SetInsertPoint(storeMergeBB, storeMergeBB->getFirstInsertionPt());
  }
  afterB.CreateCall(barFn, barArgs);

  // Readback: Σ_k d[k] / SCALE^k, in F32 (F32-class).
  auto insertIt = exitBB->getFirstNonPHIIt();
  IRBuilder<> RB(exitBB, insertIt);
  Value *rRow =
      emitTidIndex(RB, mod, h.aRowAxis, h.aRowSlowAxis, h.aRowFuseMult);
  Value *rCol =
      emitTidIndex(RB, mod, h.bColAxis, h.bColSlowAxis, h.bColFuseMult);
  Value *rCellIdx =
      RB.CreateAdd(RB.CreateMul(rRow, ConstantInt::get(i32Ty, nPad)), rCol);
  double SCALE = veltkampScale(variant.slice);
  Value *cF32 = nullptr;
  double scalePowK = 1.0;
  for (unsigned k = 0; k < N; ++k) {
    Value *dGen = RB.CreateAddrSpaceCast(dScratch[k], genPtrTy);
    Value *cell = RB.CreateGEP(f32Ty, dGen, rCellIdx);
    LoadInst *raw = RB.CreateLoad(f32Ty, cell);
    raw->setAlignment(Align(f32Byte));
    Value *term =
        (k == 0) ? raw : RB.CreateFDiv(raw, ConstantFP::get(f32Ty, scalePowK));
    cF32 = (k == 0) ? term : RB.CreateFAdd(cF32, term);
    scalePowK *= SCALE;
  }
  Value *readback = (srcAccTy == f32Ty) ? cF32 : emitFPCast(RB, cF32, srcAccTy);

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
}

void materializeOzakiIRaise(const AbstractMatmul &m,
                            const CandidateMatmul::Option &opt) {
  OzakiIRaiseMaterializer(m, opt).run();
}

std::string ozakiIOptionLabel(const CandidateMatmul::Option &opt) {
  // FP16-slice + F32-readback tensor-core error correction (TCEC).
  return (Twine("tcec") + " n=" + Twine(opt.strategyParam) + " wmma m" +
          Twine(opt.tileM) + "n" + Twine(opt.tileN) + "k" + Twine(opt.tileK) +
          " " + fpKindName(opt.inputPrec) + "/" + fpKindName(opt.accPrec))
      .str();
}

} // namespace poseidon
