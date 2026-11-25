//===- InKernelRaise.cpp - tensor-core code for a scalar-loop product -----===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The in-kernel realization of a scalar-loop matrix product: a K-streamed loop
// that fills shared-memory tiles of A and B cooperatively, multiplies them with
// WMMA fragments, stores the accumulators to a shared output tile and reads
// each thread's element back where the reduction loop's result was used. The
// direct scheme fills one rounded copy of each operand and carries one
// accumulator per output tile. The split scheme fills N Veltkamp slices of
// each operand, accumulates a[i] * b[j] into group i + j for i + j < N, and
// reads back sum_k d[k] / SCALE^k.
//
//===----------------------------------------------------------------------===//

#include "IR/Utils.h"
#include "MatmulInternal.h"
#include "Model/FPFormat.h"
#include "Targets/NVPTXTensorCore.h"
#include "Targets/Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <cmath>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace poseidon {

// fptrunc/fpext between FP types (no-op when they match). Two types of one
// width never meet: a product is raised only in float or double, whose
// multiply-adds are the ones a profile records.
static Value *emitFPCast(IRBuilder<> &B, Value *v, Type *toTy) {
  Type *fromTy = v->getType();
  if (fromTy == toTy)
    return v;
  unsigned fromBits = fromTy->getScalarSizeInBits();
  unsigned toBits = toTy->getScalarSizeInBits();
  assert(fromBits != toBits && "no conversion between half and bfloat");
  if (toBits < fromBits)
    return B.CreateFPTrunc(v, toTy);
  return B.CreateFPExt(v, toTy);
}

static GlobalVariable *getOrCreateSharedScratch(Module *M, const Twine &name,
                                                Type *eltTy, uint64_t numElts) {
  std::string nameStr = name.str();
  if (auto *gv = M->getNamedGlobal(nameStr))
    return gv;
  ArrayType *arrTy = ArrayType::get(eltTy, numElts);
  Constant *zero = ConstantAggregateZero::get(arrTy);
  auto *gv = new GlobalVariable(*M, arrTy, /*isConstant=*/false,
                                GlobalValue::InternalLinkage, zero, nameStr,
                                nullptr, GlobalVariable::NotThreadLocal,
                                *Target::of(*M).sharedAddressSpace());
  gv->setAlignment(MaybeAlign(16));
  return gv;
}

// The tid intrinsic for the given axis; aborts on Unknown / Other.
static Value *emitTidIntrinsic(IRBuilder<> &B, Module *M, TidAxis axis) {
  std::optional<unsigned> a = tidAxisIndex(axis);
  if (!a)
    report_fatal_error("emitTidIntrinsic: axis not resolved (Unknown/Other)");
  return nvptx::emitThreadIndex(B, M, *a);
}

// A (possibly fused) matrix row/column index: tid[fast] when slow is Unknown,
// otherwise tid[fast] + mult * tid[slow], where `mult` is the extent of the
// fast axis.
static Value *emitTidIndex(IRBuilder<> &B, Module *M, TidAxis fast,
                           TidAxis slow, unsigned mult) {
  Value *fastV = emitTidIntrinsic(B, M, fast);
  if (slow == TidAxis::Unknown)
    return fastV;
  if (mult == 0)
    report_fatal_error("emitTidIndex: fused index with a zero multiplier");
  Value *slowV = emitTidIntrinsic(B, M, slow);
  Type *i32Ty = Type::getInt32Ty(B.getContext());
  Value *slowOff = B.CreateMul(slowV, ConstantInt::get(i32Ty, (int32_t)mult));
  return B.CreateAdd(fastV, slowOff, "wmma.fused_idx");
}

// (linearizedThreadId, blockSize) from tid.{x,y,z} and ntid.{x,y,z}.
// `is2DBlock` skips the tid.z / ntid.z reads; only set it when the block's z
// dim is statically 1.
static std::pair<Value *, Value *>
emitThreadLinAndBlockSize(IRBuilder<> &B, Module *M, bool is2DBlock) {
  Value *tx = nvptx::emitThreadIndex(B, M, 0);
  Value *ty = nvptx::emitThreadIndex(B, M, 1);
  Value *nx = nvptx::emitBlockDim(B, M, 0);
  Value *ny = nvptx::emitBlockDim(B, M, 1);
  if (is2DBlock) {
    Value *row = B.CreateMul(ty, nx);
    Value *tlin = B.CreateAdd(row, tx);
    Value *bsz = B.CreateMul(nx, ny);
    return {tlin, bsz};
  }
  Value *tz = nvptx::emitThreadIndex(B, M, 2);
  Value *nz = nvptx::emitBlockDim(B, M, 2);
  Value *plane = B.CreateMul(tz, ny);
  Value *planeRow = B.CreateAdd(plane, ty);
  Value *row = B.CreateMul(planeRow, nx);
  Value *tlin = B.CreateAdd(row, tx);
  Value *area = B.CreateMul(nx, ny);
  Value *bsz = B.CreateMul(area, nz);
  return {tlin, bsz};
}

static double veltkampScale(FPKind slice) {
  return std::ldexp(1.0, sliceFormat(slice).scaleExponent);
}

// Quantize an F32 value to slice precision (half for F16/BF16; float for TF32,
// bit-identical to F32 with the bottom 13 mantissa bits zeroed).
static Value *emitSliceQuantize(IRBuilder<> &B, Module *mod, Value *vs_f32,
                                FPKind slice) {
  if (slice == FPKind::F16 || slice == FPKind::BF16)
    return B.CreateFPTrunc(vs_f32, llvmTypeForFPKind(B.getContext(), slice));
  return nvptx::emitRoundToTF32(B, mod, vs_f32);
}

// Slice value back to F32 for the next residual; TF32 is already F32 storage.
static Value *emitSliceToF32(IRBuilder<> &B, Value *slice, FPKind kind) {
  Type *f32Ty = Type::getFloatTy(B.getContext());
  if (kind == FPKind::TF32)
    return slice; // bit-pattern already in float
  return B.CreateFPExt(slice, f32Ty);
}

// N-slice Veltkamp split:
//   s[0]  = quantize(v)
//   r[k]  = (r[k-1] - s[k-1]_f32) * SCALE ; s[k] = quantize(r[k])
// Reconstruction: v ~ sum_k s[k] / SCALE^k, each slice capturing the
// significand bits of its format (sliceFormat).
static SmallVector<Value *, 8> emitNwayVeltkampSplit(IRBuilder<> &B,
                                                     Module *mod, Value *vs_f32,
                                                     unsigned N, FPKind slice) {
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

namespace {
// A (rows x cols) tile of global memory copied into shared-memory tiles of the
// same shape, one per value the fill's payload makes of an element.
struct TileFill {
  Value *src;
  int64_t srcRowStride, srcColStride; // bytes
  ArrayRef<Value *> dst;
  int64_t dstRowStride, dstColStride; // bytes
  uint64_t rows, cols;
  // The extents of the source inside the tile at runtime; every destination of
  // an element outside them is written an exact zero. Null when the fill does
  // not check them.
  Value *validRows, *validCols;
};
} // namespace

// r * rowStride + c * colStride in bytes, with r and c widened to i64.
static Value *emitTileOffset(IRBuilder<> &B, Value *r, Value *c,
                             int64_t rowStride, int64_t colStride) {
  Type *i64Ty = B.getInt64Ty();
  Value *r64 = B.CreateZExt(r, i64Ty);
  Value *rowOff = B.CreateMul(r64, ConstantInt::get(i64Ty, rowStride));
  Value *c64 = B.CreateZExt(c, i64Ty);
  Value *colOff = B.CreateMul(c64, ConstantInt::get(i64Ty, colStride));
  return B.CreateAdd(rowOff, colOff);
}

// Fills `t` cooperatively: the thread of linear index `tlin` in a block of
// `bsz` threads copies elements tlin, tlin + bsz, ..., or only element tlin
// when `singleShot`, which is wrong unless the block covers the tile.
// `payload` makes the value stored to each t.dst of one source element. An
// element outside the valid extents is branched around rather than selected:
// a pad lane loads nothing and spends no payload arithmetic, and the source
// index needs no clamp. Splits the current block; B ends at the start of the
// block after the fill. Value names start with `name`.
static void emitTileFill(
    IRBuilder<> &B, StringRef name, const TileFill &t, Type *srcTy, Type *dstTy,
    Value *tlin, Value *bsz, bool singleShot,
    function_ref<SmallVector<Value *, 8>(IRBuilder<> &, Value *)> payload) {
  uint64_t numElts = t.rows * t.cols;
  if (numElts == 0)
    return;
  BasicBlock *origBB = B.GetInsertBlock();
  Function *F = origBB->getParent();
  LLVMContext &ctx = F->getContext();
  Type *i32Ty = Type::getInt32Ty(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *i8Ty = Type::getInt8Ty(ctx);
  auto named = [&](const char *suffix) { return (name + suffix).str(); };

  auto load = [&](IRBuilder<> &LB, Value *r, Value *c) {
    Value *srcOff = emitTileOffset(LB, r, c, t.srcRowStride, t.srcColStride);
    Value *srcPtr = LB.CreateGEP(i8Ty, t.src, srcOff);
    Value *srcVal = LB.CreateLoad(srcTy, srcPtr);
    return payload(LB, srcVal);
  };
  auto store = [&](IRBuilder<> &SB, Value *r64, Value *c64,
                   ArrayRef<Value *> vals) {
    Value *dstOff =
        emitTileOffset(SB, r64, c64, t.dstRowStride, t.dstColStride);
    for (auto [dst, val] : llvm::zip_equal(t.dst, vals)) {
      Value *dstPtr = SB.CreateGEP(i8Ty, dst, dstOff);
      SB.CreateStore(val, dstPtr);
    }
  };
  // One element's work at index `counter`; leaves EB in the block where the
  // element ends.
  auto emitElement = [&](IRBuilder<> &EB, Value *counter) {
    Value *colsV = ConstantInt::get(i32Ty, (int32_t)t.cols);
    Value *r = EB.CreateUDiv(counter, colsV, named(".r"));
    Value *c = EB.CreateURem(counter, colsV, named(".c"));
    if (!t.validRows) {
      Value *r64 = EB.CreateZExt(r, i64Ty);
      Value *c64 = EB.CreateZExt(c, i64Ty);
      SmallVector<Value *, 8> vals = load(EB, r64, c64);
      store(EB, r64, c64, vals);
      return;
    }
    Value *rowIn = EB.CreateICmpULT(r, t.validRows);
    Value *colIn = EB.CreateICmpULT(c, t.validCols);
    Value *inRange = EB.CreateAnd(rowIn, colIn, named(".in"));
    BasicBlock *predBB = EB.GetInsertBlock();
    BasicBlock *loadBB = BasicBlock::Create(ctx, named(".load"), F);
    BasicBlock *mergeBB = BasicBlock::Create(ctx, named(".merge"), F);
    EB.CreateCondBr(inRange, loadBB, mergeBB);

    IRBuilder<> LB(loadBB);
    SmallVector<Value *, 8> vals = load(LB, r, c);
    BasicBlock *loadEndBB = LB.GetInsertBlock();
    LB.CreateBr(mergeBB);

    // The PHIs lead the merge block, ahead of the address arithmetic.
    IRBuilder<> MB(mergeBB);
    Constant *zero = Constant::getNullValue(dstTy);
    SmallVector<Value *, 8> merged;
    for (Value *val : vals) {
      PHINode *phi = MB.CreatePHI(dstTy, 2, named(".v"));
      phi->addIncoming(val, loadEndBB);
      phi->addIncoming(zero, predBB);
      merged.push_back(phi);
    }
    Value *r64 = MB.CreateZExt(r, i64Ty);
    Value *c64 = MB.CreateZExt(c, i64Ty);
    store(MB, r64, c64, merged);
    EB.SetInsertPoint(mergeBB);
  };

  Instruction *splitInst = &*B.GetInsertPoint();
  BasicBlock *afterBB =
      origBB->splitBasicBlock(splitInst->getIterator(), named(".after"));
  origBB->getTerminator()->eraseFromParent();
  Value *numEltsV = ConstantInt::get(i32Ty, (int32_t)numElts);

  if (singleShot) {
    BasicBlock *workBB = BasicBlock::Create(ctx, named(".do"), F, afterBB);
    IRBuilder<> GB(origBB);
    Value *covered = GB.CreateICmpULT(tlin, numEltsV);
    GB.CreateCondBr(covered, workBB, afterBB);
    IRBuilder<> WB(workBB);
    emitElement(WB, tlin);
    WB.CreateBr(afterBB);
  } else {
    BasicBlock *hdrBB = BasicBlock::Create(ctx, named(".hdr"), F, afterBB);
    BasicBlock *bodyBB = BasicBlock::Create(ctx, named(".body"), F, afterBB);
    IRBuilder<>(origBB).CreateBr(hdrBB);
    IRBuilder<> HB(hdrBB);
    PHINode *ctr = HB.CreatePHI(i32Ty, 2, named(".i"));
    ctr->addIncoming(tlin, origBB);
    Value *more = HB.CreateICmpULT(ctr, numEltsV);
    HB.CreateCondBr(more, bodyBB, afterBB);
    IRBuilder<> FB(bodyBB);
    emitElement(FB, ctr);
    Value *next = FB.CreateAdd(ctr, bsz, named(".next"));
    ctr->addIncoming(next, FB.GetInsertBlock());
    FB.CreateBr(hdrBB);
  }
  B.SetInsertPoint(&*afterBB->getFirstInsertionPt());
}

// Branches from `preheader` to a new block, which it returns, when the block
// shape of the launch is the profiled one and to `loopHeader` otherwise. The
// branch is uniform across the block.
static BasicBlock *emitBlockShapeGuard(BasicBlock &preheader,
                                       BasicBlock &loopHeader,
                                       const ScalarLoopHandle &h) {
  Function *F = preheader.getParent();
  Module *mod = F->getParent();
  IRBuilder<> GB(preheader.getTerminator());
  Value *sameShape = nullptr;
  for (unsigned axis = 0; axis < 3; ++axis) {
    Value *eq = GB.CreateICmpEQ(
        nvptx::emitBlockDim(GB, mod, axis),
        ConstantInt::get(GB.getInt32Ty(), h.profiledBlock[axis]));
    sameShape = sameShape ? GB.CreateAnd(sameShape, eq) : eq;
  }
  BasicBlock *raiseBB =
      BasicBlock::Create(mod->getContext(), "raise.entry", F, &loopHeader);
  IRBuilder<>(raiseBB).CreateBr(&loopHeader);
  GB.CreateCondBr(sameShape, raiseBB, &loopHeader);
  preheader.getTerminator()->eraseFromParent();
  return raiseBB;
}

void materializeInKernelRaise(const AbstractMatmul &m,
                              const MatmulUnit::Option &t) {
  const SchemeVariant &v = t.variant;
  if (t.realization != Realization::InKernel ||
      (v.scheme != Scheme::Direct && v.scheme != Scheme::Tcec))
    report_fatal_error("materializeInKernelRaise: only the in-kernel direct "
                       "and split schemes are realized here");
  const bool split = v.scheme == Scheme::Tcec;
  // Operand slices, and accumulator groups: one each in the direct scheme.
  const unsigned N = split ? sliceCount(v) : 1;
  const FPKind operand = v.operand;
  if (split) {
    if (N < 2 || N > 8)
      report_fatal_error("materializeInKernelRaise: the slice count must be "
                         "in [2, 8] (got " +
                         Twine(N) + ")");
    if (operand != FPKind::F16 && operand != FPKind::BF16 &&
        operand != FPKind::TF32)
      report_fatal_error("materializeInKernelRaise: input slice must be F16, "
                         "BF16, or TF32 (got " +
                         Twine(fpKindName(operand)) + ")");
    if (v.accumulator != FPKind::F32)
      report_fatal_error(
          "materializeInKernelRaise: accumulator must be F32 (got " +
          Twine(fpKindName(v.accumulator)) + ")");
  } else if (operand != FPKind::F16 && operand != FPKind::BF16 &&
             operand != FPKind::TF32 && operand != FPKind::F64) {
    report_fatal_error("materializeInKernelRaise: unsupported input precision");
  }
  if (t.tileM == 0 || t.tileN == 0 || t.tileK == 0 || t.mChain == 0 ||
      t.nChain == 0 || t.kChain == 0)
    report_fatal_error("materializeInKernelRaise: zero tile/chain dim");
  if (t.mChain * t.tileM != m.M + t.padM ||
      t.nChain * t.tileN != m.N + t.padN || t.kChain * t.tileK != m.K + t.padK)
    report_fatal_error(
        "materializeInKernelRaise: chain*tile doesn't match source+padding");

  const ScalarLoopHandle &h = m.scalarLoop;
  BasicBlock *preheader = h.preheader;
  BasicBlock *exitBB = h.exitBB;
  assert(preheader && exitBB &&
         "recognition admits only loops with a preheader and one exit block");
  assert(h.inKernelRaisable &&
         "an in-kernel raise is proposed only where it can take the loop's "
         "place");

  Function *F = preheader->getParent();
  Module *mod = F->getParent();

  // The raise is laid out for the profiled block shape, so it runs only in a
  // block of that shape; a launch of another shape runs the loop as written.
  BasicBlock *loopHeader = preheader->getSingleSuccessor();
  BasicBlock *loopLatch = exitBB->getSinglePredecessor();
  assert(loopHeader && loopLatch &&
         "recognition admits only a loop whose exit is entered from its latch");
  preheader = emitBlockShapeGuard(*preheader, *loopHeader, h);
  LLVMContext &ctx = mod->getContext();
  const DataLayout &DL = mod->getDataLayout();
  Type *i8Ty = Type::getInt8Ty(ctx);
  Type *i32Ty = Type::getInt32Ty(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *f32Ty = Type::getFloatTy(ctx);
  PointerType *genPtrTy = PointerType::get(ctx, /*addrspace=*/0);

  // The scratch element type matches the wmma fragment's element view; TF32
  // wmma loads bit-cast F32 storage into i32 fragments, so F32 is stored after
  // explicit rounding to TF32.
  Type *srcTy = llvmTypeForFPKind(ctx, m.elemType);
  Type *opTy = llvmTypeForFPKind(ctx, operand);
  Type *accTy = llvmTypeForFPKind(ctx, v.accumulator);
  int64_t srcByte = (int64_t)DL.getTypeAllocSize(srcTy).getFixedValue();
  int64_t opByte = (int64_t)DL.getTypeAllocSize(opTy).getFixedValue();
  int64_t accByte = (int64_t)DL.getTypeAllocSize(accTy).getFixedValue();

  // B layout from the captured reduction-axis stride: a one-element K stride
  // means K-contiguous (col-major), otherwise K-strided (row-major). B always
  // routes through scratch, so this only sets the fill's read strides.
  if (h.bStrideByte % srcByte != 0)
    report_fatal_error("materializeInKernelRaise: B reduction-axis stride " +
                       Twine(h.bStrideByte) +
                       " is not a multiple of the element size " +
                       Twine(srcByte));
  const bool bRowMajor = (h.bStrideByte != srcByte);

  // Padded extents (chain * tile); the A/B scratch K-dim is one tileK chunk
  // (K-streamed), the output scratch is the full mPad x nPad.
  const unsigned mPad = t.mChain * t.tileM;
  const unsigned nPad = t.nChain * t.tileN;

  const nvptx::Wmma wmma =
      nvptx::declareWmma(mod, t.tileM, t.tileN, t.tileK, fpKindName(operand),
                         fpKindName(v.accumulator), /*bRowMajor=*/true);
  FunctionType *mmaFTy = wmma.Mma->getFunctionType();
  auto fragSize = [](Type *T) -> unsigned {
    if (auto *ST = dyn_cast<StructType>(T))
      return ST->getNumElements();
    return 1;
  };
  unsigned aFragSize = fragSize(wmma.LoadA->getFunctionType()->getReturnType());
  unsigned bFragSize = fragSize(wmma.LoadB->getFunctionType()->getReturnType());
  unsigned dFragSize = fragSize(mmaFTy->getReturnType());
  unsigned cFragSize = mmaFTy->getNumParams() - aFragSize - bFragSize;
  if (cFragSize != dFragSize)
    report_fatal_error("materializeInKernelRaise: mma C/D fragment shapes "
                       "differ; chained accumulation requires C precision == "
                       "D precision");
  Type *cFragElemTy = mmaFTy->getParamType(aFragSize + bFragSize);

  // Value names: the split scheme's start with "ozaki", the direct scheme's
  // with "wmma".
  const StringRef tag = split ? "ozaki" : "wmma";
  auto accName = [&](unsigned k) {
    return split ? ("cAcc" + Twine(k)).str() : std::string("cFrag");
  };

  // Scratch per slice k: A (mPad x tileK) and B (tileK x nPad) of the operand
  // format, and the output (mPad x nPad) of the accumulator format; the
  // K-streamed sizing keeps shared memory bounded regardless of K. Keyed by
  // (function, op, slice, format, extents) and not by matmul id, so raises in
  // one function that need an identically shaped buffer share one (per-raise
  // buffers would exceed the 48 KB static cap). Safe because the raised
  // regions are sequential and bracketed by block barriers; the extents in
  // the key prevent reuse at another size.
  auto scratchName = [&](StringRef op, unsigned k, StringRef prec,
                         uint64_t rows, uint64_t cols) -> std::string {
    std::string slice = split ? std::to_string(k) : std::string();
    return ("__poseidon_" + Twine(split ? "ozaki" : "raise") + "_" +
            F->getName() + "_" + op + slice + "_" + prec + "_" + Twine(rows) +
            "x" + Twine(cols))
        .str();
  };
  SmallVector<GlobalVariable *, 8> aScratch(N), bScratch(N), dScratch(N);
  for (unsigned k = 0; k < N; ++k) {
    aScratch[k] = getOrCreateSharedScratch(
        mod, scratchName("a", k, fpKindName(operand), mPad, t.tileK), opTy,
        (uint64_t)mPad * t.tileK);
    bScratch[k] = getOrCreateSharedScratch(
        mod, scratchName("b", k, fpKindName(operand), t.tileK, nPad), opTy,
        (uint64_t)t.tileK * nPad);
    dScratch[k] = getOrCreateSharedScratch(
        mod, scratchName("d", k, fpKindName(v.accumulator), mPad, nPad), accTy,
        (uint64_t)mPad * nPad);
  }

  IRBuilder<> B(preheader->getTerminator());
  const nvptx::Barrier barrier = nvptx::declareBarrier(mod);

  if (!h.aStartSCEV || !h.bStartSCEV || !h.SE)
    report_fatal_error("materializeInKernelRaise: missing aStartSCEV / "
                       "bStartSCEV / SE; analyzer didn't populate them");
  SCEVExpander sex = makeSCEVExpander(*h.SE, "poseidon.raise");

  // Per-block source bases: the per-thread iteration-0 start less the
  // threadIdx contribution of the (possibly fused) index the recognizer
  // matched, so the cooperative fill sees one base across the block.
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
    report_fatal_error("materializeInKernelRaise: fused A-row unit stride " +
                       Twine(h.aRowFuseUnitByte) + " != A leading dim " +
                       Twine(h.aLeadingDimByte));
  Value *aRowThreadOff =
      B.CreateMul(rowTidI64, ConstantInt::get(i64Ty, h.aLeadingDimByte));
  Value *aRowThreadBack = B.CreateNeg(aRowThreadOff);
  Value *aBlockBase = B.CreateGEP(i8Ty, aPerThreadStart, aRowThreadBack);
  int64_t bColThreadStrideByte = bRowMajor ? srcByte : h.bLeadingDimByte;
  if (h.bColSlowAxis != TidAxis::Unknown &&
      h.bColFuseUnitByte != bColThreadStrideByte)
    report_fatal_error("materializeInKernelRaise: fused B-col unit stride " +
                       Twine(h.bColFuseUnitByte) + " != B column stride " +
                       Twine(bColThreadStrideByte));
  Value *bColThreadOff =
      B.CreateMul(colTidI64, ConstantInt::get(i64Ty, bColThreadStrideByte));
  Value *bColThreadBack = B.CreateNeg(bColThreadOff);
  Value *bBlockBase = B.CreateGEP(i8Ty, bPerThreadStart, bColThreadBack);

  int64_t bSrcKStepByte = bRowMajor ? h.bStrideByte : srcByte;
  int64_t bSrcNStrideByte = bRowMajor ? srcByte : h.bLeadingDimByte;

  // The scratch tiles are tight-packed at (mPad x tileK) / (tileK x nPad).
  int64_t aScratchRowByte = (int64_t)t.tileK * opByte;
  int64_t bScratchKStepByte = (int64_t)nPad * opByte;
  Value *ldaV = ConstantInt::get(i32Ty, (int64_t)t.tileK);
  Value *ldbV = ConstantInt::get(i32Ty, (int64_t)nPad);
  Value *ldcV = ConstantInt::get(i32Ty, (int64_t)nPad);

  SmallVector<Value *, 8> aGen(N), bGen(N), dGen(N);
  for (unsigned k = 0; k < N; ++k) {
    aGen[k] = B.CreateAddrSpaceCast(aScratch[k], genPtrTy);
    bGen[k] = B.CreateAddrSpaceCast(bScratch[k], genPtrTy);
    dGen[k] = B.CreateAddrSpaceCast(dScratch[k], genPtrTy);
  }

  // K-streaming outer loop: per k_outer, refill the A/B scratch with the next
  // K-tile chunk, barrier, run the mma chain of every (mt, nt) output tile
  // with the accumulators carried in PHIs, barrier; after the loop the
  // accumulators are stored to the output scratch.
  BasicBlock *preheaderBB = B.GetInsertBlock();
  Instruction *postLoopInst = preheader->getTerminator();
  BasicBlock *afterLoopBB = preheaderBB->splitBasicBlock(
      postLoopInst->getIterator(), tag + "_kstream.after");
  preheaderBB->getTerminator()->eraseFromParent();
  BasicBlock *loopHdrBB =
      BasicBlock::Create(ctx, tag + "_kstream.hdr", F, afterLoopBB);
  BasicBlock *loopBodyBB =
      BasicBlock::Create(ctx, tag + "_kstream.body", F, afterLoopBB);
  IRBuilder<>(preheaderBB).CreateBr(loopHdrBB);

  // Header PHIs: k_outer plus one PHI per (group, mt, nt, fragment element).
  IRBuilder<> hdrB(loopHdrBB);
  PHINode *kOuterPhi = hdrB.CreatePHI(i32Ty, 2, "k_outer");
  kOuterPhi->addIncoming(ConstantInt::get(i32Ty, 0), preheaderBB);
  Constant *cZeroElt = Constant::getNullValue(cFragElemTy);
  const unsigned numPhisPerGroup = t.mChain * t.nChain * cFragSize;
  SmallVector<SmallVector<PHINode *, 16>, 8> accPhis(N);
  for (unsigned k = 0; k < N; ++k) {
    accPhis[k].resize(numPhisPerGroup);
    for (unsigned i = 0; i < numPhisPerGroup; ++i) {
      accPhis[k][i] = hdrB.CreatePHI(cFragElemTy, 2, accName(k));
      accPhis[k][i]->addIncoming(cZeroElt, preheaderBB);
    }
  }
  Value *kLoopBound = ConstantInt::get(i32Ty, (int32_t)(t.kChain * t.tileK));
  Value *kCond = hdrB.CreateICmpULT(kOuterPhi, kLoopBound);
  hdrB.CreateCondBr(kCond, loopBodyBB, afterLoopBB);

  // The fills split the block at a terminator; this branch ends in the latch.
  IRBuilder<> bodyB(loopBodyBB);
  Instruction *placeholderBr = bodyB.CreateBr(loopHdrBB);
  bodyB.SetInsertPoint(placeholderBr);
  Value *kOuter64 = bodyB.CreateZExt(kOuterPhi, i64Ty);

  auto [tlin, bsz] = emitThreadLinAndBlockSize(bodyB, mod, h.is2DBlock);
  Value *aKOff =
      bodyB.CreateMul(kOuter64, ConstantInt::get(i64Ty, h.aStrideByte));
  Value *aSrcForIter = bodyB.CreateGEP(i8Ty, aBlockBase, aKOff);
  Value *bKOff =
      bodyB.CreateMul(kOuter64, ConstantInt::get(i64Ty, bSrcKStepByte));
  Value *bSrcForIter = bodyB.CreateGEP(i8Ty, bBlockBase, bKOff);

  // Valid source extents. K-streaming makes the last K-tile short whenever
  // padK > 0, so the valid K count is the runtime min(tileK, K - k_outer).
  Value *validK = ConstantInt::get(i32Ty, (int32_t)t.tileK);
  if (t.padK > 0) {
    Value *kLeft =
        bodyB.CreateSub(ConstantInt::get(i32Ty, (int32_t)m.K), kOuterPhi);
    validK = bodyB.CreateBinaryIntrinsic(
        Intrinsic::umin, kLeft, ConstantInt::get(i32Ty, (int32_t)t.tileK));
  }
  Value *validM = ConstantInt::get(i32Ty, (int32_t)m.M);
  Value *validN = ConstantInt::get(i32Ty, (int32_t)m.N);

  // The fills cover the padded tile and write an exact zero outside the real
  // extent: guard-masking the mma instead would be wrong, since the fragments
  // are warp-collective, and an uninitialized pad feeds 0 * Inf = NaN into
  // real outputs. Without padding the direct fill checks no extent.
  const bool padded = (t.padM > 0 || t.padN > 0 || t.padK > 0);
  const bool rangeChecked = padded || split;
  const std::string fillName =
      (tag + (rangeChecked ? "_pfill" : "_cfill")).str();
  auto payload = [&](IRBuilder<> &PB, Value *x) -> SmallVector<Value *, 8> {
    if (split) {
      Value *xF32 = emitFPCast(PB, x, f32Ty);
      return emitNwayVeltkampSplit(PB, mod, xF32, N, operand);
    }
    Value *cvt = emitFPCast(PB, x, opTy);
    // wmma.load.a/b.tf32 reads the float bit pattern as it is and the MMA
    // truncates the low 13 mantissa bits, which biases the result one way, so
    // the fill rounds to nearest first.
    if (operand == FPKind::TF32)
      cvt = nvptx::emitRoundToTF32(PB, mod, cvt);
    return {cvt};
  };
  // The block is the profiled M x N one, so a fill of at most M * N elements
  // takes one element per thread.
  const uint64_t blockArea = (uint64_t)m.M * (uint64_t)m.N;
  TileFill aFill{aSrcForIter,
                 h.aLeadingDimByte,
                 h.aStrideByte,
                 aGen,
                 aScratchRowByte,
                 opByte,
                 mPad,
                 t.tileK,
                 rangeChecked ? validM : nullptr,
                 rangeChecked ? validK : nullptr};
  TileFill bFill{bSrcForIter,
                 bSrcKStepByte,
                 bSrcNStrideByte,
                 bGen,
                 bScratchKStepByte,
                 opByte,
                 t.tileK,
                 nPad,
                 rangeChecked ? validK : nullptr,
                 rangeChecked ? validN : nullptr};
  for (const TileFill *fill : {&aFill, &bFill})
    emitTileFill(bodyB, fillName, *fill, srcTy, opTy, tlin, bsz,
                 fill->rows * fill->cols <= blockArea, payload);
  bodyB.CreateCall(barrier.Fn, barrier.Args);

  // Warp-gate the mma chain when there is a single output tile per block, and
  // always when the block's thread count is not a multiple of 32: its last
  // warp is partially populated, and warp-collective wmma ops in a partial
  // warp are undefined (the ungated chain corrupts the shared output tile).
  const bool partialWarpBlock =
      (h.blockThreads != 0) && (h.blockThreads % nvptx::kWarpSize != 0);
  const bool useWarpGate = (t.mChain * t.nChain == 1) || partialWarpBlock;
  auto emitIsWarp0 = [&](IRBuilder<> &GB, Value *lin) {
    Value *warpId = GB.CreateLShr(
        lin, ConstantInt::get(i32Ty, nvptx::kLog2WarpSize), "warp_id");
    return GB.CreateICmpEQ(warpId, ConstantInt::get(i32Ty, 0), "is_warp0");
  };
  BasicBlock *gatePreBB = nullptr;
  BasicBlock *gateMergeBB = nullptr;
  if (useWarpGate) {
    Value *isWarp0 = emitIsWarp0(bodyB, tlin);
    gatePreBB = bodyB.GetInsertBlock();
    Instruction *splitPt = &*bodyB.GetInsertPoint();
    gateMergeBB =
        gatePreBB->splitBasicBlock(splitPt->getIterator(), tag + "_gate.merge");
    gatePreBB->getTerminator()->eraseFromParent();
    BasicBlock *gateDoBB =
        BasicBlock::Create(ctx, tag + "_gate.do", F, gateMergeBB);
    IRBuilder<>(gatePreBB).CreateCondBr(isWarp0, gateDoBB, gateMergeBB);
    bodyB.SetInsertPoint(gateDoBB);
  }

  auto callMma = [&](Value *aTile, Value *bTile,
                     ArrayRef<Value *> cIn) -> Value * {
    Value *aFragV = bodyB.CreateCall(wmma.LoadA, {aTile, ldaV});
    Value *bFragV = bodyB.CreateCall(wmma.LoadB, {bTile, ldbV});
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
    return bodyB.CreateCall(wmma.Mma, args);
  };
  auto fragElement = [&](Value *frag, unsigned e) {
    return cFragSize == 1 ? frag : bodyB.CreateExtractValue(frag, e);
  };

  SmallVector<SmallVector<Value *, 16>, 8> accLoop(N);
  for (unsigned k = 0; k < N; ++k)
    accLoop[k].resize(numPhisPerGroup);
  for (unsigned mt = 0; mt < t.mChain; ++mt) {
    for (unsigned nt = 0; nt < t.nChain; ++nt) {
      // Scratch is refilled at offset 0 per iteration, so the wmma loads read
      // from the scratch base with no K offset.
      SmallVector<Value *, 8> aRowBases(N), bColBases(N);
      for (unsigned k = 0; k < N; ++k) {
        aRowBases[k] = bodyB.CreatePtrAdd(
            aGen[k],
            ConstantInt::get(i64Ty, (int64_t)mt * t.tileM * aScratchRowByte));
        bColBases[k] = bodyB.CreatePtrAdd(
            bGen[k], ConstantInt::get(i64Ty, (int64_t)nt * t.tileN * opByte));
      }
      unsigned phiBase = (mt * t.nChain + nt) * cFragSize;
      SmallVector<SmallVector<Value *, 16>, 8> cur(N);
      for (unsigned k = 0; k < N; ++k) {
        cur[k].resize(cFragSize);
        for (unsigned i = 0; i < cFragSize; ++i)
          cur[k][i] = accPhis[k][phiBase + i];
      }

      if (split) {
        // For each (i, j) with i + j < N accumulate a[i]*b[j] into group
        // c[i+j]; products with i + j >= N are below the captured resolution.
        // The dominant c[0] uses fresh acc + add rather than a chained mma:
        // the K-stream reduction stays outside the wmma's internal
        // K-reduction, and the order matters in the F32 sum.
        SmallVector<Value *, 16> zeroAcc(cFragSize, cZeroElt);
        for (unsigned i = 0; i < N; ++i) {
          for (unsigned j = 0; j + i < N; ++j) {
            unsigned kg = i + j;
            if (i == 0 && j == 0) {
              Value *tmpV = callMma(aRowBases[0], bColBases[0], zeroAcc);
              for (unsigned e = 0; e < cFragSize; ++e) {
                Value *tmpElt = fragElement(tmpV, e);
                cur[0][e] = bodyB.CreateFAdd(cur[0][e], tmpElt);
              }
              continue;
            }
            Value *dV = callMma(aRowBases[i], bColBases[j], cur[kg]);
            for (unsigned e = 0; e < cFragSize; ++e)
              cur[kg][e] = fragElement(dV, e);
          }
        }
      } else {
        Value *dV = callMma(aRowBases[0], bColBases[0], cur[0]);
        for (unsigned e = 0; e < cFragSize; ++e)
          cur[0][e] = fragElement(dV, e);
      }

      for (unsigned k = 0; k < N; ++k)
        for (unsigned i = 0; i < cFragSize; ++i)
          accLoop[k][phiBase + i] = cur[k][i];
    }
  }

  // Close the warp gate: merge the mma results (warp 0) with the unchanged
  // accumulators (other warps) so the back edge uses the merged values.
  if (useWarpGate) {
    BasicBlock *gateEndBB = bodyB.GetInsertBlock();
    bodyB.CreateBr(gateMergeBB);
    IRBuilder<> mergeB =
        builderAt(gateMergeBB, gateMergeBB->getFirstInsertionPt());
    for (unsigned k = 0; k < N; ++k) {
      for (unsigned i = 0; i < numPhisPerGroup; ++i) {
        PHINode *p = mergeB.CreatePHI(cFragElemTy, 2, accName(k) + ".merge");
        p->addIncoming(accLoop[k][i], gateEndBB);
        p->addIncoming(accPhis[k][i], gatePreBB);
        accLoop[k][i] = p;
      }
    }
    setInsertPoint(bodyB, gateMergeBB, gateMergeBB->getFirstInsertionPt());
  }

  // Barrier after the wmma reads so the next iteration's fill doesn't race
  // them.
  bodyB.CreateCall(barrier.Fn, barrier.Args);

  Value *kNext =
      bodyB.CreateAdd(kOuterPhi, ConstantInt::get(i32Ty, (int32_t)t.tileK));
  BasicBlock *latchBB = placeholderBr->getParent();
  kOuterPhi->addIncoming(kNext, latchBB);
  for (unsigned k = 0; k < N; ++k)
    for (unsigned i = 0; i < numPhisPerGroup; ++i)
      accPhis[k][i]->addIncoming(accLoop[k][i], latchBB);

  // Store the final accumulators to the output scratch, warp-gated too: only
  // warp 0's PHIs hold the result, and the barrier after makes the store
  // visible to the readback in exitBB.
  IRBuilder<> afterB = builderAt(afterLoopBB, afterLoopBB->getFirstNonPHIIt());
  BasicBlock *storeMergeBB = nullptr;
  if (useWarpGate) {
    Value *afterTlin =
        emitThreadLinAndBlockSize(afterB, mod, h.is2DBlock).first;
    Value *afterIsWarp0 = emitIsWarp0(afterB, afterTlin);
    Instruction *splitPt = &*afterB.GetInsertPoint();
    BasicBlock *curr = afterB.GetInsertBlock();
    storeMergeBB = curr->splitBasicBlock(splitPt->getIterator(),
                                         tag + "_store_gate.merge");
    curr->getTerminator()->eraseFromParent();
    BasicBlock *storeDoBB =
        BasicBlock::Create(ctx, tag + "_store_gate.do", F, storeMergeBB);
    IRBuilder<>(curr).CreateCondBr(afterIsWarp0, storeDoBB, storeMergeBB);
    afterB.SetInsertPoint(storeDoBB);
  }
  for (unsigned mt = 0; mt < t.mChain; ++mt) {
    for (unsigned nt = 0; nt < t.nChain; ++nt) {
      Value *dTileByte =
          ConstantInt::get(i64Ty, ((int64_t)mt * t.tileM * (int64_t)nPad +
                                   (int64_t)nt * t.tileN) *
                                      accByte);
      unsigned phiBase = (mt * t.nChain + nt) * cFragSize;
      for (unsigned k = 0; k < N; ++k) {
        Value *dTilePtr = afterB.CreatePtrAdd(dGen[k], dTileByte);
        SmallVector<Value *, 16> args;
        args.reserve(2 + cFragSize);
        args.push_back(dTilePtr);
        for (unsigned i = 0; i < cFragSize; ++i)
          args.push_back(accPhis[k][phiBase + i]);
        args.push_back(ldcV);
        afterB.CreateCall(wmma.StoreD, args);
      }
    }
  }
  if (useWarpGate) {
    afterB.CreateBr(storeMergeBB);
    setInsertPoint(afterB, storeMergeBB, storeMergeBB->getFirstInsertionPt());
  }
  afterB.CreateCall(barrier.Fn, barrier.Args);

  // Readback of this thread's cell, sum_k d[k] / SCALE^k: the block is the
  // profiled M x N one, so it is not clamped to the output tile, and the
  // output scratch's row stride is nPad. It ends the raise, which then leaves
  // to the exit block past the loop.
  IRBuilder<> &RB = afterB;
  Value *rRow =
      emitTidIndex(RB, mod, h.aRowAxis, h.aRowSlowAxis, h.aRowFuseMult);
  Value *rCol =
      emitTidIndex(RB, mod, h.bColAxis, h.bColSlowAxis, h.bColFuseMult);
  Value *rRowOff = RB.CreateMul(rRow, ConstantInt::get(i32Ty, nPad));
  Value *rCellIdx = RB.CreateAdd(rRowOff, rCol);
  Value *sum = nullptr;
  double scalePowK = 1.0;
  for (unsigned k = 0; k < N; ++k) {
    Value *cell = RB.CreateGEP(accTy, dGen[k], rCellIdx);
    LoadInst *raw = RB.CreateLoad(accTy, cell);
    raw->setAlignment(Align(accByte));
    if (k == 0) {
      sum = raw;
    } else {
      Value *term = RB.CreateFDiv(raw, ConstantFP::get(accTy, scalePowK));
      sum = RB.CreateFAdd(sum, term);
    }
    if (k + 1 < N)
      scalePowK *= veltkampScale(operand);
  }
  Value *readback = emitFPCast(RB, sum, srcTy);
  BasicBlock *raiseEnd = RB.GetInsertBlock();
  Instruction *toLoop = raiseEnd->getTerminator();
  IRBuilder<>(toLoop).CreateBr(exitBB);
  toLoop->eraseFromParent();

  PHINode *merged = nullptr;
  SmallVector<Use *, 8> usesToRewrite;
  for (Use &U : h.fma->uses()) {
    auto *userI = dyn_cast<Instruction>(U.getUser());
    if (!userI || h.blocks.contains(userI->getParent()))
      continue;
    usesToRewrite.push_back(&U);
  }
  for (Use *U : usesToRewrite) {
    auto *userI = cast<Instruction>(U->getUser());
    if (auto *phi = dyn_cast<PHINode>(userI); phi && phi->getParent() == exitBB)
      continue;
    if (!merged) {
      merged =
          PHINode::Create(h.fma->getType(), 2, "raise.result", exitBB->begin());
      merged->addIncoming(h.fma, loopLatch);
      merged->addIncoming(readback, raiseEnd);
    }
    U->set(merged);
  }
  for (PHINode &phi : exitBB->phis())
    if (&phi != merged)
      phi.addIncoming(phi.getIncomingValueForBlock(loopLatch) == h.fma
                          ? readback
                          : phi.getIncomingValueForBlock(loopLatch),
                      raiseEnd);
  // The reduction loop is left to dead-code elimination.
}

} // namespace poseidon
