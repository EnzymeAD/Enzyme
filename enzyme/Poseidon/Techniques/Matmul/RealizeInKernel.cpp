//===- RealizeInKernel.cpp - tensor-core code for a scalar-loop product ---===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
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
// accumulator per output tile. The split scheme fills N slices of each
// operand (MatmulInternal.h), accumulates a[i] * b[j] into group i + j for
// i + j < N, and reads back sum_k d[k] / SCALE^k.
//
//===----------------------------------------------------------------------===//

#include "Core/NumberFormat.h"
#include "Enzyme/Utils.h"
#include "MatmulInternal.h"
#include "Support/Utils.h"
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
#include "llvm/IR/IntrinsicsNVPTX.h"
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

// The NVPTX instructions the raised code is built from: the WMMA fragment
// operations, the block barrier, the TF32 rounding and the thread geometry.

// The threads that execute one WMMA fragment operation together.
static constexpr unsigned kWarpSize = 32;
// log2 of kWarpSize.
static constexpr unsigned kLog2WarpSize = 5;
static_assert(1u << kLog2WarpSize == kWarpSize);

// The launch value `kind` on `axis` (0 to 2), read at `B`'s insertion point.
static Value *emitLaunchValue(IRBuilderBase &B, Module *M, LaunchValue kind,
                              unsigned axis) {
  return Target::of(*M).emitLaunchValue(B, kind, axis);
}

namespace {
// The barrier of a thread block (__syncthreads) and its operands.
struct Barrier {
  // The barrier intrinsic, declared in the module.
  Function *Fn;
  // The operands a call of `Fn` takes.
  SmallVector<Value *, 1> Args;
};
} // namespace

// The block barrier, declared in `M`.
static Barrier declareBarrier(Module *M) {
  return {Intrinsic::getOrInsertDeclaration(
              M, Intrinsic::nvvm_barrier_cta_sync_aligned_all),
          {ConstantInt::get(Type::getInt32Ty(M->getContext()), 0)}};
}

// `x`, an f32, rounded to nearest to TF32 and kept in f32 storage, which the
// TF32 fragment loads read as it is.
static Value *emitRoundToTF32(IRBuilderBase &B, Module *M, Value *x) {
  // f2tf32.rna gives the i32 bit pattern.
  Function *f2tf32 =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::nvvm_f2tf32_rna);
  return B.CreateBitCast(B.CreateCall(f2tf32, {x}), x->getType());
}

namespace {
// The fragment operations of one m x n x k shape, through generic pointers, of
// row-major A, B and D. `operand` and `accumulator`
// name the element types as the instructions spell them (f16, bf16, tf32,
// f64; f16, f32, f64).
struct Wmma {
  // The fragment loads of A and B, the multiply-accumulate, and the store of
  // D, declared in the module.
  Function *LoadA, *LoadB, *Mma, *StoreD;
};
} // namespace

static Intrinsic::ID resolveWmmaIntrinsic(const Twine &name) {
  Intrinsic::ID id = Intrinsic::lookupIntrinsicID(name.str());
  if (id == Intrinsic::not_intrinsic)
    report_fatal_error("Poseidon: NVPTX has no intrinsic " + name);
  return id;
}

// The fragment operations of the `m` x `n` x `k` shape, declared in `M`. A
// shape and type combination NVPTX has no intrinsic for is a fatal error.
static Wmma declareWmma(Module *M, unsigned m, unsigned n, unsigned k,
                        StringRef operand, StringRef accumulator) {
  const std::string shape =
      ("m" + Twine(m) + "n" + Twine(n) + "k" + Twine(k)).str();
  const std::string a = operand.str(), d = accumulator.str();
  // An F16 product names its C/D precision pair; the other operand types
  // imply theirs and name only themselves.
  const std::string mmaSuffix = a == "f16" ? d + "." + d : a;
  Intrinsic::ID loadA = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                             ".load.a.row.stride." + a);
  Intrinsic::ID loadB = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                             ".load.b.row.stride." + a);
  Intrinsic::ID mma = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                           ".mma.row.row." + mmaSuffix);
  Intrinsic::ID storeD = resolveWmmaIntrinsic("llvm.nvvm.wmma." + shape +
                                              ".store.d.row.stride." + d);
  PointerType *genPtrTy = getInt8PtrTy(M->getContext());
  return {Intrinsic::getOrInsertDeclaration(M, loadA, {genPtrTy}),
          Intrinsic::getOrInsertDeclaration(M, loadB, {genPtrTy}),
          Intrinsic::getOrInsertDeclaration(M, mma),
          Intrinsic::getOrInsertDeclaration(M, storeD, {genPtrTy})};
}

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

// The thread index along `axis`, which must be x, y or z.
static Value *emitTidIntrinsic(IRBuilder<> &B, Module *M, TidAxis axis) {
  std::optional<unsigned> a = tidAxisIndex(axis);
  assert(a && "a raised product's indices run along thread axes");
  return emitLaunchValue(B, M, LaunchValue::ThreadIndex, *a);
}

// The thread's value of a matrix index.
static Value *emitTidIndex(IRBuilder<> &B, Module *M, const MatrixIndex &idx) {
  Value *fastV = emitTidIntrinsic(B, M, idx.fast);
  if (!idx.fused())
    return fastV;
  assert(idx.fuseMult && "a fused index has the fast axis's extent");
  Value *slowV = emitTidIntrinsic(B, M, idx.slow);
  Type *i32Ty = Type::getInt32Ty(B.getContext());
  Value *slowOff =
      B.CreateMul(slowV, ConstantInt::get(i32Ty, (int32_t)idx.fuseMult));
  return B.CreateAdd(fastV, slowOff, "wmma.fused_idx");
}

// (linearizedThreadId, blockSize) of a block of the shape `block`, which is
// the launch's wherever the raise runs: its extents are constants, so the
// raise's index arithmetic does not depend on what the compiler can prove
// about the launch from the guard. `is2DBlock` skips the tid.z read; only set
// it when the block's z extent is 1.
static std::pair<Value *, Value *>
emitThreadLinAndBlockSize(IRBuilder<> &B, Module *M, const uint32_t (&block)[3],
                          bool is2DBlock) {
  Type *i32Ty = B.getInt32Ty();
  Value *tx = emitLaunchValue(B, M, LaunchValue::ThreadIndex, 0);
  Value *ty = emitLaunchValue(B, M, LaunchValue::ThreadIndex, 1);
  Value *nx = ConstantInt::get(i32Ty, block[0]);
  Value *bsz = ConstantInt::get(i32Ty, block[0] * block[1] * block[2]);
  Value *planeRow = ty;
  if (!is2DBlock) {
    Value *tz = emitLaunchValue(B, M, LaunchValue::ThreadIndex, 2);
    planeRow =
        B.CreateAdd(B.CreateMul(tz, ConstantInt::get(i32Ty, block[1])), ty);
  }
  return {B.CreateAdd(B.CreateMul(planeRow, nx), tx), bsz};
}

static double sliceScale(NumberFormat slice) {
  return std::ldexp(1.0, sliceFormat(slice).scaleExponent);
}

// Quantize an F32 value to slice precision (half for F16/BF16; float for TF32,
// bit-identical to F32 with the bottom 13 mantissa bits zeroed).
static Value *emitSliceQuantize(IRBuilder<> &B, Module *mod, Value *vs_f32,
                                NumberFormat slice) {
  if (slice == NumberFormat::FP16 || slice == NumberFormat::BF16)
    return B.CreateFPTrunc(vs_f32, storageTypeOf(B.getContext(), slice));
  return emitRoundToTF32(B, mod, vs_f32);
}

// Slice value back to F32 for the next residual; TF32 is already F32 storage.
static Value *emitSliceToF32(IRBuilder<> &B, Value *slice, NumberFormat kind) {
  Type *f32Ty = Type::getFloatTy(B.getContext());
  if (kind == NumberFormat::TF32)
    return slice; // bit-pattern already in float
  return B.CreateFPExt(slice, f32Ty);
}

// N-slice split by rounding to the slice format (not Veltkamp's split):
//   s[0]  = quantize(v)
//   r[k]  = (r[k-1] - s[k-1]_f32) * SCALE ; s[k] = quantize(r[k])
// Reconstruction: v ~ sum_k s[k] / SCALE^k, each slice capturing the
// significand bits of its format (sliceFormat).
static SmallVector<Value *, 8> emitSliceSplit(IRBuilder<> &B, Module *mod,
                                              Value *vs_f32, unsigned N,
                                              NumberFormat slice) {
  Type *f32Ty = Type::getFloatTy(B.getContext());
  Value *scale = ConstantFP::get(f32Ty, sliceScale(slice));
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

// Branches from `preheader` to a new block, which it returns, when the launch
// has the profiled block shape and every thread of the block passes the
// loop's guards, and to `loopHeader` otherwise. Both tests read only the
// block's index and shape, so the branch is uniform across the block.
static BasicBlock *emitRaiseGuard(BasicBlock &preheader, BasicBlock &loopHeader,
                                  const ScalarLoopHandle &h) {
  Function *F = preheader.getParent();
  Module *mod = F->getParent();
  IRBuilder<> GB(preheader.getTerminator());
  Value *raise = nullptr;
  for (unsigned axis = 0; axis < 3; ++axis) {
    Value *eq = GB.CreateICmpEQ(
        emitLaunchValue(GB, mod, LaunchValue::BlockDim, axis),
        ConstantInt::get(GB.getInt32Ty(), h.profiledBlock[axis]));
    raise = raise ? GB.CreateAnd(raise, eq) : eq;
  }
  // Every thread of the block passes the guard on the axis: its last global
  // index is below the bound, (blockIdx + 1) * blockDim <= ub. The block has
  // the profiled shape wherever the raise runs, so this is
  // blockIdx < ub / blockDim with that shape's extent, one compare that
  // cannot wrap.
  for (unsigned axis = 0; axis < 3; ++axis) {
    const uint64_t ub = h.loopGuard.ub[axis];
    if (!ub)
      continue;
    assert(h.profiledBlock[axis] && "a profiled block has every extent");
    const uint64_t blocks = ub / h.profiledBlock[axis];
    Value *inside = GB.CreateICmpULT(
        emitLaunchValue(GB, mod, LaunchValue::BlockIndex, axis),
        ConstantInt::get(GB.getInt32Ty(), blocks), "raise.inside");
    raise = GB.CreateAnd(raise, inside);
  }
  BasicBlock *raiseBB =
      BasicBlock::Create(mod->getContext(), "raise.entry", F, &loopHeader);
  IRBuilder<>(raiseBB).CreateBr(&loopHeader);
  GB.CreateCondBr(raise, raiseBB, &loopHeader);
  preheader.getTerminator()->eraseFromParent();
  return raiseBB;
}

namespace {
// What every step of one in-kernel raise reads.
struct RaiseContext {
  const RecognizedMatmul &m;
  const MatmulCandidate &t;
  const ScalarLoopHandle &h;
  InKernelLayout L;
  Function *F;
  LLVMContext &ctx;
  Module *mod;
  bool split;
  // Operand slices, and accumulator groups: one each in the direct scheme.
  unsigned N;
  Type *i8Ty, *i32Ty, *i64Ty, *f32Ty;
  PointerType *genPtrTy;
  // The scratch element type matches the wmma fragment's element view; TF32
  // wmma loads bit-cast F32 storage into i32 fragments, so F32 is stored after
  // explicit rounding to TF32.
  Type *srcTy, *opTy, *accTy;
  int64_t srcByte, opByte, accByte;
  // B layout from the captured reduction-axis stride: a one-element K stride
  // means K-contiguous (col-major), otherwise K-strided (row-major). B always
  // routes through scratch, so this only sets the fill's read strides.
  bool bRowMajor;
  Wmma wmma;
  unsigned aFragSize, bFragSize, cFragSize;
  Type *cFragElemTy;
  // Warp-gate the mma chain when there is a single output tile per block, and
  // always when the block's thread count is not a multiple of 32: its last
  // warp is partially populated, and warp-collective wmma ops in a partial
  // warp are undefined (the ungated chain corrupts the shared output tile).
  bool useWarpGate;
  // Value names: the split scheme's start with "tcec", the direct scheme's
  // with "wmma".
  StringRef tag;
  // The generic pointers to each slice's A, B and D scratch.
  SmallVector<Value *, 8> aGen, bGen, dGen;

  std::string accName(unsigned k) const {
    return split ? ("cAcc" + Twine(k)).str() : std::string("cFrag");
  }
  unsigned accPhisPerGroup() const { return L.mTiles * L.nTiles * cFragSize; }
  // The scratch tiles are tight-packed at (mPad x tileK) / (tileK x nPad).
  int64_t aScratchRowByte() const { return (int64_t)t.tileK * opByte; }
};

// The K-streamed loop the raise runs in: its header PHIs, the block after it,
// and the block-uniform thread index the fills and the warp gate use.
struct KStreamLoop {
  BasicBlock *afterLoopBB;
  PHINode *kOuterPhi;
  // Per accumulator group, one PHI per (mt, nt, fragment element).
  SmallVector<SmallVector<PHINode *, 16>, 8> accPhis;
  // The back edge; the body is emitted in front of it.
  Instruction *latchBr;
  Value *tlin;
};
} // namespace

static unsigned fragSize(Type *T) {
  if (auto *ST = dyn_cast<StructType>(T))
    return ST->getNumElements();
  return 1;
}

static RaiseContext makeRaiseContext(const RecognizedMatmul &m,
                                     const MatmulCandidate &t, Function *F) {
  const SchemeVariant &v = t.variant;
  Module *mod = F->getParent();
  LLVMContext &ctx = mod->getContext();
  const DataLayout &DL = mod->getDataLayout();
  RaiseContext C{m, t, m.scalarLoop(), inKernelLayout(m, t), F, ctx};
  C.split = v.scheme == Scheme::Tcec;
  C.N = C.split ? sliceCount(v) : 1;
  C.mod = mod;
  C.i8Ty = Type::getInt8Ty(ctx);
  C.i32Ty = Type::getInt32Ty(ctx);
  C.i64Ty = Type::getInt64Ty(ctx);
  C.f32Ty = Type::getFloatTy(ctx);
  C.genPtrTy = getInt8PtrTy(ctx);
  C.srcTy = storageTypeOf(ctx, m.elemType);
  C.opTy = storageTypeOf(ctx, v.operand);
  C.accTy = storageTypeOf(ctx, v.accumulator);
  C.tag = C.split ? "tcec" : "wmma";
  C.srcByte = (int64_t)DL.getTypeAllocSize(C.srcTy).getFixedValue();
  C.opByte = (int64_t)DL.getTypeAllocSize(C.opTy).getFixedValue();
  C.accByte = (int64_t)DL.getTypeAllocSize(C.accTy).getFixedValue();
  const ScalarLoopHandle &h = C.h;
  assert(h.bStrideByte % C.srcByte == 0 &&
         "B steps along the reduction by whole elements");
  C.bRowMajor = h.bStrideByte != C.srcByte;

  C.wmma = declareWmma(mod, t.tileM, t.tileN, t.tileK, dtypeName(v.operand),
                       dtypeName(v.accumulator));
  FunctionType *mmaFTy = C.wmma.Mma->getFunctionType();
  C.aFragSize = fragSize(C.wmma.LoadA->getFunctionType()->getReturnType());
  C.bFragSize = fragSize(C.wmma.LoadB->getFunctionType()->getReturnType());
  C.cFragSize = mmaFTy->getNumParams() - C.aFragSize - C.bFragSize;
  assert(C.cFragSize == fragSize(mmaFTy->getReturnType()) &&
         "chained accumulation needs the C and D fragments of one shape");
  C.cFragElemTy = mmaFTy->getParamType(C.aFragSize + C.bFragSize);

  const bool partialWarpBlock =
      h.blockThreads() != 0 && h.blockThreads() % kWarpSize != 0;
  C.useWarpGate = C.L.mTiles * C.L.nTiles == 1 || partialWarpBlock;
  return C;
}

// The per-slice A (mPad x tileK) and B (tileK x nPad) scratch of the operand
// format and D (mPad x nPad) of the accumulator format; the K-streamed sizing
// keeps shared memory bounded regardless of K. Keyed by (function, op, slice,
// format, extents) and not by matmul id, so raises in one function that need
// an identically shaped buffer share one (per-raise buffers would exceed the
// 48 KB static cap). Safe because the raised regions are sequential and
// bracketed by block barriers; the extents in the key prevent reuse at another
// size.
static void createScratch(const RaiseContext &C,
                          SmallVectorImpl<GlobalVariable *> &a,
                          SmallVectorImpl<GlobalVariable *> &b,
                          SmallVectorImpl<GlobalVariable *> &d) {
  const SchemeVariant &v = C.t.variant;
  auto scratchName = [&](StringRef op, unsigned k, StringRef prec,
                         uint64_t rows, uint64_t cols) -> std::string {
    std::string slice = C.split ? std::to_string(k) : std::string();
    return ("__poseidon_" + Twine(C.split ? "tcec" : "raise") + "_" +
            C.F->getName() + "_" + op + slice + "_" + prec + "_" + Twine(rows) +
            "x" + Twine(cols))
        .str();
  };
  for (unsigned k = 0; k < C.N; ++k) {
    a.push_back(getOrCreateSharedScratch(
        C.mod, scratchName("a", k, dtypeName(v.operand), C.L.mPad, C.t.tileK),
        C.opTy, C.L.aElts()));
    b.push_back(getOrCreateSharedScratch(
        C.mod, scratchName("b", k, dtypeName(v.operand), C.t.tileK, C.L.nPad),
        C.opTy, C.L.bElts()));
    d.push_back(getOrCreateSharedScratch(
        C.mod,
        scratchName("d", k, dtypeName(v.accumulator), C.L.mPad, C.L.nPad),
        C.accTy, C.L.dElts()));
  }
}

// Per-block source bases of A and B: the per-thread iteration-0 start less the
// threadIdx contribution of the (possibly fused) index the recognizer matched,
// so the cooperative fill sees one base across the block.
static std::pair<Value *, Value *> emitBlockBases(const RaiseContext &C,
                                                  IRBuilder<> &B) {
  const ScalarLoopHandle &h = C.h;
  assert(h.aStartSCEV && h.bStartSCEV && h.SE &&
         "recognition records both operands' start");
  SCEVExpander sex = makeSCEVExpander(*h.SE, "poseidon.raise");
  Instruction *at = &*B.GetInsertPoint();
  Value *rowTid = emitTidIndex(B, C.mod, h.aRow);
  Value *colTid = emitTidIndex(B, C.mod, h.bCol);
  Value *rowTidI64 = B.CreateZExt(rowTid, C.i64Ty);
  Value *colTidI64 = B.CreateZExt(colTid, C.i64Ty);
  Value *aPerThreadStart = sex.expandCodeFor(h.aStartSCEV, C.genPtrTy, at);
  Value *bPerThreadStart = sex.expandCodeFor(h.bStartSCEV, C.genPtrTy, at);
  Value *aRowThreadOff =
      B.CreateMul(rowTidI64, ConstantInt::get(C.i64Ty, h.aLeadingDimByte));
  Value *aRowThreadBack = B.CreateNeg(aRowThreadOff);
  Value *aBlockBase = B.CreateGEP(C.i8Ty, aPerThreadStart, aRowThreadBack);
  // The recognizer admits a fused index only when it steps the operand by
  // these strides.
  int64_t bColThreadStrideByte = C.bRowMajor ? C.srcByte : h.bLeadingDimByte;
  Value *bColThreadOff =
      B.CreateMul(colTidI64, ConstantInt::get(C.i64Ty, bColThreadStrideByte));
  Value *bColThreadBack = B.CreateNeg(bColThreadOff);
  Value *bBlockBase = B.CreateGEP(C.i8Ty, bPerThreadStart, bColThreadBack);
  return {aBlockBase, bBlockBase};
}

// The K-streaming loop around the mma chain: per k_outer, refill the A/B
// scratch with the next K-tile chunk and barrier. The accumulators are carried
// in header PHIs. Leaves `bodyB` after the fills' barrier.
static KStreamLoop emitKStreamLoop(const RaiseContext &C, IRBuilder<> &B,
                                   IRBuilder<> &bodyB, Value *aBlockBase,
                                   Value *bBlockBase, const Barrier &barrier) {
  const RecognizedMatmul &m = C.m;
  const ScalarLoopHandle &h = C.h;
  const unsigned tileK = C.t.tileK;
  KStreamLoop K;
  BasicBlock *preheaderBB = B.GetInsertBlock();
  Instruction *postLoopInst = &*B.GetInsertPoint();
  K.afterLoopBB = preheaderBB->splitBasicBlock(postLoopInst->getIterator(),
                                               C.tag + "_kstream.after");
  preheaderBB->getTerminator()->eraseFromParent();
  BasicBlock *loopHdrBB =
      BasicBlock::Create(C.ctx, C.tag + "_kstream.hdr", C.F, K.afterLoopBB);
  BasicBlock *loopBodyBB =
      BasicBlock::Create(C.ctx, C.tag + "_kstream.body", C.F, K.afterLoopBB);
  IRBuilder<>(preheaderBB).CreateBr(loopHdrBB);

  // Header PHIs: k_outer plus one PHI per (group, mt, nt, fragment element).
  IRBuilder<> hdrB(loopHdrBB);
  K.kOuterPhi = hdrB.CreatePHI(C.i32Ty, 2, "k_outer");
  K.kOuterPhi->addIncoming(ConstantInt::get(C.i32Ty, 0), preheaderBB);
  Constant *cZeroElt = Constant::getNullValue(C.cFragElemTy);
  K.accPhis.resize(C.N);
  for (unsigned k = 0; k < C.N; ++k) {
    K.accPhis[k].resize(C.accPhisPerGroup());
    for (PHINode *&phi : K.accPhis[k]) {
      phi = hdrB.CreatePHI(C.cFragElemTy, 2, C.accName(k));
      phi->addIncoming(cZeroElt, preheaderBB);
    }
  }
  Value *kLoopBound = ConstantInt::get(C.i32Ty, (int32_t)C.L.kPad);
  Value *kCond = hdrB.CreateICmpULT(K.kOuterPhi, kLoopBound);
  hdrB.CreateCondBr(kCond, loopBodyBB, K.afterLoopBB);

  // The fills split the block at a terminator; this branch ends in the latch.
  bodyB.SetInsertPoint(loopBodyBB);
  K.latchBr = bodyB.CreateBr(loopHdrBB);
  bodyB.SetInsertPoint(K.latchBr);
  Value *kOuter64 = bodyB.CreateZExt(K.kOuterPhi, C.i64Ty);

  auto [tlin, bsz] =
      emitThreadLinAndBlockSize(bodyB, C.mod, h.profiledBlock, h.is2DBlock());
  K.tlin = tlin;
  const int64_t bSrcKStepByte = C.bRowMajor ? h.bStrideByte : C.srcByte;
  const int64_t bSrcNStrideByte = C.bRowMajor ? C.srcByte : h.bLeadingDimByte;
  Value *aKOff =
      bodyB.CreateMul(kOuter64, ConstantInt::get(C.i64Ty, h.aStrideByte));
  Value *aSrcForIter = bodyB.CreateGEP(C.i8Ty, aBlockBase, aKOff);
  Value *bKOff =
      bodyB.CreateMul(kOuter64, ConstantInt::get(C.i64Ty, bSrcKStepByte));
  Value *bSrcForIter = bodyB.CreateGEP(C.i8Ty, bBlockBase, bKOff);

  // Valid source extents. K-streaming makes the last K-tile short whenever
  // K is padded, so the valid K count is the runtime min(tileK, K - k_outer).
  Value *validK = ConstantInt::get(C.i32Ty, (int32_t)tileK);
  if (C.L.kPad > m.K) {
    Value *kLeft =
        bodyB.CreateSub(ConstantInt::get(C.i32Ty, (int32_t)m.K), K.kOuterPhi);
    validK = bodyB.CreateBinaryIntrinsic(
        Intrinsic::umin, kLeft, ConstantInt::get(C.i32Ty, (int32_t)tileK));
  }
  Value *validM = ConstantInt::get(C.i32Ty, (int32_t)m.M);
  Value *validN = ConstantInt::get(C.i32Ty, (int32_t)m.N);

  // The fills cover the padded tile and write an exact zero outside the real
  // extent: guard-masking the mma instead would be wrong, since the fragments
  // are warp-collective, and an uninitialized pad feeds 0 * Inf = NaN into
  // real outputs. Without padding the direct fill checks no extent.
  const bool rangeChecked = C.L.padded(m) || C.split;
  const std::string fillName =
      (C.tag + (rangeChecked ? "_pfill" : "_cfill")).str();
  const NumberFormat operand = C.t.variant.operand;
  auto payload = [&](IRBuilder<> &PB, Value *x) -> SmallVector<Value *, 8> {
    if (C.split) {
      Value *xF32 = emitFPCast(PB, x, C.f32Ty);
      return emitSliceSplit(PB, C.mod, xF32, C.N, operand);
    }
    Value *cvt = emitFPCast(PB, x, C.opTy);
    // wmma.load.a/b.tf32 reads the float bit pattern as it is and the MMA
    // truncates the low 13 mantissa bits, which biases the result one way, so
    // the fill rounds to nearest first.
    if (operand == NumberFormat::TF32)
      cvt = emitRoundToTF32(PB, C.mod, cvt);
    return {cvt};
  };
  // The block is the profiled M x N one, so a fill of at most M * N elements
  // takes one element per thread.
  const uint64_t blockArea = (uint64_t)m.M * (uint64_t)m.N;
  TileFill aFill{aSrcForIter,
                 h.aLeadingDimByte,
                 h.aStrideByte,
                 C.aGen,
                 C.aScratchRowByte(),
                 C.opByte,
                 C.L.mPad,
                 tileK,
                 rangeChecked ? validM : nullptr,
                 rangeChecked ? validK : nullptr};
  TileFill bFill{bSrcForIter,
                 bSrcKStepByte,
                 bSrcNStrideByte,
                 C.bGen,
                 (int64_t)C.L.nPad * C.opByte,
                 C.opByte,
                 tileK,
                 C.L.nPad,
                 rangeChecked ? validK : nullptr,
                 rangeChecked ? validN : nullptr};
  for (const TileFill *fill : {&aFill, &bFill})
    emitTileFill(bodyB, fillName, *fill, C.srcTy, C.opTy, tlin, bsz,
                 fill->rows * fill->cols <= blockArea, payload);
  bodyB.CreateCall(barrier.Fn, barrier.Args);
  return K;
}

static Value *emitIsWarp0(const RaiseContext &C, IRBuilder<> &GB, Value *lin) {
  Value *warpId =
      GB.CreateLShr(lin, ConstantInt::get(C.i32Ty, kLog2WarpSize), "warp_id");
  return GB.CreateICmpEQ(warpId, ConstantInt::get(C.i32Ty, 0), "is_warp0");
}

// Splits `B`'s block at its insertion point into a branch on warp 0 to a new
// block, where `B` is left, and the merge block, which it returns.
static BasicBlock *openWarpGate(const RaiseContext &C, IRBuilder<> &B,
                                Value *lin, const Twine &name) {
  Value *isWarp0 = emitIsWarp0(C, B, lin);
  BasicBlock *pre = B.GetInsertBlock();
  BasicBlock *merge =
      pre->splitBasicBlock(B.GetInsertPoint(), C.tag + name + ".merge");
  pre->getTerminator()->eraseFromParent();
  BasicBlock *doBB =
      BasicBlock::Create(C.ctx, C.tag + name + ".do", C.F, merge);
  IRBuilder<>(pre).CreateCondBr(isWarp0, doBB, merge);
  B.SetInsertPoint(doBB);
  return merge;
}

// The mma chain of every (mt, nt) output tile of one K-tile, warp-gated when
// C.useWarpGate, then the barrier that keeps the next fill off the tiles the
// chain reads, and the back edge.
static void emitMmaChain(const RaiseContext &C, IRBuilder<> &bodyB,
                         KStreamLoop &K, const Barrier &barrier) {
  const unsigned N = C.N, cFragSize = C.cFragSize;
  const unsigned numPhisPerGroup = C.accPhisPerGroup();
  BasicBlock *gatePreBB = nullptr;
  BasicBlock *gateMergeBB = nullptr;
  if (C.useWarpGate) {
    gatePreBB = bodyB.GetInsertBlock();
    gateMergeBB = openWarpGate(C, bodyB, K.tlin, "_gate");
  }

  Value *ldaV = ConstantInt::get(C.i32Ty, (int64_t)C.t.tileK);
  Value *ldbV = ConstantInt::get(C.i32Ty, (int64_t)C.L.nPad);
  auto callMma = [&](Value *aTile, Value *bTile,
                     ArrayRef<Value *> cIn) -> Value * {
    Value *aFragV = bodyB.CreateCall(C.wmma.LoadA, {aTile, ldaV});
    Value *bFragV = bodyB.CreateCall(C.wmma.LoadB, {bTile, ldbV});
    SmallVector<Value *, 32> args;
    args.reserve(C.aFragSize + C.bFragSize + cFragSize);
    for (unsigned i = 0; i < C.aFragSize; ++i)
      args.push_back(C.aFragSize == 1 ? aFragV
                                      : bodyB.CreateExtractValue(aFragV, i));
    for (unsigned i = 0; i < C.bFragSize; ++i)
      args.push_back(C.bFragSize == 1 ? bFragV
                                      : bodyB.CreateExtractValue(bFragV, i));
    for (unsigned i = 0; i < cFragSize; ++i)
      args.push_back(cIn[i]);
    return bodyB.CreateCall(C.wmma.Mma, args);
  };
  auto fragElement = [&](Value *frag, unsigned e) {
    return cFragSize == 1 ? frag : bodyB.CreateExtractValue(frag, e);
  };

  Constant *cZeroElt = Constant::getNullValue(C.cFragElemTy);
  SmallVector<SmallVector<Value *, 16>, 8> accLoop(N);
  for (unsigned k = 0; k < N; ++k)
    accLoop[k].resize(numPhisPerGroup);
  for (unsigned mt = 0; mt < C.L.mTiles; ++mt) {
    for (unsigned nt = 0; nt < C.L.nTiles; ++nt) {
      // Scratch is refilled at offset 0 per iteration, so the wmma loads read
      // from the scratch base with no K offset.
      SmallVector<Value *, 8> aRowBases(N), bColBases(N);
      for (unsigned k = 0; k < N; ++k) {
        aRowBases[k] = bodyB.CreatePtrAdd(
            C.aGen[k], ConstantInt::get(C.i64Ty, (int64_t)mt * C.t.tileM *
                                                     C.aScratchRowByte()));
        bColBases[k] = bodyB.CreatePtrAdd(
            C.bGen[k],
            ConstantInt::get(C.i64Ty, (int64_t)nt * C.t.tileN * C.opByte));
      }
      unsigned phiBase = (mt * C.L.nTiles + nt) * cFragSize;
      SmallVector<SmallVector<Value *, 16>, 8> cur(N);
      for (unsigned k = 0; k < N; ++k) {
        cur[k].resize(cFragSize);
        for (unsigned i = 0; i < cFragSize; ++i)
          cur[k][i] = K.accPhis[k][phiBase + i];
      }

      if (C.split) {
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
  if (C.useWarpGate) {
    BasicBlock *gateEndBB = bodyB.GetInsertBlock();
    bodyB.CreateBr(gateMergeBB);
    IRBuilder<> mergeB =
        builderAt(gateMergeBB, gateMergeBB->getFirstInsertionPt());
    for (unsigned k = 0; k < N; ++k) {
      for (unsigned i = 0; i < numPhisPerGroup; ++i) {
        PHINode *p =
            mergeB.CreatePHI(C.cFragElemTy, 2, C.accName(k) + ".merge");
        p->addIncoming(accLoop[k][i], gateEndBB);
        p->addIncoming(K.accPhis[k][i], gatePreBB);
        accLoop[k][i] = p;
      }
    }
    setInsertPoint(bodyB, gateMergeBB, gateMergeBB->getFirstInsertionPt());
  }

  // Barrier after the wmma reads so the next iteration's fill doesn't race
  // them.
  bodyB.CreateCall(barrier.Fn, barrier.Args);

  Value *kNext = bodyB.CreateAdd(K.kOuterPhi,
                                 ConstantInt::get(C.i32Ty, (int32_t)C.t.tileK));
  BasicBlock *latchBB = K.latchBr->getParent();
  K.kOuterPhi->addIncoming(kNext, latchBB);
  for (unsigned k = 0; k < N; ++k)
    for (unsigned i = 0; i < numPhisPerGroup; ++i)
      K.accPhis[k][i]->addIncoming(accLoop[k][i], latchBB);
}

// After the loop: stores the final accumulators to the output scratch,
// warp-gated too since only warp 0's PHIs hold the result, then a barrier,
// then this thread's cell read back as sum_k d[k] / SCALE^k. The block is the
// profiled M x N one, so the cell is not clamped to the output tile, and the
// output scratch's row stride is nPad. Returns the readback and the block it
// ends in, which still branches back to the loop.
static std::pair<Value *, BasicBlock *>
emitStoreAndReadback(const RaiseContext &C, const KStreamLoop &K,
                     const Barrier &barrier) {
  const ScalarLoopHandle &h = C.h;
  const unsigned cFragSize = C.cFragSize;
  // NOLINTNEXTLINE(enzyme-first-non-phi): ahead of the debug records
  BasicBlock::iterator afterAt = K.afterLoopBB->getFirstNonPHIIt();
  IRBuilder<> afterB = builderAt(K.afterLoopBB, afterAt);
  BasicBlock *storeMergeBB = nullptr;
  if (C.useWarpGate) {
    Value *afterTlin =
        emitThreadLinAndBlockSize(afterB, C.mod, h.profiledBlock, h.is2DBlock())
            .first;
    storeMergeBB = openWarpGate(C, afterB, afterTlin, "_store_gate");
  }
  Value *ldcV = ConstantInt::get(C.i32Ty, (int64_t)C.L.nPad);
  for (unsigned mt = 0; mt < C.L.mTiles; ++mt) {
    for (unsigned nt = 0; nt < C.L.nTiles; ++nt) {
      Value *dTileByte = ConstantInt::get(
          C.i64Ty, ((int64_t)mt * C.t.tileM * (int64_t)C.L.nPad +
                    (int64_t)nt * C.t.tileN) *
                       C.accByte);
      unsigned phiBase = (mt * C.L.nTiles + nt) * cFragSize;
      for (unsigned k = 0; k < C.N; ++k) {
        Value *dTilePtr = afterB.CreatePtrAdd(C.dGen[k], dTileByte);
        SmallVector<Value *, 16> args;
        args.reserve(2 + cFragSize);
        args.push_back(dTilePtr);
        for (unsigned i = 0; i < cFragSize; ++i)
          args.push_back(K.accPhis[k][phiBase + i]);
        args.push_back(ldcV);
        afterB.CreateCall(C.wmma.StoreD, args);
      }
    }
  }
  if (C.useWarpGate) {
    afterB.CreateBr(storeMergeBB);
    setInsertPoint(afterB, storeMergeBB, storeMergeBB->getFirstInsertionPt());
  }
  afterB.CreateCall(barrier.Fn, barrier.Args);

  IRBuilder<> &RB = afterB;
  Value *rRow = emitTidIndex(RB, C.mod, h.aRow);
  Value *rCol = emitTidIndex(RB, C.mod, h.bCol);
  Value *rRowOff = RB.CreateMul(rRow, ConstantInt::get(C.i32Ty, C.L.nPad));
  Value *rCellIdx = RB.CreateAdd(rRowOff, rCol);
  Value *sum = nullptr;
  double scalePowK = 1.0;
  for (unsigned k = 0; k < C.N; ++k) {
    Value *cell = RB.CreateGEP(C.accTy, C.dGen[k], rCellIdx);
    LoadInst *raw = RB.CreateLoad(C.accTy, cell);
    raw->setAlignment(Align(C.accByte));
    if (k == 0) {
      sum = raw;
    } else {
      Value *term = RB.CreateFDiv(raw, ConstantFP::get(C.accTy, scalePowK));
      sum = RB.CreateFAdd(sum, term);
    }
    if (k + 1 < C.N)
      scalePowK *= sliceScale(C.t.variant.operand);
  }
  Value *readback = emitFPCast(RB, sum, C.srcTy);
  return {readback, RB.GetInsertBlock()};
}

// Ends the raise in `raiseEnd` with a branch to the loop's exit block, where
// every use of the reduction's result outside the loop takes the readback on
// that edge and the loop's value on the edge from `loopLatch`.
static void replaceLoopResult(const ScalarLoopHandle &h, BasicBlock *loopLatch,
                              BasicBlock *raiseEnd, Value *readback) {
  BasicBlock *exitBB = h.exitBB;
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

void materializeInKernel(const RecognizedMatmul &m, const MatmulCandidate &t) {
  const SchemeVariant &v = t.variant;
  assert(t.realization == Realization::InKernel &&
         (v.scheme == Scheme::Direct || v.scheme == Scheme::Tcec) &&
         "only the in-kernel direct and split schemes are realized here");
  assert((v.scheme != Scheme::Tcec || (v.accumulator == NumberFormat::FP32 &&
                                       v.operand != NumberFormat::FP64)) &&
         "a split scheme slices into a 16- or 19-bit format and accumulates "
         "in F32");
  const ScalarLoopHandle &h = m.scalarLoop();
  BasicBlock *preheader = h.preheader;
  BasicBlock *exitBB = h.exitBB;
  assert(preheader && exitBB &&
         "recognition admits only loops with a preheader and one exit block");
  assert(h.inKernelRaisable &&
         "an in-kernel raise is proposed only where it can take the loop's "
         "place");

  // The raise is laid out for the profiled block shape and needs every thread
  // of the block, so it runs only in a block of that shape whose threads all
  // pass the loop's guards; any other block runs the code as written.
  BasicBlock *loopHeader = preheader->getSingleSuccessor();
  BasicBlock *loopLatch = exitBB->getSinglePredecessor();
  assert(loopHeader && loopLatch &&
         "recognition admits only a loop whose exit is entered from its latch");
  preheader = emitRaiseGuard(*preheader, *loopHeader, h);

  RaiseContext C = makeRaiseContext(m, t, preheader->getParent());
  SmallVector<GlobalVariable *, 8> aScratch, bScratch, dScratch;
  createScratch(C, aScratch, bScratch, dScratch);

  IRBuilder<> B(preheader->getTerminator());
  const Barrier barrier = declareBarrier(C.mod);
  auto [aBlockBase, bBlockBase] = emitBlockBases(C, B);
  for (unsigned k = 0; k < C.N; ++k) {
    C.aGen.push_back(B.CreateAddrSpaceCast(aScratch[k], C.genPtrTy));
    C.bGen.push_back(B.CreateAddrSpaceCast(bScratch[k], C.genPtrTy));
    C.dGen.push_back(B.CreateAddrSpaceCast(dScratch[k], C.genPtrTy));
  }

  // Placed in the K loop's body by emitKStreamLoop.
  IRBuilder<> bodyB(preheader);
  KStreamLoop K = emitKStreamLoop(C, B, bodyB, aBlockBase, bBlockBase, barrier);
  emitMmaChain(C, bodyB, K, barrier);
  auto [readback, raiseEnd] = emitStoreAndReadback(C, K, barrier);
  replaceLoopResult(h, loopLatch, raiseEnd, readback);
}

} // namespace poseidon
