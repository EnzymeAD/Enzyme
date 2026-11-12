//=- WmmaUtils.cpp - WMMA materialization helpers ------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WmmaUtils.h"
#include "Targets/NVPTXTensorCore.h"
#include "Targets/Target.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>

using namespace llvm;

namespace poseidon {

Type *llvmTypeForFPKind(LLVMContext &ctx, FPKind k) {
  switch (k) {
  case FPKind::F16:
    return Type::getHalfTy(ctx);
  case FPKind::BF16:
    return Type::getBFloatTy(ctx);
  case FPKind::TF32:
  case FPKind::F32:
    return Type::getFloatTy(ctx);
  case FPKind::F64:
    return Type::getDoubleTy(ctx);
  case FPKind::S8:
    return Type::getInt8Ty(ctx);
  case FPKind::S32:
    return Type::getInt32Ty(ctx);
  case FPKind::Invalid:
    report_fatal_error("llvmTypeForFPKind: Invalid FPKind");
  }
  llvm_unreachable("unknown FPKind");
}

Value *emitFPCast(IRBuilder<> &B, Value *v, Type *toTy) {
  Type *fromTy = v->getType();
  if (fromTy == toTy)
    return v;
  unsigned fromBits = fromTy->getScalarSizeInBits();
  unsigned toBits = toTy->getScalarSizeInBits();
  if (toBits < fromBits)
    return B.CreateFPTrunc(v, toTy);
  if (toBits > fromBits)
    return B.CreateFPExt(v, toTy);
  return B.CreateBitCast(v, toTy);
}

GlobalVariable *getOrCreateSharedScratch(Module *M, const Twine &name,
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

Value *emitTidIntrinsic(IRBuilder<> &B, Module *M, TidAxis axis) {
  unsigned a;
  switch (axis) {
  case TidAxis::TidX:
    a = 0;
    break;
  case TidAxis::TidY:
    a = 1;
    break;
  case TidAxis::TidZ:
    a = 2;
    break;
  default:
    report_fatal_error("emitTidIntrinsic: axis not resolved (Unknown/Other)");
  }
  return nvptx::emitThreadIndex(B, M, a);
}

Value *emitTidIndex(IRBuilder<> &B, Module *M, TidAxis fast, TidAxis slow,
                    unsigned mult) {
  Value *fastV = emitTidIntrinsic(B, M, fast);
  if (slow == TidAxis::Unknown)
    return fastV;
  if (mult == 0)
    report_fatal_error("emitTidIndex: fused index with a zero multiplier");
  Value *slowV = emitTidIntrinsic(B, M, slow);
  Type *i32Ty = Type::getInt32Ty(B.getContext());
  return B.CreateAdd(fastV,
                     B.CreateMul(slowV, ConstantInt::get(i32Ty, (int32_t)mult)),
                     "wmma.fused_idx");
}

std::pair<Value *, Value *> emitThreadLinAndBlockSize(IRBuilder<> &B, Module *M,
                                                      bool is2DBlock) {
  Value *tx = nvptx::emitThreadIndex(B, M, 0);
  Value *ty = nvptx::emitThreadIndex(B, M, 1);
  Value *nx = nvptx::emitBlockDim(B, M, 0);
  Value *ny = nvptx::emitBlockDim(B, M, 1);
  if (is2DBlock) {
    // 2D block: tz == 0, nz == 1; fold away the tz / nz terms.
    Value *tlin = B.CreateAdd(B.CreateMul(ty, nx), tx);
    Value *bsz = B.CreateMul(nx, ny);
    return {tlin, bsz};
  }
  Value *tz = nvptx::emitThreadIndex(B, M, 2);
  Value *nz = nvptx::emitBlockDim(B, M, 2);
  Value *tlin =
      B.CreateAdd(B.CreateMul(B.CreateAdd(B.CreateMul(tz, ny), ty), nx), tx);
  Value *bsz = B.CreateMul(B.CreateMul(nx, ny), nz);
  return {tlin, bsz};
}

void emitCooperativeScratchFill(IRBuilder<> &B, Value *scratchBase, Type *dstTy,
                                FPKind dstKind, Value *srcBlockBase,
                                Type *srcTy, uint64_t numRows, uint64_t numCols,
                                int64_t srcRowStrideByte,
                                int64_t srcColStrideByte,
                                int64_t dstRowStrideByte,
                                int64_t dstColStrideByte, Value *threadLin,
                                Value *blockSize, bool assumeBlockCoversAll) {
  Function *F = B.GetInsertBlock()->getParent();
  LLVMContext &ctx = F->getContext();
  Type *i32Ty = Type::getInt32Ty(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *i8Ty = Type::getInt8Ty(ctx);

  uint64_t numElts = numRows * numCols;
  if (numElts == 0)
    return;

  auto emitElementWork = [&](IRBuilder<> &bodyB, Value *counter) {
    Value *numColsV = ConstantInt::get(i32Ty, (int32_t)numCols);
    Value *r32 = bodyB.CreateUDiv(counter, numColsV, "wmma_cfill.r");
    Value *c32 = bodyB.CreateURem(counter, numColsV, "wmma_cfill.c");
    Value *r64 = bodyB.CreateZExt(r32, i64Ty);
    Value *c64 = bodyB.CreateZExt(c32, i64Ty);
    Value *srcRowOff =
        bodyB.CreateMul(r64, ConstantInt::get(i64Ty, srcRowStrideByte));
    Value *srcColOff =
        bodyB.CreateMul(c64, ConstantInt::get(i64Ty, srcColStrideByte));
    Value *srcOff = bodyB.CreateAdd(srcRowOff, srcColOff);
    Value *srcPtr = bodyB.CreateGEP(i8Ty, srcBlockBase, srcOff);
    Value *srcVal = bodyB.CreateLoad(srcTy, srcPtr);
    Value *cvt = emitFPCast(bodyB, srcVal, dstTy);
    if (dstKind == FPKind::TF32) {
      // wmma.load.a/b.tf32 reads the float bit-pattern as-is and the MMA
      // truncates the low 13 mantissa bits, so an FPTrunc-only fill applies
      // truncation (one-signed bias) instead of round-to-nearest. RN-quantize
      // before the store, as the Ozaki slice path does.
      cvt = nvptx::emitRoundToTF32(bodyB, F->getParent(), cvt);
    }
    Value *dstRowOff =
        bodyB.CreateMul(r64, ConstantInt::get(i64Ty, dstRowStrideByte));
    Value *dstColOff =
        bodyB.CreateMul(c64, ConstantInt::get(i64Ty, dstColStrideByte));
    Value *dstOff = bodyB.CreateAdd(dstRowOff, dstColOff);
    Value *dstPtr = bodyB.CreateGEP(i8Ty, scratchBase, dstOff);
    bodyB.CreateStore(cvt, dstPtr);
  };

  Instruction *splitInst = &*B.GetInsertPoint();
  BasicBlock *origBB = B.GetInsertBlock();

  if (assumeBlockCoversAll) {
    // Caller asserts blockSize >= numElts: `if (threadLin < numElts) work;`.
    BasicBlock *afterBB =
        origBB->splitBasicBlock(splitInst->getIterator(), "wmma_cfill.after");
    origBB->getTerminator()->eraseFromParent();
    BasicBlock *workBB = BasicBlock::Create(ctx, "wmma_cfill.do", F, afterBB);
    {
      IRBuilder<> guardB(origBB);
      Value *cmp = guardB.CreateICmpULT(
          threadLin, ConstantInt::get(i32Ty, (int32_t)numElts));
      guardB.CreateCondBr(cmp, workBB, afterBB);
    }
    {
      IRBuilder<> bodyB(workBB);
      emitElementWork(bodyB, threadLin);
      bodyB.CreateBr(afterBB);
    }
    B.SetInsertPoint(&*afterBB->getFirstInsertionPt());
    return;
  }

  BasicBlock *afterBB =
      origBB->splitBasicBlock(splitInst->getIterator(), "wmma_cfill.after");
  origBB->getTerminator()->eraseFromParent();

  BasicBlock *loopHdr = BasicBlock::Create(ctx, "wmma_cfill.hdr", F, afterBB);
  BasicBlock *loopBody = BasicBlock::Create(ctx, "wmma_cfill.body", F, afterBB);

  IRBuilder<>(origBB).CreateBr(loopHdr);

  IRBuilder<> hdrB(loopHdr);
  PHINode *counter = hdrB.CreatePHI(i32Ty, 2, "wmma_cfill.i");
  counter->addIncoming(threadLin, origBB);
  Value *cmp =
      hdrB.CreateICmpULT(counter, ConstantInt::get(i32Ty, (int32_t)numElts));
  hdrB.CreateCondBr(cmp, loopBody, afterBB);

  IRBuilder<> bodyB(loopBody);
  emitElementWork(bodyB, counter);
  Value *next = bodyB.CreateAdd(counter, blockSize, "wmma_cfill.next");
  counter->addIncoming(next, loopBody);
  bodyB.CreateBr(loopHdr);

  B.SetInsertPoint(&*afterBB->getFirstInsertionPt());
}

void emitCooperativeScratchFillPadded(
    IRBuilder<> &B, Value *scratchBase, Type *dstTy, FPKind dstKind,
    Value *srcBlockBase, Type *srcTy, uint64_t fillRows, uint64_t fillCols,
    Value *validRows, Value *validCols, int64_t srcRowStrideByte,
    int64_t srcColStrideByte, int64_t dstRowStrideByte,
    int64_t dstColStrideByte, Value *threadLin, Value *blockSize,
    bool assumeBlockCoversAll) {
  Function *F = B.GetInsertBlock()->getParent();
  LLVMContext &ctx = F->getContext();
  Type *i32Ty = Type::getInt32Ty(ctx);
  Type *i64Ty = Type::getInt64Ty(ctx);
  Type *i8Ty = Type::getInt8Ty(ctx);

  uint64_t numElts = fillRows * fillCols;
  if (numElts == 0)
    return;
  if (!validRows || !validCols)
    report_fatal_error("emitCooperativeScratchFillPadded: null valid extent");

  Constant *dstZero = Constant::getNullValue(dstTy);

  // Leaves `bodyB` at the merge block (the caller must re-read GetInsertBlock()
  // for any PHI edge). The conversion is branched, not selected: on an
  // FP64-starved part the narrowing convert issues on the FP64 pipe the raise
  // exists to vacate, and at a tile-starved shape most of the padded tile is
  // pad; branching also removes the need to clamp the source index.
  auto emitElementWork = [&](IRBuilder<> &bodyB, Value *counter) {
    Value *fillColsV = ConstantInt::get(i32Ty, (int32_t)fillCols);
    Value *r32 = bodyB.CreateUDiv(counter, fillColsV, "wmma_pfill.r");
    Value *c32 = bodyB.CreateURem(counter, fillColsV, "wmma_pfill.c");
    Value *inR = bodyB.CreateICmpULT(r32, validRows);
    Value *inC = bodyB.CreateICmpULT(c32, validCols);
    Value *inRange = bodyB.CreateAnd(inR, inC, "wmma_pfill.in");

    BasicBlock *predBB = bodyB.GetInsertBlock();
    BasicBlock *loadBB = BasicBlock::Create(ctx, "wmma_pfill.load", F);
    BasicBlock *mergeBB = BasicBlock::Create(ctx, "wmma_pfill.merge", F);
    bodyB.CreateCondBr(inRange, loadBB, mergeBB);

    IRBuilder<> lB(loadBB);
    Value *srcRowOff = lB.CreateMul(lB.CreateZExt(r32, i64Ty),
                                    ConstantInt::get(i64Ty, srcRowStrideByte));
    Value *srcColOff = lB.CreateMul(lB.CreateZExt(c32, i64Ty),
                                    ConstantInt::get(i64Ty, srcColStrideByte));
    Value *srcOff = lB.CreateAdd(srcRowOff, srcColOff);
    Value *srcPtr = lB.CreateGEP(i8Ty, srcBlockBase, srcOff);
    Value *srcVal = lB.CreateLoad(srcTy, srcPtr);
    Value *cvt = emitFPCast(lB, srcVal, dstTy);
    if (dstKind == FPKind::TF32) {
      // Same RN quantization the unpadded fill applies (see there).
      cvt = nvptx::emitRoundToTF32(lB, F->getParent(), cvt);
    }
    BasicBlock *loadEndBB = lB.GetInsertBlock();
    lB.CreateBr(mergeBB);

    IRBuilder<> mB(mergeBB);
    PHINode *out = mB.CreatePHI(dstTy, 2, "wmma_pfill.v");
    out->addIncoming(cvt, loadEndBB);
    out->addIncoming(dstZero, predBB);
    Value *r64 = mB.CreateZExt(r32, i64Ty);
    Value *c64 = mB.CreateZExt(c32, i64Ty);
    Value *dstRowOff =
        mB.CreateMul(r64, ConstantInt::get(i64Ty, dstRowStrideByte));
    Value *dstColOff =
        mB.CreateMul(c64, ConstantInt::get(i64Ty, dstColStrideByte));
    Value *dstOff = mB.CreateAdd(dstRowOff, dstColOff);
    Value *dstPtr = mB.CreateGEP(i8Ty, scratchBase, dstOff);
    mB.CreateStore(out, dstPtr);
    bodyB.SetInsertPoint(mergeBB);
  };

  Instruction *splitInst = &*B.GetInsertPoint();
  BasicBlock *origBB = B.GetInsertBlock();

  if (assumeBlockCoversAll) {
    BasicBlock *afterBB =
        origBB->splitBasicBlock(splitInst->getIterator(), "wmma_pfill.after");
    origBB->getTerminator()->eraseFromParent();
    BasicBlock *workBB = BasicBlock::Create(ctx, "wmma_pfill.do", F, afterBB);
    {
      IRBuilder<> guardB(origBB);
      Value *cmp = guardB.CreateICmpULT(
          threadLin, ConstantInt::get(i32Ty, (int32_t)numElts));
      guardB.CreateCondBr(cmp, workBB, afterBB);
    }
    {
      IRBuilder<> bodyB(workBB);
      emitElementWork(bodyB, threadLin);
      bodyB.CreateBr(afterBB); // emitElementWork left bodyB at its merge block
    }
    B.SetInsertPoint(&*afterBB->getFirstInsertionPt());
    return;
  }

  BasicBlock *afterBB =
      origBB->splitBasicBlock(splitInst->getIterator(), "wmma_pfill.after");
  origBB->getTerminator()->eraseFromParent();

  BasicBlock *loopHdr = BasicBlock::Create(ctx, "wmma_pfill.hdr", F, afterBB);
  BasicBlock *loopBody = BasicBlock::Create(ctx, "wmma_pfill.body", F, afterBB);

  IRBuilder<>(origBB).CreateBr(loopHdr);

  IRBuilder<> hdrB(loopHdr);
  PHINode *counter = hdrB.CreatePHI(i32Ty, 2, "wmma_pfill.i");
  counter->addIncoming(threadLin, origBB);
  Value *cmp =
      hdrB.CreateICmpULT(counter, ConstantInt::get(i32Ty, (int32_t)numElts));
  hdrB.CreateCondBr(cmp, loopBody, afterBB);

  IRBuilder<> bodyB(loopBody);
  emitElementWork(bodyB, counter);
  Value *next = bodyB.CreateAdd(counter, blockSize, "wmma_pfill.next");
  // The latch is the block emitElementWork left the builder in, not loopBody.
  counter->addIncoming(next, bodyB.GetInsertBlock());
  bodyB.CreateBr(loopHdr);

  B.SetInsertPoint(&*afterBB->getFirstInsertionPt());
}

} // namespace poseidon
