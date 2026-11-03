//===- CallDerivatives.cpp - Implementation of known call derivatives --===//
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
// This file contains the implementation of functions in instruction visitor
// AdjointGenerator that generate corresponding augmented forward pass code,
// and adjoints for certain known functions.
//
//===----------------------------------------------------------------------===//

#include "AdjointGenerator.h"
#include "FlangRuntime.h"

using namespace llvm;

extern "C" {
void (*EnzymeShadowAllocRewrite)(LLVMValueRef, void *, LLVMValueRef, uint64_t,
                                 LLVMValueRef, uint8_t) = nullptr;
}

/// Whether \p Begin has exactly one llvm.julia.gc_preserve_end outside the
/// abort-only blocks the reverse pass drops, as reversing the region requires;
/// CanonicalizeGCPreserveEnds merges them where it can.
static bool hasSingleGCPreserveEnd(CallInst *Begin, GradientUtils *gutils) {
  unsigned Ends = 0;
  for (auto U : Begin->users())
    if (auto CI = dyn_cast<CallInst>(U))
      if (getFuncNameFromCall(CI) == "llvm.julia.gc_preserve_end" &&
          !gutils->notForAnalysis.count(CI->getParent()))
        ++Ends;
  return Ends == 1;
}

/// Forward mode of an MPI call whose derivative is the same call with the
/// buffers \p ShadowArgs replaced by their shadows (for the reductions, of a
/// sum). At vector width > 1 the call is replayed once per lane. All other
/// arguments, including the trailing `ierr` of the Fortran ABI, are passed as
/// in the original call. An inactive constant buffer, such as MPI_IN_PLACE or
/// MPI_STATUS_IGNORE, is passed unchanged.
static void createMPIForwardCall(CallInst &call, ArrayRef<unsigned> ShadowArgs,
                                 GradientUtils *gutils, IRBuilder<> &Builder2) {
  std::vector<ValueType> BundleTypes(call.arg_size(), ValueType::Primal);
  SmallVector<Value *, 2> shadows;
  for (unsigned i : ShadowArgs) {
    Value *arg = call.getArgOperand(i);
    if (isa<Constant>(arg) && gutils->isConstantValue(arg)) {
      shadows.push_back(nullptr);
      continue;
    }
    BundleTypes[i] = ValueType::Shadow;
    shadows.push_back(gutils->invertPointerM(arg, Builder2));
  }
  auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                         /*lookup*/ false);

  for (unsigned l = 0; l < gutils->getWidth(); l++) {
    SmallVector<Value *, 8> args;
    for (auto &op : call.args())
      args.push_back(gutils->getNewFromOriginal(op));
    for (auto en : llvm::enumerate(ShadowArgs)) {
      Value *sh = shadows[en.index()];
      if (!sh)
        continue;
      if (gutils->getWidth() > 1)
        sh = gutils->extractMeta(Builder2, sh, l);
      Type *T = args[en.value()]->getType();
      if (sh->getType()->isIntegerTy())
        sh = Builder2.CreateIntToPtr(sh, T);
      else if (sh->getType() != T)
        sh = Builder2.CreatePointerCast(sh, T);
      args[en.value()] = sh;
    }
    auto dcall = Builder2.CreateCall(call.getFunctionType(),
                                     call.getCalledOperand(), args, Defs);
    dcall->setCallingConv(call.getCallingConv());
    dcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
  }
}

void AdjointGenerator::handleMPI(llvm::CallInst &call, llvm::Function *called,
                                 llvm::StringRef funcName) {
  using namespace llvm;

  assert(called);

  IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
  BuilderZ.setFastMathFlags(getFast());

  // In forward mode, the derivative of an MPI call is the call on the shadows,
  // which is replayed per lane at vector width > 1 (createMPIForwardCall).
  if (gutils->getWidth() > 1 && Mode != DerivativeMode::ForwardMode &&
      Mode != DerivativeMode::ForwardModeError) {
    std::string s;
    raw_string_ostream ss(s);
    ss << funcName << " is not supported in vector reverse mode: " << call;
    EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
    return;
  }

  // MPI send / recv can only send float/integers
  if (funcName == "PMPI_Isend" || funcName == "MPI_Isend" ||
      funcName == "PMPI_Irecv" || funcName == "MPI_Irecv") {
    if (!gutils->isConstantInstruction(&call)) {
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined) {
        assert(!gutils->isConstantValue(call.getOperand(0)));
        assert(!gutils->isConstantValue(call.getOperand(6)));
        Value *d_req = gutils->invertPointerM(call.getOperand(6), BuilderZ);
        if (d_req->getType()->isIntegerTy()) {
          d_req = BuilderZ.CreateIntToPtr(
              d_req, getUnqual(getInt8PtrTy(call.getContext())));
        }

        auto i64 = Type::getInt64Ty(call.getContext());
        auto impi = getMPIHelper(call.getContext());

        Value *impialloc =
            CreateAllocation(BuilderZ, impi, ConstantInt::get(i64, 1));
        BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));

        d_req = BuilderZ.CreateBitCast(d_req, getUnqual(impialloc->getType()));
        Value *d_req_prev = BuilderZ.CreateLoad(impialloc->getType(), d_req);
        BuilderZ.CreateStore(
            BuilderZ.CreatePointerCast(d_req_prev,
                                       getInt8PtrTy(call.getContext())),
            getMPIMemberPtr<MPI_Elem::Old>(BuilderZ, impialloc, impi));
        BuilderZ.CreateStore(impialloc, d_req);

        if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
          Value *tysize =
              MPI_TYPE_SIZE(gutils->getNewFromOriginal(call.getOperand(2)),
                            BuilderZ, call.getType(), called);

          auto len_arg = BuilderZ.CreateZExtOrTrunc(
              gutils->getNewFromOriginal(call.getOperand(1)),
              Type::getInt64Ty(call.getContext()));
          len_arg = BuilderZ.CreateMul(
              len_arg,
              BuilderZ.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);

          Value *firstallocation =
              CreateAllocation(BuilderZ, Type::getInt8Ty(call.getContext()),
                               len_arg, "mpirecv_malloccache");
          BuilderZ.CreateStore(firstallocation, getMPIMemberPtr<MPI_Elem::Buf>(
                                                    BuilderZ, impialloc, impi));
          BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));
        } else {
          Value *ibuf = gutils->invertPointerM(call.getOperand(0), BuilderZ);
          if (ibuf->getType()->isIntegerTy())
            ibuf =
                BuilderZ.CreateIntToPtr(ibuf, getInt8PtrTy(call.getContext()));
          BuilderZ.CreateStore(
              ibuf, getMPIMemberPtr<MPI_Elem::Buf>(BuilderZ, impialloc, impi));
        }

        BuilderZ.CreateStore(
            BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(1)), i64),
            getMPIMemberPtr<MPI_Elem::Count>(BuilderZ, impialloc, impi));

        Value *dataType = gutils->getNewFromOriginal(call.getOperand(2));
        if (dataType->getType()->isIntegerTy())
          dataType = BuilderZ.CreateIntToPtr(
              dataType, getInt8PtrTy(dataType->getContext()));
        BuilderZ.CreateStore(
            BuilderZ.CreatePointerCast(dataType,
                                       getInt8PtrTy(call.getContext())),
            getMPIMemberPtr<MPI_Elem::DataType>(BuilderZ, impialloc, impi));

        BuilderZ.CreateStore(
            BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(3)), i64),
            getMPIMemberPtr<MPI_Elem::Src>(BuilderZ, impialloc, impi));

        BuilderZ.CreateStore(
            BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(4)), i64),
            getMPIMemberPtr<MPI_Elem::Tag>(BuilderZ, impialloc, impi));

        Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
        if (comm->getType()->isIntegerTy())
          comm = BuilderZ.CreateIntToPtr(comm,
                                         getInt8PtrTy(dataType->getContext()));
        BuilderZ.CreateStore(
            BuilderZ.CreatePointerCast(comm, getInt8PtrTy(call.getContext())),
            getMPIMemberPtr<MPI_Elem::Comm>(BuilderZ, impialloc, impi));

        BuilderZ.CreateStore(
            ConstantInt::get(
                Type::getInt8Ty(impialloc->getContext()),
                (funcName == "MPI_Isend" || funcName == "PMPI_Isend")
                    ? (int)MPI_CallType::ISEND
                    : (int)MPI_CallType::IRECV),
            getMPIMemberPtr<MPI_Elem::Call>(BuilderZ, impialloc, impi));
        // TODO old
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        Type *statusType = nullptr;
#if LLVM_VERSION_MAJOR < 17
        if (Function *recvfn = called->getParent()->getFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Wait"))) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        }
#endif
        if (statusType == nullptr) {
          statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
          llvm::errs() << " warning could not automatically determine mpi "
                          "status type, assuming [24 x i8]\n";
        }
        Value *req =
            lookup(gutils->getNewFromOriginal(call.getOperand(6)), Builder2);
        Value *d_req = lookup(
            gutils->invertPointerM(call.getOperand(6), Builder2), Builder2);
        if (d_req->getType()->isIntegerTy()) {
          d_req =
              Builder2.CreateIntToPtr(d_req, getInt8PtrTy(call.getContext()));
        }
        auto impi = getMPIHelper(call.getContext());
        Type *helperTy = getUnqual(impi);
        Value *helper = Builder2.CreatePointerCast(d_req, getUnqual(helperTy));
        helper = Builder2.CreateLoad(helperTy, helper);

        auto i64 = Type::getInt64Ty(call.getContext());

        Value *firstallocation;
        firstallocation = Builder2.CreateLoad(
            getInt8PtrTy(call.getContext()),
            getMPIMemberPtr<MPI_Elem::Buf>(Builder2, helper, impi));
        Value *len_arg = nullptr;
        if (auto C = dyn_cast<Constant>(
                gutils->getNewFromOriginal(call.getOperand(1)))) {
          len_arg = Builder2.CreateZExtOrTrunc(C, i64);
        } else {
          len_arg = Builder2.CreateLoad(
              i64, getMPIMemberPtr<MPI_Elem::Count>(Builder2, helper, impi));
        }
        Value *tysize = nullptr;
        if (auto C = dyn_cast<Constant>(
                gutils->getNewFromOriginal(call.getOperand(2)))) {
          tysize = C;
        } else {
          tysize = Builder2.CreateLoad(
              getInt8PtrTy(call.getContext()),
              getMPIMemberPtr<MPI_Elem::DataType>(Builder2, helper, impi));
        }

        Value *prev;
        prev = Builder2.CreateLoad(
            getInt8PtrTy(call.getContext()),
            getMPIMemberPtr<MPI_Elem::Old>(Builder2, helper, impi));

        Builder2.CreateStore(prev, Builder2.CreatePointerCast(
                                       d_req, getUnqual(prev->getType())));

        assert(shouldFree());

        assert(tysize);
        tysize = MPI_TYPE_SIZE(tysize, Builder2, call.getType(), called);

        Value *args[] = {/*req*/ req,
                         /*status*/ IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(statusType)};
        FunctionCallee waitFunc = nullptr;
        for (auto name : {
                 "MPI_Wait",
             })
          if (Function *recvfn = called->getParent()->getFunction(
                  getRenamedPerCallingConv(called->getName(), name))) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (statusArg->getType()->isIntegerTy())
              args[1] = Builder2.CreatePtrToInt(args[1], statusArg->getType());
            else
              args[1] = Builder2.CreateBitCast(args[1], statusArg->getType());
            waitFunc = recvfn;
            break;
          }
        if (!waitFunc) {
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();
          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          waitFunc = called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Wait"), FT);
        }
        assert(waitFunc);

        // Need to preserve the shadow Request (operand 6 in isend/irecv),
        // which becomes operand 0 for iwait.
        auto ReqDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::None, ValueType::None, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::Shadow},
            Builder2, /*lookup*/ true);

        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::None,
             ValueType::None},
            Builder2, /*lookup*/ true);

        auto fcall = Builder2.CreateCall(waitFunc, args, ReqDefs);
        fcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
        if (auto F = dyn_cast<Function>(waitFunc.getCallee()))
          fcall->setCallingConv(F->getCallingConv());
        len_arg = Builder2.CreateMul(
            len_arg,
            Builder2.CreateZExtOrTrunc(tysize,
                                       Type::getInt64Ty(Builder2.getContext())),
            "", true, true);
        if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv") {
          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(Builder2.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(Builder2.getContext());
          assert(!gutils->isConstantValue(call.getOperand(0)));
          auto dbuf = firstallocation;
          Value *nargs[] = {dbuf, val_arg, len_arg, volatile_arg};
          Type *tys[] = {dbuf->getType(), len_arg->getType()};

          auto memset = cast<CallInst>(Builder2.CreateCall(
              getIntrinsicDeclaration(called->getParent(), Intrinsic::memset,
                                      tys),
              nargs, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);
        } else if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
          assert(!gutils->isConstantValue(call.getOperand(0)));
          Value *shadow = lookup(
              gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);

          // TODO add operand bundle (unless force inlined?)
          DifferentiableMemCopyFloats(call, call.getOperand(0), firstallocation,
                                      shadow, len_arg, Builder2, BufferDefs);

          if (shouldFree()) {
            CreateDealloc(Builder2, firstallocation);
          }
        } else
          assert(0 && "illegal mpi");

        CreateDealloc(Builder2, helper);
      }
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(0)));
        assert(!gutils->isConstantValue(call.getOperand(6)));

        createMPIForwardCall(call, {/*buf*/ 0, /*request*/ 6}, gutils,
                             Builder2);
        return;
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Wait" || funcName == "PMPI_Wait") {
    Value *d_reqp = nullptr;
    auto impi = getMPIHelper(call.getContext());
    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      Value *req = gutils->getNewFromOriginal(call.getOperand(0));
      Value *d_req = gutils->invertPointerM(call.getOperand(0), BuilderZ);

      if (req->getType()->isIntegerTy()) {
        req = BuilderZ.CreateIntToPtr(
            req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      Value *isNull = nullptr;
      if (auto GV = gutils->newFunc->getParent()->getNamedValue(
              "ompi_request_null")) {
        Value *reql = BuilderZ.CreatePointerCast(req, getUnqual(GV->getType()));
        reql = BuilderZ.CreateLoad(GV->getType(), reql);
        isNull = BuilderZ.CreateICmpEQ(reql, GV);
      }

      if (d_req->getType()->isIntegerTy()) {
        d_req = BuilderZ.CreateIntToPtr(
            d_req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      d_reqp = BuilderZ.CreateLoad(
          getUnqual(impi),
          BuilderZ.CreatePointerCast(d_req, getUnqual(getUnqual(impi))));
      if (isNull)
        d_reqp =
            CreateSelect(BuilderZ, isNull,
                         Constant::getNullValue(d_reqp->getType()), d_reqp);
      if (auto I = dyn_cast<Instruction>(d_reqp))
        gutils->TapesToPreventRecomputation.insert(I);
      d_reqp = gutils->cacheForReverse(
          BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
    }
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(0)));
      Value *req =
          lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);

      if (Mode != DerivativeMode::ReverseModeCombined) {
        d_reqp = BuilderZ.CreatePHI(getUnqual(impi), 0);
        d_reqp = gutils->cacheForReverse(
            BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
      } else
        assert(d_reqp);
      d_reqp = lookup(d_reqp, Builder2);

      Value *isNull = Builder2.CreateICmpEQ(
          d_reqp, Constant::getNullValue(d_reqp->getType()));

      BasicBlock *currentBlock = Builder2.GetInsertBlock();
      BasicBlock *nonnullBlock = gutils->addReverseBlock(
          currentBlock, currentBlock->getName() + "_nonnull");
      BasicBlock *endBlock = gutils->addReverseBlock(
          nonnullBlock, currentBlock->getName() + "_end",
          /*fork*/ true, /*push*/ false);

      Builder2.CreateCondBr(isNull, endBlock, nonnullBlock);
      Builder2.SetInsertPoint(nonnullBlock);

      Value *cache = Builder2.CreateLoad(impi, d_reqp);

      Value *args[] = {
          getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache, impi),
          req};
      Type *types[sizeof(args) / sizeof(*args) - 1];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
        types[i] = args[i]->getType();
      Function *dwait = getOrInsertDifferentialMPI_Wait(
          *called->getParent(), types, call.getOperand(0)->getType(),
          called->getName());

      // Need to preserve the shadow Request (operand 0 in wait).
      // However, this doesn't end up preserving
      // the underlying buffers for the adjoint. To rememdy, force inline.
      auto cal =
          Builder2.CreateCall(dwait, args,
                              gutils->getInvertedBundles(
                                  &call, {ValueType::Shadow, ValueType::None},
                                  Builder2, /*lookup*/ true));
      cal->setCallingConv(dwait->getCallingConv());
      cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
      cal->addFnAttr(Attribute::AlwaysInline);
      Builder2.CreateBr(endBlock);
      {
        auto found = gutils->reverseBlockToPrimal.find(endBlock);
        assert(found != gutils->reverseBlockToPrimal.end());
        SmallVector<BasicBlock *, 4> &vec =
            gutils->reverseBlocks[found->second];
        assert(vec.size());
        vec.push_back(endBlock);
      }
      Builder2.SetInsertPoint(endBlock);
    } else if (Mode == DerivativeMode::ForwardMode ||
               Mode == DerivativeMode::ForwardModeError) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(0)));

      createMPIForwardCall(call, {/*request*/ 0, /*status*/ 1}, gutils,
                           Builder2);
      return;
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
    Value *d_reqp = nullptr;
    auto impi = getMPIHelper(call.getContext());
    PointerType *reqType = getUnqual(impi);
    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      Value *count = gutils->getNewFromOriginal(call.getOperand(0));
      Value *req = gutils->getNewFromOriginal(call.getOperand(1));
      Value *d_req = gutils->invertPointerM(call.getOperand(1), BuilderZ);

      if (req->getType()->isIntegerTy()) {
        req = BuilderZ.CreateIntToPtr(
            req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      if (d_req->getType()->isIntegerTy()) {
        d_req = BuilderZ.CreateIntToPtr(
            d_req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      Function *dsave = getOrInsertDifferentialWaitallSave(
          *gutils->oldFunc->getParent(),
          {count->getType(), req->getType(), d_req->getType()}, reqType);

      d_reqp = BuilderZ.CreateCall(dsave, {count, req, d_req});
      cast<CallInst>(d_reqp)->setCallingConv(dsave->getCallingConv());
      cast<CallInst>(d_reqp)->setDebugLoc(
          gutils->getNewFromOriginal(call.getDebugLoc()));
      d_reqp = gutils->cacheForReverse(
          BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
    }
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(1)));
      Value *count =
          lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);
      Value *req_orig =
          lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2);

      if (Mode != DerivativeMode::ReverseModeCombined) {
        d_reqp = BuilderZ.CreatePHI(getUnqual(reqType), 0);
        d_reqp = gutils->cacheForReverse(
            BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
      }

      d_reqp = lookup(d_reqp, Builder2);

      BasicBlock *currentBlock = Builder2.GetInsertBlock();
      BasicBlock *loopBlock = gutils->addReverseBlock(
          currentBlock, currentBlock->getName() + "_loop");
      BasicBlock *nonnullBlock = gutils->addReverseBlock(
          loopBlock, currentBlock->getName() + "_nonnull");
      BasicBlock *eloopBlock = gutils->addReverseBlock(
          nonnullBlock, currentBlock->getName() + "_eloop");
      BasicBlock *endBlock =
          gutils->addReverseBlock(eloopBlock, currentBlock->getName() + "_end",
                                  /*fork*/ true, /*push*/ false);

      Builder2.CreateCondBr(
          Builder2.CreateICmpNE(count,
                                ConstantInt::get(count->getType(), 0, false)),
          loopBlock, endBlock);

      Builder2.SetInsertPoint(loopBlock);
      auto idx = Builder2.CreatePHI(count->getType(), 2);
      idx->addIncoming(ConstantInt::get(count->getType(), 0, false),
                       currentBlock);
      Value *inc = Builder2.CreateAdd(
          idx, ConstantInt::get(count->getType(), 1, false), "", true, true);
      idx->addIncoming(inc, eloopBlock);

      Value *idxs[] = {idx};
      Value *req = Builder2.CreateInBoundsGEP(reqType, req_orig, idxs);
      Value *d_req = Builder2.CreateInBoundsGEP(reqType, d_reqp, idxs);

      d_req = Builder2.CreateLoad(
          getUnqual(impi),
          Builder2.CreatePointerCast(d_req, getUnqual(getUnqual(impi))));

      Value *isNull = Builder2.CreateICmpEQ(
          d_req, Constant::getNullValue(d_req->getType()));

      Builder2.CreateCondBr(isNull, eloopBlock, nonnullBlock);
      Builder2.SetInsertPoint(nonnullBlock);

      Value *cache = Builder2.CreateLoad(impi, d_req);

      Value *args[] = {
          getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache, impi),
          req};
      Type *types[sizeof(args) / sizeof(*args) - 1];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
        types[i] = args[i]->getType();
      Function *dwait = getOrInsertDifferentialMPI_Wait(
          *called->getParent(), types, req->getType(), called->getName());
      // Need to preserve the shadow Request (operand 6 in isend/irecv), which
      // becomes operand 0 for iwait. However, this doesn't end up preserving
      // the underlying buffers for the adjoint. To remedy, force inline the
      // function.
      auto cal = Builder2.CreateCall(
          dwait, args,
          gutils->getInvertedBundles(&call,
                                     {ValueType::None, ValueType::None,
                                      ValueType::None, ValueType::None,
                                      ValueType::None, ValueType::None,
                                      ValueType::Shadow},
                                     Builder2, /*lookup*/ true));
      cal->setCallingConv(dwait->getCallingConv());
      cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
      cal->addFnAttr(Attribute::AlwaysInline);
      Builder2.CreateBr(eloopBlock);

      Builder2.SetInsertPoint(eloopBlock);
      Builder2.CreateCondBr(Builder2.CreateICmpEQ(inc, count), endBlock,
                            loopBlock);
      {
        auto found = gutils->reverseBlockToPrimal.find(endBlock);
        assert(found != gutils->reverseBlockToPrimal.end());
        SmallVector<BasicBlock *, 4> &vec =
            gutils->reverseBlocks[found->second];
        assert(vec.size());
        vec.push_back(endBlock);
      }
      Builder2.SetInsertPoint(endBlock);
      if (shouldFree()) {
        CreateDealloc(Builder2, d_reqp);
      }
    } else if (Mode == DerivativeMode::ForwardMode ||
               Mode == DerivativeMode::ForwardModeError) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(1)));

      createMPIForwardCall(call, {/*array_of_requests*/ 1}, gutils, Builder2);
      return;
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Send" || funcName == "MPI_Ssend" ||
      funcName == "PMPI_Send" || funcName == "PMPI_Ssend") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
      if (!forwardMode)
        shadow = lookup(shadow, Builder2);
      if (shadow->getType()->isIntegerTy())
        shadow =
            Builder2.CreateIntToPtr(shadow, getInt8PtrTy(call.getContext()));

      Type *statusType = nullptr;
#if LLVM_VERSION_MAJOR < 17
      if (called->getContext().supportsTypedPointers()) {
        if (Function *recvfn = called->getParent()->getFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Recv"))) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        }
      }
#endif
      if (statusType == nullptr) {
        statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
        llvm::errs() << " warning could not automatically determine mpi "
                        "status type, assuming [24 x i8]\n";
      }

      Value *count = gutils->getNewFromOriginal(call.getOperand(1));
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *src = gutils->getNewFromOriginal(call.getOperand(3));
      if (!forwardMode)
        src = lookup(src, Builder2);

      Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
      if (!forwardMode)
        tag = lookup(tag, Builder2);

      Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      if (forwardMode) {
        createMPIForwardCall(call, {/*buf*/ 0}, gutils, Builder2);
        return;
      }

      Value *args[] = {
          /*buf*/ NULL,
          /*count*/ count,
          /*datatype*/ datatype,
          /*src*/ src,
          /*tag*/ tag,
          /*comm*/ comm,
          /*status*/
          IRBuilder<>(gutils->inversionAllocs).CreateAlloca(statusType)};

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      auto len_arg = Builder2.CreateZExtOrTrunc(
          args[1], Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      Value *firstallocation =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           len_arg, "mpirecv_malloccache");
      args[0] = firstallocation;

      Type *types[sizeof(args) / sizeof(*args)];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
        types[i] = args[i]->getType();
      FunctionType *FT = FunctionType::get(call.getType(), types, false);

      Builder2.SetInsertPoint(Builder2.GetInsertBlock());

      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::None, ValueType::None, ValueType::None,
           ValueType::None, ValueType::None, ValueType::None},
          Builder2, /*lookup*/ true);

      auto fcall = Builder2.CreateCall(
          called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Recv"), FT),
          args);
      fcall->setCallingConv(call.getCallingConv());

      DifferentiableMemCopyFloats(call, call.getOperand(0), firstallocation,
                                  shadow, len_arg, Builder2, BufferDefs);

      if (shouldFree()) {
        CreateDealloc(Builder2, firstallocation);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Recv" || funcName == "PMPI_Recv") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
      if (!forwardMode)
        shadow = lookup(shadow, Builder2);

      Value *count = gutils->getNewFromOriginal(call.getOperand(1));
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *source = gutils->getNewFromOriginal(call.getOperand(3));
      if (!forwardMode)
        source = lookup(source, Builder2);

      Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
      if (!forwardMode)
        tag = lookup(tag, Builder2);

      Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      if (forwardMode) {
        createMPIForwardCall(call, {/*buf*/ 0}, gutils, Builder2);
        return;
      }

      Value *args[] = {shadow, count, datatype, source, tag, comm};

      auto Defs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal, ValueType::Primal, ValueType::Primal,
           ValueType::None},
          Builder2, /*lookup*/ !forwardMode);

      Type *types[sizeof(args) / sizeof(*args)];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
        types[i] = args[i]->getType();
      FunctionType *FT = FunctionType::get(call.getType(), types, false);

      auto fcall = Builder2.CreateCall(
          called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Send"), FT),
          args, Defs);
      fcall->setCallingConv(call.getCallingConv());

      auto dst_arg =
          Builder2.CreateBitCast(args[0], getInt8PtrTy(call.getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto len_arg = Builder2.CreateZExtOrTrunc(
          args[1], Type::getInt64Ty(call.getContext()));
      auto tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);
      auto volatile_arg = ConstantInt::getFalse(call.getContext());

      Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
      Type *tys[] = {dst_arg->getType(), len_arg->getType()};

      auto MemsetDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::None, ValueType::None, ValueType::None,
           ValueType::None, ValueType::None, ValueType::None},
          Builder2, /*lookup*/ true);
      auto memset = cast<CallInst>(Builder2.CreateCall(
          getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                  Intrinsic::memset, tys),
          nargs));
      memset->addParamAttr(0, Attribute::NonNull);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root,
  //           MPI_Comm comm )
  // 1. if root, malloc intermediate buffer
  // 2. reduce sum diff(buffer) into intermediate
  // 3. if root, set shadow(buffer) = intermediate [memcpy] then free
  // 3-e. else, set shadow(buffer) = 0 [memset]
  if (funcName == "MPI_Bcast" || funcName == "PMPI_Bcast") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
      if (!forwardMode)
        shadow = lookup(shadow, Builder2);
      if (shadow->getType()->isIntegerTy())
        shadow =
            Builder2.CreateIntToPtr(shadow, getInt8PtrTy(call.getContext()));

      ConcreteType CT =
          TR.firstPointer(1, call.getOperand(0), &call, gutils, &Builder2);
      auto MPI_OP_type = getInt8PtrTy(call.getContext());
      Type *MPI_OP_Ptr_type = getUnqual(MPI_OP_type);

      Value *count = gutils->getNewFromOriginal(call.getOperand(1));
      if (!forwardMode)
        count = lookup(count, Builder2);
      Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);
      Value *root = gutils->getNewFromOriginal(call.getOperand(3));
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(call.getOperand(4));
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      if (forwardMode) {
        createMPIForwardCall(call, {/*buffer*/ 0}, gutils, Builder2);
        return;
      }

      Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType(), called);
      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      auto len_arg = Builder2.CreateZExtOrTrunc(
          count, Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // 1. if root, malloc intermediate buffer, else undef
      PHINode *buf;

      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        Value *rootbuf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             len_arg, "mpireduce_malloccache");
        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(mergeBlock);

        buf = Builder2.CreatePHI(rootbuf->getType(), 2);
        buf->addIncoming(rootbuf, rootBlock);
        buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);
      }

      // Need to preserve the shadow buffer.
      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal, ValueType::Primal},
          Builder2, /*lookup*/ true);

      // 2. reduce sum diff(buffer) into intermediate
      {
        // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
        // MPI_Datatype datatype,
        //     MPI_Op op, int root, MPI_Comm comm)
        Value *args[] = {
            /*sendbuf*/ shadow,
            /*recvbuf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*op (MPI_SUM)*/
            getOrInsertOpFloatSum(*gutils->newFunc->getParent(), called,
                                  MPI_OP_Ptr_type, MPI_OP_type, CT,
                                  root->getType(), Builder2),
            /*int root*/ root,
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);

        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Reduce"), FT),
            args, BufferDefs);
      }

      // 3. if root, set shadow(buffer) = intermediate [memcpy]
      BasicBlock *currentBlock = Builder2.GetInsertBlock();
      BasicBlock *rootBlock = gutils->addReverseBlock(
          currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
      BasicBlock *nonrootBlock = gutils->addReverseBlock(
          rootBlock, currentBlock->getName() + "_nonroot", gutils->newFunc);
      BasicBlock *mergeBlock = gutils->addReverseBlock(
          nonrootBlock, currentBlock->getName() + "_post", gutils->newFunc);

      Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                            nonrootBlock);

      Builder2.SetInsertPoint(rootBlock);

      {
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *nargs[] = {shadow, buf, len_arg, volatile_arg};

        Type *tys[] = {shadow->getType(), buf->getType(), len_arg->getType()};

        auto memcpyF = getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                               Intrinsic::memcpy, tys);

        auto mem =
            cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
        mem->setCallingConv(memcpyF->getCallingConv());

        // Free up the memory of the buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }
      }

      Builder2.CreateBr(mergeBlock);

      Builder2.SetInsertPoint(nonrootBlock);

      // 3-e. else, set shadow(buffer) = 0 [memset]
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto volatile_arg = ConstantInt::getFalse(call.getContext());
      Value *args[] = {shadow, val_arg, len_arg, volatile_arg};
      Type *tys[] = {args[0]->getType(), args[2]->getType()};
      auto memset = cast<CallInst>(Builder2.CreateCall(
          getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                  Intrinsic::memset, tys),
          args, BufferDefs));
      memset->addParamAttr(0, Attribute::NonNull);
      Builder2.CreateBr(mergeBlock);

      Builder2.SetInsertPoint(mergeBlock);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffer
  // 1.5 if root, set intermediate = diff(recvbuffer)
  // 2. MPI_Bcast intermediate to all
  // 3. if root, Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
  // MPI_Datatype datatype,
  //                      MPI_Op op, int root, MPI_Comm comm)

  llvm::StringRef canonMPIName = canonicalizeMPIName(funcName);

  if (canonMPIName == "MPI_Reduce") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      // TODO insert a check for sum

      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      // Get the operations from MPI_Reduce
      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_recvbuf = call.getOperand(1);
      Value *orig_count = call.getOperand(2);
      Value *orig_datatype = call.getOperand(3);
      Value *orig_op = call.getOperand(4);
      Value *orig_root = call.getOperand(5);
      Value *orig_comm = call.getOperand(6);

      // The Fortran MPI ABI ("mpi_reduce_", "mpi_reduce__", ...) passes all
      // arguments by reference and appends an `ierr` argument.
      bool fortranABI = isFortranMPICall(called->getName());

      bool isSum = false;
      if (Constant *C = dyn_cast<Constant>(orig_op)) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_op_sum") {
            isSum = true;
          } else if (fortranABI && GV->isConstant() &&
                     GV->hasDefinitiveInitializer()) {
            // The Fortran ABI passes the operator as a reference to an
            // integer handle
            if (auto *CI = dyn_cast<ConstantInt>(GV->getInitializer())) {
              // MPICH native ABI (also covers the MPICH ABI Compatibility
              // Initiative: Intel MPI, MVAPICH, Cray MPICH)
              if (CI->getValue() == 1476395011) {
                isSum = true;
              }
              // MPI 5.0 standard ABI (Chapter 20), where predefined op
              // handles are fixed compile-time constants, identical in C
              // and Fortran: MPI_SUM == 33.
              if (CI->getValue() == 33) {
                isSum = true;
              }
            }
          }
        }
        // MPICH native ABI
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 1476395011) {
            isSum = true;
          }
        }
        // MPI 5.0 standard ABI: predefined op handles are small-integer
        // pointer constants (inttoptr), with MPI_SUM == 33.
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 33) {
            isSum = true;
          }
        }
      }
      if (!isSum) {
        if (fortranABI) {
          // Integer MPI operator handles of a native Fortran ABI are
          // implementation-defined and cannot be mapped portably at compile
          // time; warn and assume the common case of MPI_SUM. This remains
          // true with MPI 5.0: although its standard ABI (Chapter 20) does
          // fix handle values as portable compile-time constants (MPI_SUM
          // == 33, recognized above), supporting that ABI is optional and
          // neither Open MPI nor MPICH use it by default, so code compiled
          // against the default/native ABIs still carries
          // implementation-defined handle values.
          llvm::errs() << "warning: cannot determine MPI op used in `" << call
                       << "`, assuming MPI_SUM\n";
        } else {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_reduce op: " << *orig_op << "\n";
          EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
          return;
        }
      }

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so build the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[1] = ValueType::Shadow;
      auto BufferDefs =
          gutils->getInvertedBundles(&call, BufferBundleTypes, Builder2,
                                     /*lookup*/ !forwardMode);

      Value *count = gutils->getNewFromOriginal(orig_count);
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(orig_datatype);
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *op = gutils->getNewFromOriginal(orig_op);
      if (!forwardMode)
        op = lookup(op, Builder2);

      Value *root = gutils->getNewFromOriginal(orig_root);
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      // The Fortran ABI passes the count and root arguments by reference;
      // load them where their value is needed.
      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *countVal = fortranABI ? Builder2.CreateLoad(i32Ty, count) : count;
      Value *rootVal = fortranABI ? Builder2.CreateLoad(i32Ty, root) : root;

      Value *rank = MPI_COMM_RANK(comm, Builder2, rootVal->getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 1}, gutils,
                             Builder2);
        return;
      }

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      // Get the length for the allocation of the intermediate buffer
      auto len_arg = Builder2.CreateZExtOrTrunc(
          countVal, Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           len_arg, "mpireduce_malloccache");

      // 1.5 if root, set intermediate = diff(recvbuffer)
      {

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        {
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *nargs[] = {buf, shadow_recvbuf, len_arg, volatile_arg};

          Type *tys[] = {nargs[0]->getType(), nargs[1]->getType(),
                         len_arg->getType()};

          auto memcpyF = getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                                 Intrinsic::memcpy, tys);

          auto mem =
              cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
          mem->setCallingConv(memcpyF->getCallingConv());
        }

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }

      // 2. MPI_Bcast intermediate to all
      {
        // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int
        // root,
        //     MPI_Comm comm )
        // The Fortran ABI passes count/datatype/root/comm by reference (as
        // held here) and appends an `ierr` argument.
        SmallVector<Value *, 8> args = {
            /*buf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*int root*/ root,
            /*comm*/ comm,
        };
        if (fortranABI) {
          args.push_back(IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(i32Ty, nullptr, "enzyme_mpi_ierr"));
        }
        SmallVector<Type *, 8> types;
        for (auto *arg : args) {
          types.push_back(arg->getType());
        }

        FunctionType *FT = FunctionType::get(
            fortranABI ? Type::getVoidTy(call.getContext()) : call.getType(),
            types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Bcast"), FT),
            args, BufferDefs);
      }

      // 3. if root, Zero diff(recvbuffer) [memset to 0]
      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  len_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffers
  // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
  // 3. Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
  //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)

  if (canonMPIName == "MPI_Allreduce") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      // TODO insert a check for sum

      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      // Get the operations from MPI_Allreduce
      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_recvbuf = call.getOperand(1);
      Value *orig_count = call.getOperand(2);
      Value *orig_datatype = call.getOperand(3);
      Value *orig_op = call.getOperand(4);
      Value *orig_comm = call.getOperand(5);

      // The Fortran MPI ABI ("mpi_allreduce_", "mpi_allreduce__", ...) passes
      // all arguments by reference and appends an `ierr` argument.
      bool fortranABI = isFortranMPICall(called->getName());

      bool isSum = false;
      if (Constant *C = dyn_cast<Constant>(orig_op)) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_op_sum") {
            isSum = true;
          } else if (fortranABI && GV->isConstant() &&
                     GV->hasDefinitiveInitializer()) {
            // The Fortran ABI passes the operator as a reference to an
            // integer handle
            if (auto *CI = dyn_cast<ConstantInt>(GV->getInitializer())) {
              // MPICH native ABI (also covers the MPICH ABI Compatibility
              // Initiative: Intel MPI, MVAPICH, Cray MPICH)
              if (CI->getValue() == 1476395011) {
                isSum = true;
              }
              // MPI 5.0 standard ABI (Chapter 20), where predefined op
              // handles are fixed compile-time constants, identical in C
              // and Fortran: MPI_SUM == 33.
              if (CI->getValue() == 33) {
                isSum = true;
              }
            }
          }
        }
        // MPICH native ABI
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 1476395011) {
            isSum = true;
          }
        }
        // MPI 5.0 standard ABI: predefined op handles are small-integer
        // pointer constants (inttoptr), with MPI_SUM == 33.
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 33) {
            isSum = true;
          }
        }
      }
      if (!isSum) {
        if (fortranABI) {
          // Integer MPI operator handles of a native Fortran ABI are
          // implementation-defined and cannot be mapped portably at compile
          // time; warn and assume the common case of MPI_SUM. This remains
          // true with MPI 5.0: although its standard ABI (Chapter 20) does
          // fix handle values as portable compile-time constants (MPI_SUM
          // == 33, recognized above), supporting that ABI is optional and
          // neither Open MPI nor MPICH use it by default, so code compiled
          // against the default/native ABIs still carries
          // implementation-defined handle values.
          llvm::errs() << "warning: cannot determine MPI op used in `" << call
                       << "`, assuming MPI_SUM\n";
        } else {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
          return;
        }
      }

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so build the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[1] = ValueType::Shadow;
      auto BufferDefs =
          gutils->getInvertedBundles(&call, BufferBundleTypes, Builder2,
                                     /*lookup*/ !forwardMode);

      Value *count = gutils->getNewFromOriginal(orig_count);
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(orig_datatype);
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      Value *op = gutils->getNewFromOriginal(orig_op);
      if (!forwardMode)
        op = lookup(op, Builder2);

      // The Fortran ABI passes the count argument by reference; load it
      // where its value is needed.
      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *countVal = fortranABI ? Builder2.CreateLoad(i32Ty, count) : count;

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 1}, gutils,
                             Builder2);
        return;
      }

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      // Get the length for the allocation of the intermediate buffer
      auto len_arg = Builder2.CreateZExtOrTrunc(
          countVal, Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           len_arg, "mpireduce_malloccache");

      // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
      {
        // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
        //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
        // The Fortran ABI passes count/datatype/op/comm by reference (as
        // held here) and appends an `ierr` argument.
        SmallVector<Value *, 8> args = {
            /*sendbuf*/ shadow_recvbuf,
            /*recvbuf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*op*/ op,
            /*comm*/ comm,
        };
        if (fortranABI) {
          args.push_back(IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(i32Ty, nullptr, "enzyme_mpi_ierr"));
        }
        SmallVector<Type *, 8> types;
        for (auto *arg : args)
          types.push_back(arg->getType());

        FunctionType *FT = FunctionType::get(
            fortranABI ? Type::getVoidTy(call.getContext()) : call.getType(),
            types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Allreduce"),
                FT),
            args, BufferDefs);
      }

      // 3. Zero diff(recvbuffer) [memset to 0]
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto volatile_arg = ConstantInt::getFalse(call.getContext());
      Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
      Type *tys[] = {args[0]->getType(), args[2]->getType()};
      auto memset = cast<CallInst>(Builder2.CreateCall(
          getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                  Intrinsic::memset, tys),
          args, BufferDefs));
      memset->addParamAttr(0, Attribute::NonNull);

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  len_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffer
  // 2. Scatter diff(recvbuffer) to intermediate buffer
  // 3. if root, Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
  //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
  //           int root, MPI_Comm comm)

  if (funcName == "MPI_Gather" || funcName == "PMPI_Gather") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_sendcount = call.getOperand(1);
      Value *orig_sendtype = call.getOperand(2);
      Value *orig_recvbuf = call.getOperand(3);
      Value *orig_recvcount = call.getOperand(4);
      Value *orig_recvtype = call.getOperand(5);
      Value *orig_root = call.getOperand(6);
      Value *orig_comm = call.getOperand(7);

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
      if (!forwardMode)
        recvcount = lookup(recvcount, Builder2);

      Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
      if (!forwardMode)
        recvtype = lookup(recvtype, Builder2);

      Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
      if (!sendcount)
        sendcount = lookup(sendcount, Builder2);

      Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
      if (!forwardMode)
        sendtype = lookup(sendtype, Builder2);

      bool fortranABI = isFortranMPICall(called->getName());

      Value *root = gutils->getNewFromOriginal(orig_root);
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *sendcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, sendcount) : sendcount;
      Value *recvcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, recvcount) : recvcount;
      Value *rootVal = fortranABI ? Builder2.CreateLoad(i32Ty, root) : root;

      Value *rank = MPI_COMM_RANK(comm, Builder2, rootVal->getType(), called);
      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 3}, gutils,
                             Builder2);
        return;
      }

      // Get the length for the allocation of the intermediate buffer
      auto sendlen_arg = Builder2.CreateZExtOrTrunc(
          sendcountVal, Type::getInt64Ty(call.getContext()));
      sendlen_arg =
          Builder2.CreateMul(sendlen_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so size the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[3] = ValueType::Shadow;
      auto BufferDefs = gutils->getInvertedBundles(&call, BufferBundleTypes,
                                                   Builder2, /*lookup*/ true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           sendlen_arg, "mpireduce_malloccache");

      // 2. Scatter diff(recvbuffer) to intermediate buffer
      {
        // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
        // sendtype,
        //     void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
        //     MPI_Comm comm)
        //
        // The Fortran MPI ABI passes all arguments by reference and appends
        // an `ierr` argument, so the generated call must match the convention
        // of the caller.
        SmallVector<Value *, 10> args;
        if (fortranABI) {
          args = {shadow_recvbuf, recvcount, recvtype, buf,
                  sendcount,      sendtype,  root,     comm};
          for (size_t i = 8, e = call.arg_size(); i < e; i++)
            args.push_back(lookup(
                gutils->getNewFromOriginal(call.getArgOperand(i)), Builder2));
        } else {
          args = {shadow_recvbuf, recvcountVal, recvtype, buf,
                  sendcountVal,   sendtype,     rootVal,  comm};
        }
        SmallVector<Type *, 10> types;
        for (auto *arg : args)
          types.push_back(arg->getType());

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Scatter"), FT),
            args, BufferDefs);
      }

      // 3. if root, Zero diff(recvbuffer) [memset to 0]
      {

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);
        auto recvlen_arg = Builder2.CreateZExtOrTrunc(
            recvcountVal, Type::getInt64Ty(call.getContext()));
        recvlen_arg =
            Builder2.CreateMul(recvlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        recvlen_arg = Builder2.CreateMul(
            recvlen_arg,
            Builder2.CreateZExtOrTrunc(
                MPI_COMM_SIZE(comm, Builder2, rootVal->getType(), called),
                Type::getInt64Ty(call.getContext())),
            "", true, true);

        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  sendlen_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. if root, malloc intermediate buffer, else undef
  // 2. Gather diff(recvbuffer) to intermediate buffer
  // 3. Zero diff(recvbuffer) [memset to 0]
  // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. if root, free intermediate buffer

  // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
  // sendtype,
  //           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
  //           MPI_Comm comm)
  if (canonMPIName == "MPI_Scatter") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_sendcount = call.getOperand(1);
      Value *orig_sendtype = call.getOperand(2);
      Value *orig_recvbuf = call.getOperand(3);
      Value *orig_recvcount = call.getOperand(4);
      Value *orig_recvtype = call.getOperand(5);
      Value *orig_root = call.getOperand(6);
      Value *orig_comm = call.getOperand(7);

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
      if (!forwardMode)
        recvcount = lookup(recvcount, Builder2);

      Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
      if (!forwardMode)
        recvtype = lookup(recvtype, Builder2);

      Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
      if (!forwardMode)
        sendcount = lookup(sendcount, Builder2);

      Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
      if (!forwardMode)
        sendtype = lookup(sendtype, Builder2);

      Value *root = gutils->getNewFromOriginal(orig_root);
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      bool fortranABI = isFortranMPICall(called->getName());

      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *sendcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, sendcount) : sendcount;
      Value *recvcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, recvcount) : recvcount;
      Value *rootVal = fortranABI ? Builder2.CreateLoad(i32Ty, root) : root;

      Value *rank = MPI_COMM_RANK(comm, Builder2, rootVal->getType(), called);
      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 3}, gutils,
                             Builder2);
        return;
      }
      // Get the length for the allocation of the intermediate buffer
      auto recvlen_arg = Builder2.CreateZExtOrTrunc(
          recvcountVal, Type::getInt64Ty(call.getContext()));
      recvlen_arg =
          Builder2.CreateMul(recvlen_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so size the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[3] = ValueType::Shadow;
      auto BufferDefs = gutils->getInvertedBundles(&call, BufferBundleTypes,
                                                   Builder2, /*lookup*/ true);

      // 1. if root, malloc intermediate buffer, else undef
      PHINode *buf;
      PHINode *sendlen_phi;

      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcountVal, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        sendlen_arg = Builder2.CreateMul(
            sendlen_arg,
            Builder2.CreateZExtOrTrunc(
                MPI_COMM_SIZE(comm, Builder2, rootVal->getType(), called),
                Type::getInt64Ty(call.getContext())),
            "", true, true);

        Value *rootbuf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             sendlen_arg, "mpireduce_malloccache");

        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(mergeBlock);

        buf = Builder2.CreatePHI(rootbuf->getType(), 2);
        buf->addIncoming(rootbuf, rootBlock);
        buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);

        sendlen_phi = Builder2.CreatePHI(sendlen_arg->getType(), 2);
        sendlen_phi->addIncoming(sendlen_arg, rootBlock);
        sendlen_phi->addIncoming(UndefValue::get(sendlen_arg->getType()),
                                 currentBlock);
      }

      // 2. Gather diff(recvbuffer) to intermediate buffer
      {
        // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype
        // sendtype,
        //     void *recvbuf, int recvcount, MPI_Datatype recvtype,
        //     int root, MPI_Comm comm)
        //
        // The Fortran MPI ABI passes all arguments by reference and appends
        // an `ierr` argument, so the generated call must match the convention
        // of the caller.
        SmallVector<Value *, 10> args;
        if (fortranABI) {
          args = {shadow_recvbuf, recvcount, recvtype, buf,
                  sendcount,      sendtype,  root,     comm};
          for (size_t i = 8, e = call.arg_size(); i < e; i++)
            args.push_back(lookup(
                gutils->getNewFromOriginal(call.getArgOperand(i)), Builder2));
        } else {
          args = {shadow_recvbuf, recvcountVal, recvtype, buf,
                  sendcountVal,   sendtype,     rootVal,  comm};
        }
        SmallVector<Type *, 10> types;
        for (auto *arg : args)
          types.push_back(arg->getType());

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Gather"), FT),
            args, BufferDefs);
      }

      // 3. Zero diff(recvbuffer) [memset to 0]
      {
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);
      }

      // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
      // 5. if root, free intermediate buffer

      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_phi, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffer
  // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
  // intermediate buffer
  // 3. Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype
  // sendtype,
  //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
  //           MPI_Comm comm)

  if (funcName == "MPI_Allgather" || funcName == "PMPI_Allgather") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_sendcount = call.getOperand(1);
      Value *orig_sendtype = call.getOperand(2);
      Value *orig_recvbuf = call.getOperand(3);
      Value *orig_recvcount = call.getOperand(4);
      Value *orig_recvtype = call.getOperand(5);
      Value *orig_comm = call.getOperand(6);

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);

      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);

      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
      if (!forwardMode)
        recvcount = lookup(recvcount, Builder2);

      Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
      if (!forwardMode)
        recvtype = lookup(recvtype, Builder2);

      Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
      if (!forwardMode)
        sendcount = lookup(sendcount, Builder2);

      Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
      if (!forwardMode)
        sendtype = lookup(sendtype, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 3}, gutils,
                             Builder2);
        return;
      }
      // Get the length for the allocation of the intermediate buffer
      auto sendlen_arg = Builder2.CreateZExtOrTrunc(
          sendcount, Type::getInt64Ty(call.getContext()));
      sendlen_arg =
          Builder2.CreateMul(sendlen_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // Need to preserve the shadow send/recv buffers.
      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal},
          Builder2, /*lookup*/ true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           sendlen_arg, "mpireduce_malloccache");

      ConcreteType CT =
          TR.firstPointer(1, orig_sendbuf, &call, gutils, &Builder2);
      auto MPI_OP_type = getInt8PtrTy(call.getContext());
      Type *MPI_OP_Ptr_type = getUnqual(MPI_OP_type);

      // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
      // intermediate buffer
      {
        // int MPI_Reduce_scatter_block(const void* send_buffer,
        //                    void* receive_buffer,
        //                    int count,
        //                    MPI_Datatype datatype,
        //                    MPI_Op operation,
        //                    MPI_Comm communicator);
        Value *args[] = {
            /*sendbuf*/ shadow_recvbuf,
            /*recvbuf*/ buf,
            /*recvcount*/ sendcount,
            /*recvtype*/ sendtype,
            /*op (MPI_SUM)*/
            getOrInsertOpFloatSum(*gutils->newFunc->getParent(), called,
                                  MPI_OP_Ptr_type, MPI_OP_type, CT,
                                  call.getType(), Builder2),
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(),
                                         "MPI_Reduce_scatter_block"),
                FT),
            args, BufferDefs);
      }

      // 3. zero diff(recvbuffer) [memset to 0]
      {
        auto recvlen_arg = Builder2.CreateZExtOrTrunc(
            recvcount, Type::getInt64Ty(call.getContext()));
        recvlen_arg =
            Builder2.CreateMul(recvlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        recvlen_arg = Builder2.CreateMul(
            recvlen_arg,
            Builder2.CreateZExtOrTrunc(
                MPI_COMM_SIZE(comm, Builder2, call.getType(), called),
                Type::getInt64Ty(call.getContext())),
            "", true, true);
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);
      }

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  sendlen_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Adjoint of barrier is to place a barrier at the corresponding
  // location in the reverse.
  if (funcName == "MPI_Barrier" || funcName == "PMPI_Barrier") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      auto callval = call.getCalledOperand();
      // Copy all arguments to match the call's arity: Fortran ABI manglings
      // of MPI_Barrier (e.g. "mpi_barrier_") take an extra `ierr` argument.
      SmallVector<Value *, 4> args;
      for (unsigned i = 0; i < call.arg_size(); i++)
        args.push_back(lookup(gutils->getNewFromOriginal(call.getArgOperand(i)),
                              Builder2));
      Builder2.CreateCall(call.getFunctionType(), callval, args);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Remove free's in forward pass so the comm can be used in the reverse
  // pass
  if (funcName == "MPI_Comm_free" || funcName == "MPI_Comm_disconnect") {
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Adjoint of MPI_Comm_split / MPI_Graph_create (which allocates a comm in a
  // pointer) is to free the created comm at the corresponding place in the
  // reverse pass
  auto commFound = MPIInactiveCommAllocators.find(funcName);
  if (commFound != MPIInactiveCommAllocators.end()) {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      Value *args[] = {lookup(call.getOperand(commFound->second), Builder2)};
      Type *types[] = {args[0]->getType()};

      FunctionType *FT = FunctionType::get(call.getType(), types, false);
      Builder2.CreateCall(
          called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Comm_free"), FT),
          args);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // MPI_Init / MPI_Finalize don't participate in the computation being
  // differentiated - just duplicate them as-is in both passes.
  if (funcName == "MPI_Init" || funcName == "PMPI_Init" ||
      funcName == "MPI_Init_thread" || funcName == "PMPI_Init_thread" ||
      funcName == "MPI_Finalize" || funcName == "PMPI_Finalize" ||
      funcName == "MPI_Test" || funcName == "PMPI_Test" ||
      funcName == "MPI_Probe" || funcName == "PMPI_Probe") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ReverseModePrimal) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      SmallVector<Value *, 8> args;
      for (unsigned i = 0; i < call.arg_size(); ++i) {
        args.push_back(lookup(gutils->getNewFromOriginal(call.getArgOperand(i)),
                              Builder2));
      }
      Builder2.CreateCall(call.getFunctionType(), call.getCalledOperand(),
                          args);
    }
    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError ||
        Mode == DerivativeMode::ForwardModeSplit) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);
      SmallVector<Value *, 8> args;
      for (unsigned i = 0; i < call.arg_size(); ++i) {
        args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
      }
      Builder2.CreateCall(call.getFunctionType(), call.getCalledOperand(),
                          args);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  llvm::errs() << *gutils->oldFunc->getParent() << "\n";
  llvm::errs() << *gutils->oldFunc << "\n";
  llvm::errs() << call << "\n";
  llvm::errs() << called << "\n";
  llvm_unreachable("Unhandled MPI FUNCTION");
}

// Classify the op callee of a julia.atomicmodify pseudo-intrinsic call
// (op has signature elty (elty oldval, args...), with elty an integer type
// holding the bits of the modified value) as an atomicrmw-style operation
// whose derivative Enzyme knows. On success sets opArgNo to the index of the
// op parameter providing the update value and FT to the floating point type
// the update is performed in (nullptr for Xchg).
static llvm::AtomicRMWInst::BinOp
classifyAtomicModifyOp(llvm::Function *op, unsigned &opArgNo, llvm::Type *&FT) {
  using namespace llvm;
  FT = nullptr;
  if (!op || op->empty() || op->isVarArg() ||
      op->getFunctionType()->getNumParams() == 0)
    return AtomicRMWInst::BAD_BINOP;
  ReturnInst *Ret = nullptr;
  for (auto &BB : *op)
    if (auto R = dyn_cast<ReturnInst>(BB.getTerminator())) {
      if (Ret)
        return AtomicRMWInst::BAD_BINOP;
      Ret = R;
    }
  if (!Ret || !Ret->getReturnValue())
    return AtomicRMWInst::BAD_BINOP;
  auto peel = [](Value *V) {
    while (auto BC = dyn_cast<BitCastInst>(V))
      V = BC->getOperand(0);
    return V;
  };
  Value *RV = peel(Ret->getReturnValue());
  Value *Old = op->getArg(0);
  if (auto A = dyn_cast<Argument>(RV)) {
    if (A == Old)
      return AtomicRMWInst::BAD_BINOP;
    opArgNo = A->getArgNo();
    return AtomicRMWInst::Xchg;
  }
  if (auto BO = dyn_cast<BinaryOperator>(RV)) {
    Value *L = peel(BO->getOperand(0));
    Value *R = peel(BO->getOperand(1));
    FT = BO->getType();
    if (!FT->isFPOrFPVectorTy())
      return AtomicRMWInst::BAD_BINOP;
    if (BO->getOpcode() == Instruction::FAdd) {
      if (L == Old && isa<Argument>(R) && R != Old) {
        opArgNo = cast<Argument>(R)->getArgNo();
        return AtomicRMWInst::FAdd;
      }
      if (R == Old && isa<Argument>(L) && L != Old) {
        opArgNo = cast<Argument>(L)->getArgNo();
        return AtomicRMWInst::FAdd;
      }
    } else if (BO->getOpcode() == Instruction::FSub) {
      if (L == Old && isa<Argument>(R) && R != Old) {
        opArgNo = cast<Argument>(R)->getArgNo();
        return AtomicRMWInst::FSub;
      }
    }
  }
  return AtomicRMWInst::BAD_BINOP;
}

namespace {
enum class OMPReductionOp {
  Sum,
  Max,
  Min,
  IntInactive,
};

struct OMPReductionItem {
  int64_t byteOffset;
  Value *origPrivPtr;
  Type *elemTy;
  OMPReductionOp op;
};
} // namespace

/// Match \p V as \p Arg directly or a load from an unoptimized stack slot
/// whose only store saves \p Arg (as Clang emits at -O0 in reduction_func).
static bool isOMPReduceArgDataPtr(Value *V, Argument *Arg) {
  V = V->stripPointerCasts();
  if (V == Arg)
    return true;
  auto *LI = dyn_cast<LoadInst>(V);
  if (!LI)
    return false;
  auto *AI = dyn_cast<AllocaInst>(LI->getPointerOperand()->stripPointerCasts());
  if (!AI)
    return false;
  StoreInst *UniqueStore = nullptr;
  for (User *U : AI->users()) {
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand()->stripPointerCasts() != AI)
        return false;
      if (UniqueStore)
        return false;
      UniqueStore = SI;
    } else if (!isa<LoadInst>(U) && !isa<CastInst>(U)) {
      return false;
    }
  }
  return UniqueStore &&
         UniqueStore->getValueOperand()->stripPointerCasts() == Arg;
}

/// Match \p V as the pointer loaded from `(char*)Arg + byteOff` in the
/// array-of-pointers OpenMP reduction ABI.
static bool matchOMPReduceSlotPtr(Value *V, Argument *Arg, int64_t &byteOff,
                                  const DataLayout &DL) {
  auto *LI = dyn_cast<LoadInst>(V->stripPointerCasts());
  if (!LI || !LI->getType()->isPointerTy())
    return false;
  int64_t off = 0;
  Value *Base =
      GetPointerBaseWithConstantOffset(LI->getPointerOperand(), off, DL);
  if (!isOMPReduceArgDataPtr(Base, Arg))
    return false;
  byteOff = off;
  return true;
}

static bool classifyOMPMaxMinPred(FCmpInst::Predicate Pred, bool truePicksArg0,
                                  OMPReductionOp &outOp) {
  switch (Pred) {
  case FCmpInst::FCMP_OGT:
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_OGE:
  case FCmpInst::FCMP_UGE:
    outOp = truePicksArg0 ? OMPReductionOp::Max : OMPReductionOp::Min;
    return true;
  case FCmpInst::FCMP_OLT:
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_OLE:
  case FCmpInst::FCMP_ULE:
    outOp = truePicksArg0 ? OMPReductionOp::Min : OMPReductionOp::Max;
    return true;
  default:
    return false;
  }
}

/// Inspect an OpenMP reduction combiner `void reduce_func(void *lhs, void
/// *rhs)` and extract each reduced variable's caller private pointer, scalar
/// type, and reduction operator. Returns false if any store in `reduce_func` is
/// not a recognized reduction update.
static bool extractOMPReductionItems(CallInst &call, Function *reduceFn,
                                     DominatorTree *OrigDT,
                                     const DataLayout &DL,
                                     SmallVectorImpl<OMPReductionItem> &items) {
  if (!reduceFn || reduceFn->empty() || reduceFn->arg_size() != 2)
    return false;

  Argument *Arg0 = reduceFn->getArg(0);
  Argument *Arg1 = reduceFn->getArg(1);

  int64_t callArrOff = 0;
  Value *callArrBase =
      GetPointerBaseWithConstantOffset(call.getArgOperand(4), callArrOff, DL);

  auto findCallerPrivPtr = [&](int64_t slotByteOff, bool indirect) -> Value * {
    if (!indirect) {
      if (slotByteOff == 0)
        return call.getArgOperand(4);
      return nullptr;
    }
    int64_t targetOff = callArrOff + slotByteOff;
    // Prefer the closest store preceding `call` in the same block, falling back
    // to stores in strictly dominating blocks.
    BasicBlock *CallBB = call.getParent();
    for (auto It = call.getReverseIterator(); It != CallBB->rend(); ++It) {
      if (&*It == &call)
        continue;
      if (auto *SI = dyn_cast<StoreInst>(&*It)) {
        int64_t stOff = 0;
        Value *stBase = GetPointerBaseWithConstantOffset(
            SI->getPointerOperand(), stOff, DL);
        if (stBase == callArrBase && stOff == targetOff)
          return SI->getValueOperand()->stripPointerCasts();
      }
    }
    Value *Found = nullptr;
    for (BasicBlock &BB : *CallBB->getParent()) {
      if (&BB == CallBB || (OrigDT && !OrigDT->dominates(&BB, CallBB)))
        continue;
      for (Instruction &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          int64_t stOff = 0;
          Value *stBase = GetPointerBaseWithConstantOffset(
              SI->getPointerOperand(), stOff, DL);
          if (stBase == callArrBase && stOff == targetOff)
            Found = SI->getValueOperand()->stripPointerCasts();
        }
      }
    }
    return Found;
  };

  for (BasicBlock &BB : *reduceFn) {
    for (Instruction &I : BB) {
      auto *SI = dyn_cast<StoreInst>(&I);
      if (!SI)
        continue;

      // Ignore -O0 argument-spill stores into local allocas.
      if (isa<AllocaInst>(SI->getPointerOperand()->stripPointerCasts())) {
        Value *Spilled = SI->getValueOperand()->stripPointerCasts();
        if (Spilled == Arg0 || Spilled == Arg1)
          continue;
        return false;
      }

      int64_t byteOff = 0;
      bool indirect = true;
      if (!matchOMPReduceSlotPtr(SI->getPointerOperand(), Arg0, byteOff, DL)) {
        int64_t directOff = 0;
        Value *DirectBase = GetPointerBaseWithConstantOffset(
            SI->getPointerOperand(), directOff, DL);
        if (!isOMPReduceArgDataPtr(DirectBase, Arg0))
          return false;
        byteOff = directOff;
        indirect = false;
      }

      Value *Val = SI->getValueOperand();
      Type *elemTy = Val->getType();

      auto isLoadOfArg = [&](Value *V, Argument *WhichArg) -> bool {
        auto *LI = dyn_cast<LoadInst>(V);
        if (!LI || LI->getType() != elemTy)
          return false;
        if (indirect) {
          int64_t off = 0;
          return matchOMPReduceSlotPtr(LI->getPointerOperand(), WhichArg, off,
                                       DL) &&
                 off == byteOff;
        }
        int64_t off = 0;
        Value *Base =
            GetPointerBaseWithConstantOffset(LI->getPointerOperand(), off, DL);
        return isOMPReduceArgDataPtr(Base, WhichArg) && off == byteOff;
      };

      auto matchFCmpOnArgs = [&](Value *Cond,
                                 FCmpInst::Predicate &NormPred) -> bool {
        auto *FC = dyn_cast<FCmpInst>(Cond);
        if (!FC)
          return false;
        Value *Op0 = FC->getOperand(0);
        Value *Op1 = FC->getOperand(1);
        if (isLoadOfArg(Op0, Arg0) && isLoadOfArg(Op1, Arg1)) {
          NormPred = FC->getPredicate();
          return true;
        }
        if (isLoadOfArg(Op0, Arg1) && isLoadOfArg(Op1, Arg0)) {
          NormPred = FC->getSwappedPredicate();
          return true;
        }
        return false;
      };

      OMPReductionOp opKind;
      if (elemTy->isIntOrIntVectorTy()) {
        opKind = OMPReductionOp::IntInactive;
      } else if (!elemTy->isFloatingPointTy()) {
        return false;
      } else if (auto *BO = dyn_cast<BinaryOperator>(Val)) {
        if (BO->getOpcode() == Instruction::FAdd &&
            ((isLoadOfArg(BO->getOperand(0), Arg0) &&
              isLoadOfArg(BO->getOperand(1), Arg1)) ||
             (isLoadOfArg(BO->getOperand(0), Arg1) &&
              isLoadOfArg(BO->getOperand(1), Arg0)))) {
          opKind = OMPReductionOp::Sum;
        } else {
          return false;
        }
      } else if (auto *II = dyn_cast<IntrinsicInst>(Val)) {
        Intrinsic::ID IID = II->getIntrinsicID();
        if ((IID == Intrinsic::maxnum || IID == Intrinsic::maximum ||
             IID == Intrinsic::minnum || IID == Intrinsic::minimum) &&
            ((isLoadOfArg(II->getArgOperand(0), Arg0) &&
              isLoadOfArg(II->getArgOperand(1), Arg1)) ||
             (isLoadOfArg(II->getArgOperand(0), Arg1) &&
              isLoadOfArg(II->getArgOperand(1), Arg0)))) {
          opKind = (IID == Intrinsic::maxnum || IID == Intrinsic::maximum)
                       ? OMPReductionOp::Max
                       : OMPReductionOp::Min;
        } else {
          return false;
        }
      } else if (auto *Sel = dyn_cast<SelectInst>(Val)) {
        FCmpInst::Predicate NormPred;
        if (!matchFCmpOnArgs(Sel->getCondition(), NormPred))
          return false;
        bool truePicksArg0 = false;
        if (isLoadOfArg(Sel->getTrueValue(), Arg0) &&
            isLoadOfArg(Sel->getFalseValue(), Arg1)) {
          truePicksArg0 = true;
        } else if (isLoadOfArg(Sel->getTrueValue(), Arg1) &&
                   isLoadOfArg(Sel->getFalseValue(), Arg0)) {
          truePicksArg0 = false;
        } else {
          return false;
        }
        if (!classifyOMPMaxMinPred(NormPred, truePicksArg0, opKind))
          return false;
      } else if (auto *PN = dyn_cast<PHINode>(Val)) {
        if (PN->getNumIncomingValues() != 2)
          return false;
        BasicBlock *B0 = PN->getIncomingBlock(0);
        BasicBlock *B1 = PN->getIncomingBlock(1);
        BasicBlock *Dom = B0->getSinglePredecessor();
        if (!Dom || Dom != B1->getSinglePredecessor() || !hasTerminator(Dom))
          return false;
        auto *BI = dyn_cast<BranchInst>(Dom->getTerminator());
        if (!BI || !BI->isConditional())
          return false;
        FCmpInst::Predicate NormPred;
        if (!matchFCmpOnArgs(BI->getCondition(), NormPred))
          return false;
        Value *TrueV = (BI->getSuccessor(0) == B0) ? PN->getIncomingValue(0)
                                                   : PN->getIncomingValue(1);
        Value *FalseV = (BI->getSuccessor(0) == B0) ? PN->getIncomingValue(1)
                                                    : PN->getIncomingValue(0);
        bool truePicksArg0 = false;
        if (isLoadOfArg(TrueV, Arg0) && isLoadOfArg(FalseV, Arg1)) {
          truePicksArg0 = true;
        } else if (isLoadOfArg(TrueV, Arg1) && isLoadOfArg(FalseV, Arg0)) {
          truePicksArg0 = false;
        } else {
          return false;
        }
        if (!classifyOMPMaxMinPred(NormPred, truePicksArg0, opKind))
          return false;
      } else if (isLoadOfArg(Val, Arg1)) {
        // Conditional overwrite: `if (rhs > lhs) *lhsp = rhs;`
        BasicBlock *StoreBB = SI->getParent();
        BasicBlock *Dom = StoreBB->getSinglePredecessor();
        if (!Dom || !hasTerminator(Dom))
          return false;
        auto *BI = dyn_cast<BranchInst>(Dom->getTerminator());
        if (!BI || !BI->isConditional() || !hasTerminator(StoreBB))
          return false;
        auto *StoreBr = dyn_cast<BranchInst>(StoreBB->getTerminator());
        if (!StoreBr || StoreBr->isConditional())
          return false;
        BasicBlock *MergeBB = StoreBr->getSuccessor(0);
        bool storeOnTrue = false;
        if (BI->getSuccessor(0) == StoreBB && BI->getSuccessor(1) == MergeBB) {
          storeOnTrue = true;
        } else if (BI->getSuccessor(1) == StoreBB &&
                   BI->getSuccessor(0) == MergeBB) {
          storeOnTrue = false;
        } else {
          return false;
        }
        FCmpInst::Predicate NormPred;
        if (!matchFCmpOnArgs(BI->getCondition(), NormPred))
          return false;
        if (!classifyOMPMaxMinPred(NormPred, /*truePicksArg0=*/!storeOnTrue,
                                   opKind))
          return false;
      } else {
        return false;
      }

      Value *origPriv = findCallerPrivPtr(byteOff, indirect);
      if (!origPriv)
        return false;
      items.push_back({byteOff, origPriv, elemTy, opKind});
    }
  }

  if (items.empty())
    return false;
  llvm::sort(items, [](const OMPReductionItem &A, const OMPReductionItem &B) {
    return A.byteOffset < B.byteOffset;
  });
  for (size_t i = 1; i < items.size(); ++i)
    if (items[i].byteOffset == items[i - 1].byteOffset)
      return false;
  return true;
}

static GlobalVariable *getOrCreateOMPReduceLock(Module &M, StringRef Name) {
  if (GlobalVariable *GV = M.getGlobalVariable(Name, /*AllowInternal=*/true))
    return GV;
  ArrayType *LockTy = ArrayType::get(Type::getInt32Ty(M.getContext()), 8);
  auto *GV = new GlobalVariable(M, LockTy, /*isConstant=*/false,
                                GlobalValue::InternalLinkage,
                                ConstantAggregateZero::get(LockTy), Name);
  GV->setAlignment(Align(8));
  return GV;
}

/// Build an OpenMP `void (ptr lhs_bundle, ptr rhs_bundle)` wrapper that unpacks
/// `[primal_data, shadow_data_0, ..., shadow_data_{W-1}]` and calls Enzyme's
/// forward-mode derivative `diffRedFn`.
static Function *getOrCreateOMPFwdReduceWrapper(Module &M,
                                                FunctionType *origRedFT,
                                                Function *diffRedFn,
                                                unsigned width) {
  std::string Name = ("__enzyme_omp_fwd_reduce_" + diffRedFn->getName() + "_w" +
                      std::to_string(width))
                         .str();
  if (Function *Existing = M.getFunction(Name))
    return Existing;

  LLVMContext &Ctx = M.getContext();
  Function *Wrap =
      Function::Create(origRedFT, GlobalValue::InternalLinkage, Name, &M);
  Wrap->addFnAttr(Attribute::NoUnwind);
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Wrap);
  IRBuilder<> B(Entry);

  Type *I8PtrTy = getInt8PtrTy(Ctx);
  ArrayType *BundleTy = ArrayType::get(I8PtrTy, 1 + width);
  Type *BundlePtrTy = getUnqual(BundleTy);

  Value *LhsBundle = Wrap->getArg(0);
  if (LhsBundle->getType() != BundlePtrTy)
    LhsBundle = B.CreatePointerCast(LhsBundle, BundlePtrTy);
  Value *RhsBundle = Wrap->getArg(1);
  if (RhsBundle->getType() != BundlePtrTy)
    RhsBundle = B.CreatePointerCast(RhsBundle, BundlePtrTy);

  FunctionType *DiffFT = diffRedFn->getFunctionType();
  Type *PrimalParamTy0 = DiffFT->getParamType(0);
  Type *ShadowParamTy0 = DiffFT->getParamType(1);
  Type *PrimalParamTy1 = DiffFT->getParamType(2);
  Type *ShadowParamTy1 = DiffFT->getParamType(3);

  auto loadSlot = [&](Value *Bundle, unsigned idx, Type *TargetTy) -> Value * {
    Value *Idxs[] = {B.getInt64(0), B.getInt64(idx)};
    Value *GEP = B.CreateInBoundsGEP(BundleTy, Bundle, Idxs);
    Value *Loaded = B.CreateLoad(I8PtrTy, GEP);
    if (Loaded->getType() != TargetTy)
      Loaded = B.CreatePointerCast(Loaded, TargetTy);
    return Loaded;
  };

  Value *LhsPrimal = loadSlot(LhsBundle, 0, PrimalParamTy0);
  Value *RhsPrimal = loadSlot(RhsBundle, 0, PrimalParamTy1);

  Value *LhsShadow = nullptr;
  Value *RhsShadow = nullptr;
  if (width == 1) {
    LhsShadow = loadSlot(LhsBundle, 1, ShadowParamTy0);
    RhsShadow = loadSlot(RhsBundle, 1, ShadowParamTy1);
  } else {
    Type *LaneTy0 = cast<ArrayType>(ShadowParamTy0)->getElementType();
    Type *LaneTy1 = cast<ArrayType>(ShadowParamTy1)->getElementType();
    LhsShadow = UndefValue::get(ShadowParamTy0);
    RhsShadow = UndefValue::get(ShadowParamTy1);
    for (unsigned w = 0; w < width; ++w) {
      Value *LW = loadSlot(LhsBundle, 1 + w, LaneTy0);
      LhsShadow = B.CreateInsertValue(LhsShadow, LW, {w});
      Value *RW = loadSlot(RhsBundle, 1 + w, LaneTy1);
      RhsShadow = B.CreateInsertValue(RhsShadow, RW, {w});
    }
  }

  Value *CallArgs[] = {LhsPrimal, LhsShadow, RhsPrimal, RhsShadow};
  B.CreateCall(DiffFT, diffRedFn, CallArgs);
  B.CreateRetVoid();
  return Wrap;
}

/// Build an O(log P) reverse-mode combiner `void (ptr lhs, ptr rhs)` over a
/// struct of `M` `{ elTy, [width x elTy], i32 win_tid, i32 init }` records.
/// Sums `d_sum` across all threads and tracks the winning thread `win_tid`
/// achieving the global max or min `pre_val` (ties broken by smallest `gtid`).
static Function *getOrCreateOMPRevMaxMinReduceFn(Module &M,
                                                 FunctionType *origRedFT,
                                                 StructType *allMaxMinTy,
                                                 ArrayRef<bool> itemIsMax,
                                                 unsigned width) {
  LLVMContext &Ctx = M.getContext();
  std::string Sig;
  raw_string_ostream SS(Sig);
  SS << "__enzyme_omp_rev_maxmin_w" << width;
  for (unsigned m = 0; m < itemIsMax.size(); ++m) {
    auto *RecTy = cast<StructType>(allMaxMinTy->getElementType(m));
    SS << "_" << (itemIsMax[m] ? "max" : "min") << "_"
       << RecTy->getElementType(0)->getTypeID();
  }
  SS.flush();
  if (Function *Existing = M.getFunction(Sig))
    return Existing;

  Function *Fn =
      Function::Create(origRedFT, GlobalValue::InternalLinkage, Sig, &M);
  Fn->addFnAttr(Attribute::NoUnwind);
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> B(Entry);

  Type *StructPtrTy = getUnqual(allMaxMinTy);
  Value *LhsBase = Fn->getArg(0);
  if (LhsBase->getType() != StructPtrTy)
    LhsBase = B.CreatePointerCast(LhsBase, StructPtrTy);
  Value *RhsBase = Fn->getArg(1);
  if (RhsBase->getType() != StructPtrTy)
    RhsBase = B.CreatePointerCast(RhsBase, StructPtrTy);

  Type *I32Ty = Type::getInt32Ty(Ctx);

  for (unsigned m = 0; m < itemIsMax.size(); ++m) {
    auto *RecTy = cast<StructType>(allMaxMinTy->getElementType(m));
    auto *DArrTy = cast<ArrayType>(RecTy->getElementType(1));

    Value *LhsRecPtr = B.CreateStructGEP(allMaxMinTy, LhsBase, m);
    Value *RhsRecPtr = B.CreateStructGEP(allMaxMinTy, RhsBase, m);

    Value *LhsInitPtr = B.CreateStructGEP(RecTy, LhsRecPtr, 3);
    Value *LhsInit = B.CreateLoad(I32Ty, LhsInitPtr);
    Value *RhsRec = B.CreateLoad(RecTy, RhsRecPtr);

    Value *IsUninit = B.CreateICmpEQ(LhsInit, ConstantInt::get(I32Ty, 0));
    BasicBlock *CopyBB = BasicBlock::Create(Ctx, "copy_rhs", Fn);
    BasicBlock *CombineBB = BasicBlock::Create(Ctx, "combine", Fn);
    BasicBlock *NextBB = BasicBlock::Create(Ctx, "next", Fn);
    B.CreateCondBr(IsUninit, CopyBB, CombineBB);

    B.SetInsertPoint(CopyBB);
    B.CreateStore(RhsRec, LhsRecPtr);
    B.CreateBr(NextBB);

    B.SetInsertPoint(CombineBB);
    Value *LhsRec = B.CreateLoad(RecTy, LhsRecPtr);
    Value *LhsVal = B.CreateExtractValue(LhsRec, {0});
    Value *RhsVal = B.CreateExtractValue(RhsRec, {0});
    Value *LhsTid = B.CreateExtractValue(LhsRec, {2});
    Value *RhsTid = B.CreateExtractValue(RhsRec, {2});

    Value *StrictWin = itemIsMax[m] ? B.CreateFCmpOGT(LhsVal, RhsVal)
                                    : B.CreateFCmpOLT(LhsVal, RhsVal);
    Value *ValEq = B.CreateFCmpOEQ(LhsVal, RhsVal);
    Value *TidTie = B.CreateICmpSLE(LhsTid, RhsTid);
    Value *TieWin = B.CreateAnd(ValEq, TidTie);
    Value *LhsWins = B.CreateOr(StrictWin, TieWin);

    Value *NewVal = B.CreateSelect(LhsWins, LhsVal, RhsVal);
    Value *NewTid = B.CreateSelect(LhsWins, LhsTid, RhsTid);

    Value *LhsDArr = B.CreateExtractValue(LhsRec, {1});
    Value *RhsDArr = B.CreateExtractValue(RhsRec, {1});
    Value *NewDArr = UndefValue::get(DArrTy);
    for (unsigned w = 0; w < width; ++w) {
      Value *LD = B.CreateExtractValue(LhsDArr, {w});
      Value *RD = B.CreateExtractValue(RhsDArr, {w});
      Value *SumD = B.CreateFAdd(LD, RD);
      NewDArr = B.CreateInsertValue(NewDArr, SumD, {w});
    }

    Value *Updated = UndefValue::get(RecTy);
    Updated = B.CreateInsertValue(Updated, NewVal, {0});
    Updated = B.CreateInsertValue(Updated, NewDArr, {1});
    Updated = B.CreateInsertValue(Updated, NewTid, {2});
    Updated = B.CreateInsertValue(Updated, ConstantInt::get(I32Ty, 1), {3});
    B.CreateStore(Updated, LhsRecPtr);
    B.CreateBr(NextBB);

    B.SetInsertPoint(NextBB);
  }

  B.CreateRetVoid();
  return Fn;
}

bool AdjointGenerator::handleKnownCallDerivatives(
    CallInst &call, Function *called, StringRef funcName,
    bool subsequent_calls_may_write, const std::vector<bool> &overwritten_args,
    CallInst *const newCall) {
  bool subretused = false;
  bool shadowReturnUsed = false;
  DIFFE_TYPE subretType =
      gutils->getReturnDiffeType(&call, &subretused, &shadowReturnUsed);

  IRBuilder<> BuilderZ(newCall);
  BuilderZ.setFastMathFlags(getFast());

  // Julia's atomic modify pseudo-intrinsic (introduced in Julia 1.13):
  //   {old, new} = julia.atomicmodify.iN.pAS(ptr, op, ordering, syncscope,
  //                                          args...)
  // which atomically performs old = *ptr; new = op(old, args...);
  // *ptr = new, where op's parameter i (i >= 1) is forwarded from call
  // operand i + 3. The pseudo-intrinsic must be kept intact (including in
  // generated derivative code); it is only expanded to atomicrmw/cmpxchg by
  // Julia's ExpandAtomicModify pass after GC lowering.
  if (startsWith(funcName, "julia.atomicmodify.")) {
    // Retire the primal call replayed in the pass being generated. Like the
    // generic call path, tape its result in the augmented primal and read it
    // back from the tape in the derivative pass if the latter needs the
    // primal value: the modification must never be repeated by recomputing
    // the call there.
    auto finishPrimal = [&]() {
      bool primalNeededInReverse = false;
      {
        auto found = gutils->knownRecomputeHeuristic.find(&call);
        if (found != gutils->knownRecomputeHeuristic.end())
          primalNeededInReverse = !found->second;
      }
      if (!primalNeededInReverse &&
          Mode != DerivativeMode::ReverseModeCombined &&
          Mode != DerivativeMode::ForwardMode &&
          Mode != DerivativeMode::ForwardModeError && subretused &&
          !gutils->unnecessaryIntermediates.count(&call)) {
        std::map<UsageKey, bool> Seen =
            gutils->populateSeenFromKnownRecompute();
        auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                              ? DerivativeMode::ReverseModeGradient
                              : Mode;
        primalNeededInReverse =
            DifferentialUseAnalysis::is_value_needed_in_reverse<
                QueryType::Primal>(gutils, &call, minCutMode, Seen,
                                   oldUnreachable);
      }
      if (primalNeededInReverse) {
        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&call, CacheType::Self, BuilderZ));
        eraseIfUnused(call);
      } else if (Mode == DerivativeMode::ReverseModeGradient ||
                 Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        eraseIfUnused(call);
      }
    };

    // The pseudo-intrinsic called with the operands of the primal call,
    // except for the location and (if non-null) the operand at valIdx.
    auto emitCall = [&](Value *ptr, unsigned valIdx, Value *val) -> Value * {
      SmallVector<Value *, 6> args;
      for (size_t i = 0; i < call.arg_size(); i++) {
        if (i == 0)
          args.push_back(ptr);
        else if (i == valIdx && val)
          args.push_back(val);
        else
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
      }
      auto CI = BuilderZ.CreateCall(call.getFunctionType(), called, args);
      CI->setCallingConv(call.getCallingConv());
      CI->setAttributes(call.getAttributes());
      CI->copyMetadata(*newCall);
      return CI;
    };

    // Fully inactive calls only need the primal replayed. This must be
    // handled here rather than falling through to the generic call path,
    // as the latter may decide against the constant fallback (e.g. for a
    // nocapture, non-readonly pointer argument) and then reject the
    // variadic call with `Number of arg operands != function parameters`.
    if (gutils->isConstantInstruction(&call) &&
        gutils->isConstantValue(&call)) {
      finishPrimal();
      return true;
    }

    Type *elty = cast<StructType>(call.getType())->getElementType(0);
    unsigned opArgNo = 0;
    Type *FT = nullptr;
    auto opKind = classifyAtomicModifyOp(
        dyn_cast<Function>(call.getArgOperand(1)), opArgNo, FT);
    if (opKind != AtomicRMWInst::BAD_BINOP &&
        call.arg_size() - 3 != cast<Function>(call.getArgOperand(1))
                                   ->getFunctionType()
                                   ->getNumParams())
      opKind = AtomicRMWInst::BAD_BINOP;
    // Call operand forwarded to op's update value parameter.
    unsigned valIdx = opArgNo + 3;
    Value *valOp = opKind != AtomicRMWInst::BAD_BINOP
                       ? call.getArgOperand(valIdx)
                       : nullptr;

    bool constval = gutils->isConstantValue(&call);
    bool constptr = gutils->isConstantValue(call.getArgOperand(0));

    // No shadow memory is involved; replaying the primal suffices.
    if (constval && constptr) {
      finishPrimal();
      return true;
    }

    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto storeSize = (DL.getTypeSizeInBits(elty) + 7) / 8;
    auto vd = TR.firstPointer(storeSize, call.getArgOperand(0), &call, gutils,
                              /*errifnotfound*/ nullptr,
                              /*pointerIntSame*/ true);

    bool constargs = true;
    for (size_t i = 4; i < call.arg_size(); i++)
      if (!gutils->isConstantValue(call.getArgOperand(i))) {
        constargs = false;
        break;
      }

    // Non-differentiable (integer/pointer) data modified within duplicated
    // memory: replicate the modification on the shadow location with the
    // primal arguments (like inactive stores into active memory), keeping
    // e.g. lock states and counters of shadow objects consistent. As for an
    // inactive store, only the modified data has to be inactive, not the
    // operands op derives it from; when the type of the data is unknown, an
    // active operand however suggests floating point data, which is left
    // to the rules for recognized ops below.
    if (constval &&
        (vd.isKnown() ? !vd.isFloat() : (looseTypeAnalysis && constargs))) {
      // Which pass emits the shadow modification is decided exactly as for
      // an inactive store into duplicated memory (visitCommonStore); in
      // particular the augmented primal and the split derivative pass must
      // not both replay it, and a shadow that is only materialized in the
      // reverse pass must be updated there.
      bool forwardsShadow, backwardsShadow;
      shadowStoreSchedule(call, call.getArgOperand(0), forwardsShadow,
                          backwardsShadow);

      if ((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
          (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeCombined &&
           (forwardsShadow || backwardsShadow)) ||
          Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        Value *dptr = gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
        applyChainRule(
            BuilderZ, [&](Value *dptr) { emitCall(dptr, 0, nullptr); }, dptr);
      }
      finishPrimal();
      return true;
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError ||
        Mode == DerivativeMode::ForwardModeSplit) {
      if (opKind != AtomicRMWInst::BAD_BINOP) {
        // new = op(old, v) is linear in (old, v) for FAdd/FSub/Xchg, so the
        // same op applied to the shadow location and the shadow of v
        // computes the tangent {dold, dnew}.
        //
        // The shadow of v is its tangent (zero if inactive), except for an
        // inactive value exchanged into duplicated memory: like an inactive
        // store, that writes the primal value for pointer/integer data (and
        // zero for floating point data) into the shadow location.
        Value *dval = nullptr;
        if (!gutils->isConstantValue(valOp)) {
          dval = gutils->invertPointerM(valOp, BuilderZ);
        } else if (opKind == AtomicRMWInst::Xchg && !constptr) {
          if (!gutils->runtimeActivity && vd == BaseType::Pointer &&
              !isa<UndefValue>(valOp) && !isa<ConstantPointerNull>(valOp)) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Mismatched activity for: " << call
               << " const val: " << *valOp;
            if (CustomErrorHandler) {
              LLVMBuilderRef BR = startCallbackBuilder(BuilderZ);
              dval = unwrap(CustomErrorHandler(str.c_str(), wrap(&call),
                                               ErrorType::MixedActivityError,
                                               gutils, wrap(valOp), BR));
              finishCallbackBuilder(BR, BuilderZ);
            } else
              EmitWarningAlways("MixedActivityError", call, ss.str(),
                                MixedActivityHint);
          }
          if (!dval)
            dval = gutils->invertPointerM(valOp, BuilderZ,
                                          TypeTree(vd).Only(-1, nullptr));
        }
        if (!dval)
          dval =
              Constant::getNullValue(gutils->getShadowType(valOp->getType()));

        auto rule = [&](Value *dptr, Value *dval) -> Value * {
          if (!dptr) {
            // Inactive location: dold = 0 and dnew = dv (-dv for FSub), in
            // the bits of the modified value.
            Value *dnew = dval;
            if (opKind == AtomicRMWInst::FSub) {
              dnew = BuilderZ.CreateBitCast(dnew, FT);
              dnew = BuilderZ.CreateFNeg(dnew);
            }
            dnew = BuilderZ.CreateBitCast(dnew, elty);
            return BuilderZ.CreateInsertValue(
                Constant::getNullValue(call.getType()), dnew, {1});
          }
          return emitCall(dptr, valIdx, dval);
        };
        Value *diff = applyChainRule(
            call.getType(), BuilderZ, rule,
            constptr ? nullptr
                     : gutils->invertPointerM(call.getArgOperand(0), BuilderZ),
            dval);
        if (!constval)
          setDiffe(&call, diff, BuilderZ);
        finishPrimal();
        return true;
      }
    } else if (Mode == DerivativeMode::ReverseModePrimal) {
      // Nothing to do in the primal pass beyond replaying the call. Besides
      // the inactive case this also covers the augmented primal paired with
      // a ForwardModeSplit derivative, which handles any recognized op.
      if (constval || opKind != AtomicRMWInst::BAD_BINOP) {
        if (!constval)
          resolveShadowPlaceholder(call);
        finishPrimal();
        return true;
      }
    } else if ((Mode == DerivativeMode::ReverseModeCombined ||
                Mode == DerivativeMode::ReverseModeGradient) &&
               constval &&
               (opKind == AtomicRMWInst::FAdd ||
                opKind == AtomicRMWInst::FSub)) {
      if (!gutils->isConstantValue(valOp) && !constptr) {
        auto order = static_cast<AtomicOrdering>(
            cast<ConstantInt>(call.getArgOperand(2))->getZExtValue());
        auto ssid = static_cast<SyncScope::ID>(
            cast<ConstantInt>(call.getArgOperand(3))->getZExtValue());
        auto align = call.getParamAlign(0).value_or(DL.getABITypeAlign(elty));
        addAtomicRMWValueAdjoint(call, call.getArgOperand(0), valOp, elty, FT,
                                 /*negate*/ opKind == AtomicRMWInst::FSub,
                                 align, order, ssid, /*isVolatile*/ false);
      }
      finishPrimal();
      return true;
    }

    // Remaining cases (active result in reverse mode, or an op that is not
    // a recognized linear modification) are not supported.
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << *gutils->oldFunc << "\n" << call << "\n";
    ss << " Active atomic modify not yet handled";
    Value *rval = EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
    if (!constval) {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError ||
          Mode == DerivativeMode::ForwardModeSplit) {
        if (!rval)
          rval = Constant::getNullValue(gutils->getShadowType(call.getType()));
        setDiffe(&call, rval, BuilderZ);
      } else {
        resolveShadowPlaceholder(call, rval);
      }
    }
    if (!call.getType()->isVoidTy()) {
      for (auto &U :
           make_early_inc_range(gutils->getNewFromOriginal(&call)->uses())) {
        U.set(UndefValue::get(call.getType()));
      }
    }
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }

  if (Mode != DerivativeMode::ReverseModePrimal && called) {
    if (funcName == "__kmpc_for_static_init_4" ||
        funcName == "__kmpc_for_static_init_4u" ||
        funcName == "__kmpc_for_static_init_8" ||
        funcName == "__kmpc_for_static_init_8u") {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      auto fini = called->getParent()->getFunction("__kmpc_for_static_fini");
      assert(fini);
      Value *args[] = {
          lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
          lookup(gutils->getNewFromOriginal(call.getArgOperand(1)), Builder2)};
      auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
      fcall->setCallingConv(fini->getCallingConv());
      return true;
    }
  }

  // Canonicalize MPI routine names across calling conventions: the C
  // convention ("MPI_Recv", "PMPI_Recv") as well as Fortran ABI manglings
  // ("mpi_recv_", "mpi_comm_rank__", ...) all map to the canonical C name
  // (without profiling prefix) used throughout handleMPI.
  llvm::StringRef canonMPIName = canonicalizeMPIName(funcName);
  if (!canonMPIName.empty() &&
      (!gutils->isConstantInstruction(&call) || canonMPIName == "MPI_Barrier" ||
       canonMPIName == "MPI_Comm_free" ||
       canonMPIName == "MPI_Comm_disconnect" || canonMPIName == "MPI_Init" ||
       canonMPIName == "MPI_Init_thread" || canonMPIName == "MPI_Finalize" ||
       canonMPIName == "MPI_Test" || canonMPIName == "MPI_Probe" ||
       MPIInactiveCommAllocators.find(canonMPIName) !=
           MPIInactiveCommAllocators.end())) {
    handleMPI(call, called, canonMPIName);
    return true;
  }

  if (auto blas = extractBLAS(funcName)) {
    if (handleBLAS(call, called, *blas, overwritten_args))
      return true;
  }

  if (funcName == "printf" || funcName == "puts" ||
      startsWith(funcName, "_ZN3std2io5stdio6_print") ||
      startsWith(funcName, "_ZN4core3fmt")) {
    if (Mode == DerivativeMode::ReverseModeGradient) {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    }
    return true;
  }
  if (called && (called->getName().contains("__enzyme_float") ||
                 called->getName().contains("__enzyme_double") ||
                 called->getName().contains("__enzyme_integer") ||
                 called->getName().contains("__enzyme_pointer"))) {
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }

  // Handle lgamma, safe to recompute so no store/change to forward
  if (called) {
    if (funcName == "__kmpc_for_static_init_4" ||
        funcName == "__kmpc_for_static_init_4u" ||
        funcName == "__kmpc_for_static_init_8" ||
        funcName == "__kmpc_for_static_init_8u") {
      if (Mode != DerivativeMode::ReverseModePrimal) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto fini = called->getParent()->getFunction("__kmpc_for_static_fini");
        assert(fini);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
        fcall->setCallingConv(fini->getCallingConv());
      }
      return true;
    }
    if (funcName == "__kmpc_for_static_fini") {
      if (Mode != DerivativeMode::ReverseModePrimal) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      }
      return true;
    }
    // TODO check
    // Adjoint of barrier is to place a barrier at the corresponding
    // location in the reverse.
    if (funcName == "__kmpc_barrier") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto callval = call.getCalledOperand();
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2)};
        Builder2.CreateCall(call.getFunctionType(), callval, args);
      }
      return true;
    }
    if (funcName == "__kmpc_critical") {
      if (Mode != DerivativeMode::ReverseModePrimal) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto crit2 = called->getParent()->getFunction("__kmpc_end_critical");
        assert(crit2);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
        fcall->setCallingConv(crit2->getCallingConv());
      }
      return true;
    }
    if (funcName == "__kmpc_end_critical") {
      if (Mode != DerivativeMode::ReverseModePrimal) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto crit2 = called->getParent()->getFunction("__kmpc_critical");
        assert(crit2);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
        fcall->setCallingConv(crit2->getCallingConv());
      }
      return true;
    }

    if (funcName == "__kmpc_end_reduce" ||
        funcName == "__kmpc_end_reduce_nowait") {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError ||
          Mode == DerivativeMode::ReverseModePrimal) {
        return true;
      }
      if (Mode == DerivativeMode::ForwardModeSplit) {
        EmitNoDerivativeError(
            "OpenMP reduction not yet supported in split forward mode", call,
            gutils, BuilderZ);
        return true;
      }
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      }

      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      Module &M = *called->getParent();
      LLVMContext &Ctx = call.getContext();
      Type *VoidTy = Type::getVoidTy(Ctx);

      Value *origLoc = gutils->getNewFromOriginal(call.getArgOperand(0));
      Value *loc = lookup(origLoc, Builder2);
      Value *origGtid = gutils->getNewFromOriginal(call.getArgOperand(1));
      Value *gtid = lookup(origGtid, Builder2);

      if (funcName == "__kmpc_end_reduce") {
        FunctionCallee barCallee = M.getOrInsertFunction(
            "__kmpc_barrier",
            FunctionType::get(VoidTy, {loc->getType(), gtid->getType()},
                              false));
        if (auto *barFn =
                dyn_cast<Function>(barCallee.getCallee()->stripPointerCasts()))
          attributeKnownFunctions(*barFn);
        FunctionType *barFT = barCallee.getFunctionType();
        Value *barLoc = loc;
        if (barLoc->getType() != barFT->getParamType(0))
          barLoc = Builder2.CreatePointerCast(barLoc, barFT->getParamType(0));
        Value *barArgs[] = {barLoc, gtid};
        Builder2.CreateCall(barCallee, barArgs);
      }

      Type *lckArgTy = call.getArgOperand(2)->getType();
      GlobalVariable *critGV =
          getOrCreateOMPReduceLock(M, "__enzyme_omp_reduce_crit_lck");
      FunctionCallee critCallee = M.getOrInsertFunction(
          "__kmpc_critical",
          FunctionType::get(VoidTy, {loc->getType(), gtid->getType(), lckArgTy},
                            false));
      if (auto *critFn =
              dyn_cast<Function>(critCallee.getCallee()->stripPointerCasts()))
        attributeKnownFunctions(*critFn);
      FunctionType *critFT = critCallee.getFunctionType();
      Value *critLoc = loc;
      if (critLoc->getType() != critFT->getParamType(0))
        critLoc = Builder2.CreatePointerCast(critLoc, critFT->getParamType(0));
      Value *critLock = critGV;
      if (critLock->getType() != critFT->getParamType(2))
        critLock =
            Builder2.CreatePointerCast(critLock, critFT->getParamType(2));
      Value *critArgs[] = {critLoc, gtid, critLock};
      Builder2.CreateCall(critCallee, critArgs);
      return true;
    }

    if (funcName == "__kmpc_reduce" || funcName == "__kmpc_reduce_nowait") {
      Module &M = *called->getParent();
      LLVMContext &Ctx = call.getContext();
      const DataLayout &DL = M.getDataLayout();
      unsigned width = gutils->getWidth();

      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        if (!gutils->isConstantValue(&call)) {
          Value *zeroDiff =
              Constant::getNullValue(gutils->getShadowType(call.getType()));
          setDiffe(&call, zeroDiff, BuilderZ);
        }
        if (gutils->isConstantInstruction(&call) &&
            gutils->isConstantValue(call.getArgOperand(4))) {
          return true;
        }
        Function *reduceFn =
            dyn_cast<Function>(call.getArgOperand(5)->stripPointerCasts());
        if (!reduceFn || reduceFn->empty() || reduceFn->arg_size() != 2) {
          EmitNoDerivativeError(
              "Cannot differentiate OpenMP reduction with unknown combiner",
              call, gutils, BuilderZ);
          return true;
        }

        FnTypeInfo nextTypeInfo(reduceFn);
        TypeTree argTree = TR.query(call.getArgOperand(4));
        std::set<int64_t> argKnown =
            TR.knownIntegralValues(call.getArgOperand(4));
        for (Argument &arg : reduceFn->args()) {
          nextTypeInfo.Arguments.insert(
              std::pair<Argument *, TypeTree>(&arg, argTree));
          nextTypeInfo.KnownValues.insert(
              std::pair<Argument *, std::set<int64_t>>(&arg, argKnown));
        }
        nextTypeInfo.Return = TypeTree();

        std::vector<DIFFE_TYPE> subArgs = {DIFFE_TYPE::DUP_ARG,
                                           DIFFE_TYPE::DUP_ARG};
        std::vector<bool> subOverwritten = {true, false};
        Function *diffRedFn = gutils->Logic.CreateForwardDiff(
            RequestContext(&call, &BuilderZ), reduceFn, DIFFE_TYPE::CONSTANT,
            subArgs, TR.analyzer->interprocedural, /*returnValue*/ false, Mode,
            /*freeMemory*/ true, gutils->runtimeActivity, gutils->strongZero,
            width, /*additionalArg*/ nullptr, nextTypeInfo,
            /*subsequent_calls_may_write*/ true, subOverwritten,
            /*augmented*/ nullptr, /*omp*/ false);

        Function *wrapFn = getOrCreateOMPFwdReduceWrapper(
            M, reduceFn->getFunctionType(), diffRedFn, width);

        Type *I8PtrTy = getInt8PtrTy(Ctx);
        ArrayType *bundleTy = ArrayType::get(I8PtrTy, 1 + width);
        IRBuilder<> allocB(gutils->inversionAllocs);
        AllocaInst *bundleAlloca =
            allocB.CreateAlloca(bundleTy, nullptr, "omp_fwd_reduce_bundle");

        Value *primalData = gutils->getNewFromOriginal(call.getArgOperand(4));
        if (primalData->getType() != I8PtrTy)
          primalData = BuilderZ.CreatePointerCast(primalData, I8PtrTy);
        Value *slot0Idxs[] = {BuilderZ.getInt64(0), BuilderZ.getInt64(0)};
        Value *slot0Ptr =
            BuilderZ.CreateInBoundsGEP(bundleTy, bundleAlloca, slot0Idxs);
        BuilderZ.CreateStore(primalData, slot0Ptr);

        Value *shadowData =
            gutils->invertPointerM(call.getArgOperand(4), BuilderZ);
        for (unsigned w = 0; w < width; ++w) {
          Value *laneShadow =
              (width == 1)
                  ? shadowData
                  : GradientUtils::extractMeta(BuilderZ, shadowData, w);
          if (laneShadow->getType() != I8PtrTy)
            laneShadow = BuilderZ.CreatePointerCast(laneShadow, I8PtrTy);
          Value *slotWIdxs[] = {BuilderZ.getInt64(0), BuilderZ.getInt64(1 + w)};
          Value *slotWPtr =
              BuilderZ.CreateInBoundsGEP(bundleTy, bundleAlloca, slotWIdxs);
          BuilderZ.CreateStore(laneShadow, slotWPtr);
        }

        newCall->setArgOperand(
            2, ConstantInt::get(call.getArgOperand(2)->getType(), 1 + width));
        uint64_t bundleBytes = DL.getTypeAllocSize(bundleTy);
        newCall->setArgOperand(
            3, ConstantInt::get(call.getArgOperand(3)->getType(), bundleBytes));
        Value *bundleArg = bundleAlloca;
        Type *dataParamTy = newCall->getFunctionType()->getParamType(4);
        if (bundleArg->getType() != dataParamTy)
          bundleArg = BuilderZ.CreatePointerCast(bundleArg, dataParamTy);
        newCall->setArgOperand(4, bundleArg);
        Value *wrapArg = wrapFn;
        Type *fnParamTy = newCall->getFunctionType()->getParamType(5);
        if (wrapArg->getType() != fnParamTy)
          wrapArg = BuilderZ.CreatePointerCast(wrapArg, fnParamTy);
        newCall->setArgOperand(5, wrapArg);
        return true;
      }

      if (Mode == DerivativeMode::ForwardModeSplit) {
        EmitNoDerivativeError(
            "OpenMP reduction not yet supported in split forward mode", call,
            gutils, BuilderZ);
        return true;
      }

      if (!gutils->isConstantValue(&call)) {
        resolveShadowPlaceholder(call);
      }

      Function *reduceFn =
          dyn_cast<Function>(call.getArgOperand(5)->stripPointerCasts());
      SmallVector<OMPReductionItem, 4> allItems;
      bool extracted = extractOMPReductionItems(call, reduceFn, gutils->OrigDT,
                                                DL, allItems);
      bool callActive = !gutils->isConstantInstruction(&call) ||
                        !gutils->isConstantValue(call.getArgOperand(4));
      if (!extracted && callActive) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << "Cannot differentiate unsupported OpenMP reduction combiner in "
              "reverse mode: "
           << call << "\n";
        EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
      }

      SmallVector<OMPReductionItem, 4> sumItems;
      SmallVector<OMPReductionItem, 4> maxMinItems;
      if (extracted) {
        for (const OMPReductionItem &it : allItems) {
          if (it.op == OMPReductionOp::IntInactive ||
              gutils->isConstantValue(it.origPrivPtr))
            continue;
          if (it.op == OMPReductionOp::Sum)
            sumItems.push_back(it);
          else
            maxMinItems.push_back(it);
        }
      }

      Type *I32Ty = Type::getInt32Ty(Ctx);
      Type *I64Ty = Type::getInt64Ty(Ctx);
      Type *VoidTy = Type::getVoidTy(Ctx);

      StructType *preValStructTy = nullptr;
      StructType *allMaxMinTy = nullptr;
      SmallVector<bool, 4> maxMinIsMax;
      if (!maxMinItems.empty()) {
        SmallVector<Type *, 4> preFields;
        SmallVector<Type *, 4> recFields;
        for (const OMPReductionItem &it : maxMinItems) {
          preFields.push_back(it.elemTy);
          ArrayType *dArrTy = ArrayType::get(it.elemTy, width);
          recFields.push_back(
              StructType::get(Ctx, {it.elemTy, dArrTy, I32Ty, I32Ty}));
          maxMinIsMax.push_back(it.op == OMPReductionOp::Max);
        }
        preValStructTy = StructType::get(Ctx, preFields);
        allMaxMinTy = StructType::get(Ctx, recFields);
      }

      StructType *scratchTy = nullptr;
      if (!sumItems.empty() || !maxMinItems.empty()) {
        SmallVector<Type *, 4> scratchFields;
        for (const OMPReductionItem &it : sumItems)
          scratchFields.push_back(ArrayType::get(it.elemTy, width));
        if (allMaxMinTy)
          scratchFields.push_back(allMaxMinTy);
        scratchTy = StructType::get(Ctx, scratchFields);
      }

      Value *preValAgg = nullptr;
      if (preValStructTy) {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          Value *agg = UndefValue::get(preValStructTy);
          for (unsigned m = 0; m < maxMinItems.size(); ++m) {
            Value *privPtr =
                gutils->getNewFromOriginal(maxMinItems[m].origPrivPtr);
            unsigned AS = privPtr->getType()->getPointerAddressSpace();
            Type *elPtrTy = getPointerType(maxMinItems[m].elemTy, AS);
            if (privPtr->getType() != elPtrTy)
              privPtr = BuilderZ.CreatePointerCast(privPtr, elPtrTy);
            Value *preV = BuilderZ.CreateLoad(maxMinItems[m].elemTy, privPtr,
                                              "omp_reduce_pre");
            agg = BuilderZ.CreateInsertValue(agg, preV, {m});
          }
          unsigned shIdx = getIndex(&call, CacheType::Shadow, BuilderZ);
          preValAgg = gutils->cacheForReverse(BuilderZ, agg, shIdx);
        } else {
#if LLVM_VERSION_MAJOR >= 18
          auto It = BuilderZ.GetInsertPoint();
          It.setHeadBit(true);
          BuilderZ.SetInsertPoint(It);
#endif
          PHINode *ph = BuilderZ.CreatePHI(preValStructTy, 0, "omp_reduce_pre");
          unsigned shIdx = getIndex(&call, CacheType::Shadow, BuilderZ);
          preValAgg = gutils->cacheForReverse(BuilderZ, ph, shIdx);
        }
      }

      Value *sharedScratch = nullptr;
      if (scratchTy) {
        BasicBlock *entryBB =
            gutils->getNewFromOriginal(&gutils->oldFunc->getEntryBlock());
        Value *alloc = nullptr;
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          IRBuilder<> allocEntryB(getFirstNonPHI(entryBB));
          alloc = CreateAllocation(allocEntryB, scratchTy,
                                   ConstantInt::get(I64Ty, 1),
                                   "omp_reduce_scratch");
        } else {
          IRBuilder<> phiB(&*entryBB->begin());
          alloc = phiB.CreatePHI(getUnqual(scratchTy), 0, "omp_reduce_scratch");
        }
        IRBuilder<> entryB(getFirstNonPHI(entryBB));
        unsigned tapeIdx = getIndex(&call, CacheType::Tape, BuilderZ);
        bool savedOMP = gutils->omp;
        gutils->omp = false;
        sharedScratch = gutils->cacheForReverse(entryB, alloc, tapeIdx);
        gutils->omp = savedOMP;
      }

      unsigned selfIdx = getIndex(&call, CacheType::Self, BuilderZ);
      Value *cachedReduce = gutils->cacheForReverse(BuilderZ, newCall, selfIdx);

      if (Mode == DerivativeMode::ReverseModePrimal) {
        return true;
      }

      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      Value *origLoc = gutils->getNewFromOriginal(call.getArgOperand(0));
      Value *loc = lookup(origLoc, Builder2);
      Value *origGtid = gutils->getNewFromOriginal(call.getArgOperand(1));
      Value *gtid = lookup(origGtid, Builder2);
      Value *reduceVal = lookup(cachedReduce, Builder2);
      Value *preValRev = preValAgg ? lookup(preValAgg, Builder2) : nullptr;
      Value *scratchPtr =
          sharedScratch ? lookup(sharedScratch, Builder2) : nullptr;
      if (scratchPtr && scratchPtr->getType() != getUnqual(scratchTy)) {
        scratchPtr =
            Builder2.CreatePointerCast(scratchPtr, getUnqual(scratchTy));
      }

      auto lookupShadowLanePtrs = [&](const OMPReductionItem &it) {
        SmallVector<Value *, 2> ptrs;
        Value *inv = gutils->invertPointerM(it.origPrivPtr, Builder2);
        inv = lookup(inv, Builder2);
        for (unsigned w = 0; w < width; ++w) {
          Value *lp =
              (width == 1) ? inv : GradientUtils::extractMeta(Builder2, inv, w);
          unsigned AS = lp->getType()->getPointerAddressSpace();
          Type *elPtrTy = getPointerType(it.elemTy, AS);
          if (lp->getType() != elPtrTy)
            lp = Builder2.CreatePointerCast(lp, elPtrTy);
          ptrs.push_back(lp);
        }
        return ptrs;
      };

      SmallVector<SmallVector<Value *, 2>, 4> sumShadowPtrs;
      for (const OMPReductionItem &it : sumItems)
        sumShadowPtrs.push_back(lookupShadowLanePtrs(it));
      SmallVector<SmallVector<Value *, 2>, 4> maxMinShadowPtrs;
      for (const OMPReductionItem &it : maxMinItems)
        maxMinShadowPtrs.push_back(lookupShadowLanePtrs(it));

      auto emitBarrier = [&](IRBuilder<> &B) {
        FunctionCallee barCallee = M.getOrInsertFunction(
            "__kmpc_barrier",
            FunctionType::get(VoidTy, {loc->getType(), gtid->getType()},
                              false));
        if (auto *barFn =
                dyn_cast<Function>(barCallee.getCallee()->stripPointerCasts()))
          attributeKnownFunctions(*barFn);
        FunctionType *barFT = barCallee.getFunctionType();
        Value *barLoc = loc;
        if (barLoc->getType() != barFT->getParamType(0))
          barLoc = B.CreatePointerCast(barLoc, barFT->getParamType(0));
        Value *barArgs[] = {barLoc, gtid};
        B.CreateCall(barCallee, barArgs);
      };

      // 1. Root thread (%red == 1) publishes reduced adjoints / resets init
      //    flags in shared scratch and closes the reverse critical section.
      Value *oneVal = ConstantInt::get(reduceVal->getType(), 1);
      Value *isCase1 = Builder2.CreateICmpEQ(reduceVal, oneVal);
      BasicBlock *curBB1 = Builder2.GetInsertBlock();
      BasicBlock *case1RevBB = gutils->addReverseBlock(
          curBB1, curBB1->getName() + "_omp_red_case1", /*forkCache=*/true,
          /*push=*/false);
      BasicBlock *afterCase1BB = gutils->addReverseBlock(
          curBB1, curBB1->getName() + "_omp_red_after_case1",
          /*forkCache=*/true, /*push=*/true);
      Builder2.CreateCondBr(isCase1, case1RevBB, afterCase1BB);

      Builder2.SetInsertPoint(case1RevBB);
      for (unsigned s = 0; s < sumItems.size(); ++s) {
        Value *sumFieldPtr = Builder2.CreateStructGEP(scratchTy, scratchPtr, s);
        ArrayType *sArrTy = ArrayType::get(sumItems[s].elemTy, width);
        for (unsigned w = 0; w < width; ++w) {
          Value *dVal =
              Builder2.CreateLoad(sumItems[s].elemTy, sumShadowPtrs[s][w]);
          Value *idxs[] = {Builder2.getInt64(0), Builder2.getInt64(w)};
          Value *dst = Builder2.CreateInBoundsGEP(sArrTy, sumFieldPtr, idxs);
          Builder2.CreateStore(dVal, dst);
        }
      }
      if (!maxMinItems.empty()) {
        Value *mmBasePtr =
            Builder2.CreateStructGEP(scratchTy, scratchPtr, sumItems.size());
        for (unsigned m = 0; m < maxMinItems.size(); ++m) {
          auto *recTy = cast<StructType>(allMaxMinTy->getElementType(m));
          Value *recPtr = Builder2.CreateStructGEP(allMaxMinTy, mmBasePtr, m);
          Value *initPtr = Builder2.CreateStructGEP(recTy, recPtr, 3);
          Builder2.CreateStore(ConstantInt::get(I32Ty, 0), initPtr);
        }
      }
      {
        Type *lckArgTy = call.getArgOperand(6)->getType();
        GlobalVariable *critGV =
            getOrCreateOMPReduceLock(M, "__enzyme_omp_reduce_crit_lck");
        FunctionCallee endCritCallee = M.getOrInsertFunction(
            "__kmpc_end_critical",
            FunctionType::get(
                VoidTy, {loc->getType(), gtid->getType(), lckArgTy}, false));
        if (auto *endCritFn = dyn_cast<Function>(
                endCritCallee.getCallee()->stripPointerCasts()))
          attributeKnownFunctions(*endCritFn);
        FunctionType *endCritFT = endCritCallee.getFunctionType();
        Value *ecLoc = loc;
        if (ecLoc->getType() != endCritFT->getParamType(0))
          ecLoc = Builder2.CreatePointerCast(ecLoc, endCritFT->getParamType(0));
        Value *ecLock = critGV;
        if (ecLock->getType() != endCritFT->getParamType(2))
          ecLock =
              Builder2.CreatePointerCast(ecLock, endCritFT->getParamType(2));
        Value *ecArgs[] = {ecLoc, gtid, ecLock};
        Builder2.CreateCall(endCritCallee, ecArgs);
      }
      Builder2.CreateBr(afterCase1BB);

      Builder2.SetInsertPoint(afterCase1BB);

      if (scratchTy) {
        emitBarrier(Builder2);

        // 2. Sum reductions: broadcast d_sum to non-root threads (%red == 0).
        if (!sumItems.empty()) {
          Value *zeroVal = ConstantInt::get(reduceVal->getType(), 0);
          Value *isNonRoot = Builder2.CreateICmpEQ(reduceVal, zeroVal);
          BasicBlock *curBBSum = Builder2.GetInsertBlock();
          BasicBlock *nonRootBB = gutils->addReverseBlock(
              curBBSum, curBBSum->getName() + "_omp_red_sum",
              /*forkCache=*/true, /*push=*/false);
          BasicBlock *afterSumBB = gutils->addReverseBlock(
              curBBSum, curBBSum->getName() + "_omp_red_after_sum",
              /*forkCache=*/true, /*push=*/true);
          Builder2.CreateCondBr(isNonRoot, nonRootBB, afterSumBB);

          Builder2.SetInsertPoint(nonRootBB);
          for (unsigned s = 0; s < sumItems.size(); ++s) {
            Value *sumFieldPtr =
                Builder2.CreateStructGEP(scratchTy, scratchPtr, s);
            ArrayType *sArrTy = ArrayType::get(sumItems[s].elemTy, width);
            for (unsigned w = 0; w < width; ++w) {
              Value *idxs[] = {Builder2.getInt64(0), Builder2.getInt64(w)};
              Value *src =
                  Builder2.CreateInBoundsGEP(sArrTy, sumFieldPtr, idxs);
              Value *dShared = Builder2.CreateLoad(sumItems[s].elemTy, src);
              Value *oldD =
                  Builder2.CreateLoad(sumItems[s].elemTy, sumShadowPtrs[s][w]);
              Value *newD = Builder2.CreateFAdd(oldD, dShared);
              Builder2.CreateStore(newD, sumShadowPtrs[s][w]);
            }
          }
          Builder2.CreateBr(afterSumBB);
          Builder2.SetInsertPoint(afterSumBB);
        }

        // 3. Max/Min reductions: O(log P) reverse tree reduction over
        //    (pre_val, d_sum, gtid) to route the adjoint to the winning thread.
        if (!maxMinItems.empty()) {
          IRBuilder<> revAllocB(gutils->inversionAllocs);
          AllocaInst *localMM =
              revAllocB.CreateAlloca(allMaxMinTy, nullptr, "omp_rev_mm_local");
          Value *gtidI32 = Builder2.CreateSExtOrTrunc(gtid, I32Ty);

          for (unsigned m = 0; m < maxMinItems.size(); ++m) {
            auto *recTy = cast<StructType>(allMaxMinTy->getElementType(m));
            auto *dArrTy = cast<ArrayType>(recTy->getElementType(1));
            Value *preM = Builder2.CreateExtractValue(preValRev, {m});
            Value *myDArr = UndefValue::get(dArrTy);
            for (unsigned w = 0; w < width; ++w) {
              Value *myD = Builder2.CreateLoad(maxMinItems[m].elemTy,
                                               maxMinShadowPtrs[m][w]);
              Value *zeroElem = Constant::getNullValue(maxMinItems[m].elemTy);
              Builder2.CreateStore(zeroElem, maxMinShadowPtrs[m][w]);
              myDArr = Builder2.CreateInsertValue(myDArr, myD, {w});
            }
            Value *recVal = UndefValue::get(recTy);
            recVal = Builder2.CreateInsertValue(recVal, preM, {0});
            recVal = Builder2.CreateInsertValue(recVal, myDArr, {1});
            recVal = Builder2.CreateInsertValue(recVal, gtidI32, {2});
            recVal = Builder2.CreateInsertValue(
                recVal, ConstantInt::get(I32Ty, 1), {3});
            Value *localRecPtr =
                Builder2.CreateStructGEP(allMaxMinTy, localMM, m);
            Builder2.CreateStore(recVal, localRecPtr);
          }

          Function *revRedFn = getOrCreateOMPRevMaxMinReduceFn(
              M, reduceFn->getFunctionType(), allMaxMinTy, maxMinIsMax, width);
          GlobalVariable *revLockGV =
              getOrCreateOMPReduceLock(M, "__enzyme_omp_reduce_rev_lck");
          FunctionCallee redNowaitCallee = M.getOrInsertFunction(
              "__kmpc_reduce_nowait", called->getFunctionType());
          if (auto *rnFn = dyn_cast<Function>(
                  redNowaitCallee.getCallee()->stripPointerCasts()))
            attributeKnownFunctions(*rnFn);
          FunctionType *rnFT = redNowaitCallee.getFunctionType();
          Value *rnLoc = loc;
          if (rnLoc->getType() != rnFT->getParamType(0))
            rnLoc = Builder2.CreatePointerCast(rnLoc, rnFT->getParamType(0));
          Value *rnNumVars =
              ConstantInt::get(rnFT->getParamType(2), maxMinItems.size());
          uint64_t mmBytes = DL.getTypeAllocSize(allMaxMinTy);
          Value *rnSize = ConstantInt::get(rnFT->getParamType(3), mmBytes);
          Value *rnData = localMM;
          if (rnData->getType() != rnFT->getParamType(4))
            rnData = Builder2.CreatePointerCast(rnData, rnFT->getParamType(4));
          Value *rnFunc = revRedFn;
          if (rnFunc->getType() != rnFT->getParamType(5))
            rnFunc = Builder2.CreatePointerCast(rnFunc, rnFT->getParamType(5));
          Value *rnLock = revLockGV;
          if (rnLock->getType() != rnFT->getParamType(6))
            rnLock = Builder2.CreatePointerCast(rnLock, rnFT->getParamType(6));
          Value *rnArgs[] = {rnLoc,  gtid,   rnNumVars, rnSize,
                             rnData, rnFunc, rnLock};
          Value *revRedRes = Builder2.CreateCall(redNowaitCallee, rnArgs);

          Value *revOne = ConstantInt::get(revRedRes->getType(), 1);
          Value *isRevRoot = Builder2.CreateICmpEQ(revRedRes, revOne);
          BasicBlock *curBBMM = Builder2.GetInsertBlock();
          BasicBlock *revRootBB = gutils->addReverseBlock(
              curBBMM, curBBMM->getName() + "_omp_rev_mm_root",
              /*forkCache=*/true, /*push=*/false);
          BasicBlock *afterRevRedBB = gutils->addReverseBlock(
              curBBMM, curBBMM->getName() + "_omp_rev_mm_done",
              /*forkCache=*/true, /*push=*/true);
          Builder2.CreateCondBr(isRevRoot, revRootBB, afterRevRedBB);

          Builder2.SetInsertPoint(revRootBB);
          Value *mmSharedPtr =
              Builder2.CreateStructGEP(scratchTy, scratchPtr, sumItems.size());
          FunctionType *combFT = revRedFn->getFunctionType();
          Value *combLhs = mmSharedPtr;
          if (combLhs->getType() != combFT->getParamType(0))
            combLhs =
                Builder2.CreatePointerCast(combLhs, combFT->getParamType(0));
          Value *combRhs = localMM;
          if (combRhs->getType() != combFT->getParamType(1))
            combRhs =
                Builder2.CreatePointerCast(combRhs, combFT->getParamType(1));
          Value *combArgs[] = {combLhs, combRhs};
          Builder2.CreateCall(combFT, revRedFn, combArgs);

          Type *lckParamTy = rnFT->getParamType(6);
          FunctionCallee endRNCallee = M.getOrInsertFunction(
              "__kmpc_end_reduce_nowait",
              FunctionType::get(
                  VoidTy,
                  {rnFT->getParamType(0), rnFT->getParamType(1), lckParamTy},
                  false));
          if (auto *ernFn = dyn_cast<Function>(
                  endRNCallee.getCallee()->stripPointerCasts()))
            attributeKnownFunctions(*ernFn);
          FunctionType *endRNFT = endRNCallee.getFunctionType();
          Value *ernLoc = loc;
          if (ernLoc->getType() != endRNFT->getParamType(0))
            ernLoc =
                Builder2.CreatePointerCast(ernLoc, endRNFT->getParamType(0));
          Value *ernLock = revLockGV;
          if (ernLock->getType() != endRNFT->getParamType(2))
            ernLock =
                Builder2.CreatePointerCast(ernLock, endRNFT->getParamType(2));
          Value *ernArgs[] = {ernLoc, gtid, ernLock};
          Builder2.CreateCall(endRNCallee, ernArgs);
          Builder2.CreateBr(afterRevRedBB);

          Builder2.SetInsertPoint(afterRevRedBB);
          emitBarrier(Builder2);

          Value *mmSharedBase =
              Builder2.CreateStructGEP(scratchTy, scratchPtr, sumItems.size());
          for (unsigned m = 0; m < maxMinItems.size(); ++m) {
            auto *recTy = cast<StructType>(allMaxMinTy->getElementType(m));
            auto *dArrTy = cast<ArrayType>(recTy->getElementType(1));
            Value *recPtr =
                Builder2.CreateStructGEP(allMaxMinTy, mmSharedBase, m);
            Value *winTidPtr = Builder2.CreateStructGEP(recTy, recPtr, 2);
            Value *winTid = Builder2.CreateLoad(I32Ty, winTidPtr);
            Value *isWinner = Builder2.CreateICmpEQ(gtidI32, winTid);
            BasicBlock *curWinBB = Builder2.GetInsertBlock();
            BasicBlock *winBB = gutils->addReverseBlock(
                curWinBB, curWinBB->getName() + "_omp_mm_win",
                /*forkCache=*/true, /*push=*/false);
            BasicBlock *afterWinBB = gutils->addReverseBlock(
                curWinBB, curWinBB->getName() + "_omp_mm_after_win",
                /*forkCache=*/true, /*push=*/true);
            Builder2.CreateCondBr(isWinner, winBB, afterWinBB);

            Builder2.SetInsertPoint(winBB);
            Value *dSumPtr = Builder2.CreateStructGEP(recTy, recPtr, 1);
            Value *dSumArr = Builder2.CreateLoad(dArrTy, dSumPtr);
            for (unsigned w = 0; w < width; ++w) {
              Value *dW = Builder2.CreateExtractValue(dSumArr, {w});
              Value *curD = Builder2.CreateLoad(maxMinItems[m].elemTy,
                                                maxMinShadowPtrs[m][w]);
              Value *updD = Builder2.CreateFAdd(curD, dW);
              Builder2.CreateStore(updD, maxMinShadowPtrs[m][w]);
            }
            Builder2.CreateBr(afterWinBB);
            Builder2.SetInsertPoint(afterWinBB);
          }
        }

        emitBarrier(Builder2);
        if (!gutils->omp && shouldFree()) {
          CreateDealloc(Builder2, scratchPtr);
        }
      } else if (funcName == "__kmpc_reduce") {
        emitBarrier(Builder2);
      }

      return true;
    }

    if (startsWith(funcName, "__kmpc") &&
        funcName != "__kmpc_global_thread_num") {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << " unhandled openmp function: " << call << "\n";
      EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
      return true;
    }

    auto mod = call.getParent()->getParent()->getParent();
#include "CallDerivatives.inc"

    if (funcName == "llvm.julia.gc_preserve_end") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {

        auto begin_call = cast<CallInst>(call.getOperand(0));

        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        if (!hasSingleGCPreserveEnd(begin_call, gutils)) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << "cannot reverse gc preserve region with several ends: "
             << *begin_call << "\n";
          EmitNoDerivativeError(ss.str(), call, gutils, Builder2);
          return true;
        }
        SmallVector<Value *, 1> args;
        for (auto &arg : begin_call->args()) {
          bool primalUsed = false;
          bool shadowUsed = false;
          gutils->getReturnDiffeType(arg, &primalUsed, &shadowUsed);

          if (primalUsed)
            args.push_back(
                gutils->lookupM(gutils->getNewFromOriginal(arg), Builder2));

          if (!gutils->isConstantValue(arg) && shadowUsed) {
            Value *ptrshadow = gutils->lookupM(
                gutils->invertPointerM(arg, BuilderZ), Builder2);
            if (gutils->getWidth() == 1)
              args.push_back(ptrshadow);
            else
              for (size_t i = 0; i < gutils->getWidth(); ++i)
                args.push_back(gutils->extractMeta(Builder2, ptrshadow, i));
          }
        }

        auto newp = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                "llvm.julia.gc_preserve_begin",
                FunctionType::get(Type::getTokenTy(call.getContext()),
                                  ArrayRef<Type *>(), true)),
            args);
        auto ifound = gutils->invertedPointers.find(begin_call);
        assert(ifound != gutils->invertedPointers.end());
        auto placeholder = cast<CallInst>(&*ifound->second);
        gutils->invertedPointers.erase(ifound);
        gutils->invertedPointers.insert(std::make_pair(
            (const Value *)begin_call, InvertedPointerVH(gutils, newp)));

        gutils->replaceAWithB(placeholder, newp);
        gutils->erase(placeholder);
      }
      return true;
    }
    if (funcName == "llvm.julia.gc_preserve_begin") {
      SmallVector<Value *, 1> args;
      for (auto &arg : call.args()) {
        bool primalUsed = false;
        bool shadowUsed = false;
        gutils->getReturnDiffeType(arg, &primalUsed, &shadowUsed);

        if (primalUsed)
          args.push_back(gutils->getNewFromOriginal(arg));

        if (!gutils->isConstantValue(arg) && shadowUsed) {
          Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
          if (gutils->getWidth() == 1)
            args.push_back(ptrshadow);
          else
            for (size_t i = 0; i < gutils->getWidth(); ++i)
              args.push_back(gutils->extractMeta(BuilderZ, ptrshadow, i));
        }
      }

      auto newp = BuilderZ.CreateCall(called, args);
      auto oldp = gutils->getNewFromOriginal(&call);
      gutils->replaceAWithB(oldp, newp);
      gutils->erase(oldp);

      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        auto ifound = gutils->invertedPointers.find(&call);
        assert(ifound != gutils->invertedPointers.end());
        auto placeholder = cast<CallInst>(&*ifound->second);
        if (!hasSingleGCPreserveEnd(&call, gutils)) {
          gutils->invertedPointers.erase(ifound);
          gutils->erase(placeholder);
          return true;
        }
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                "llvm.julia.gc_preserve_end",
                FunctionType::get(Builder2.getVoidTy(), call.getType(), false)),
            placeholder);
      }
      return true;
    }

    // void _FortranAAssign(Descriptor &to, const Descriptor &from,
    //                      const char *sourceFile, int sourceLine)
    // is LLVM flang's assignment between descriptors; its derivative is the
    // same assignment applied to the shadow descriptors.
    if (funcName == "_FortranAAssign" && call.arg_size() == 4) {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        if (!gutils->isConstantInstruction(&call)) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          Value *file = gutils->getNewFromOriginal(call.getArgOperand(2));
          Value *line = gutils->getNewFromOriginal(call.getArgOperand(3));
          Value *shadowTo =
              gutils->invertPointerM(call.getArgOperand(0), Builder2);
          Value *shadowFrom =
              gutils->invertPointerM(call.getArgOperand(1), Builder2);

          auto rule = [&](Value *sTo, Value *sFrom) {
            auto dcall = Builder2.CreateCall(called->getFunctionType(), called,
                                             {sTo, sFrom, file, line});
            dcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
          };
          applyChainRule(Builder2, rule, shadowTo, shadowFrom);

          eraseIfUnused(call);
          return true;
        }
      }
    }

    // void _FortranAEtime(const Descriptor *values, const Descriptor *time,
    //                     const char *sourceFile, int line)
    // is LLVM flang's ETIME. Its results are inactive, but it overwrites the
    // real(4) elements values(1:2) and time, whose shadow must then be zero:
    // after the call in forward mode, and in the reverse pass.
    if (funcName == "_FortranAEtime" && call.arg_size() == 4 &&
        (!gutils->isConstantValue(call.getArgOperand(0)) ||
         !gutils->isConstantValue(call.getArgOperand(1))) &&
        (Mode == DerivativeMode::ForwardMode ||
         Mode == DerivativeMode::ReverseModePrimal ||
         Mode == DerivativeMode::ReverseModeGradient ||
         Mode == DerivativeMode::ReverseModeCombined)) {
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      Type *I8PtrTy = getInt8PtrTy(call.getContext());
      Type *IdxTy = DL.getIntPtrType(call.getContext());
      unsigned P = DL.getPointerSize();
      // Offsets in a CFI descriptor: base_addr, and extent and sm of dim[0],
      // which follows elem_len, version, rank, type, attribute and extra.
      unsigned DimOff = 2 * P + 8;

      // The shadow elements to zero, or null where the shadow is the primal
      // or the element does not exist.
      auto shadowElements = [&](IRBuilder<> &B) {
        auto field = [&](Value *desc, Type *T, unsigned off) {
          Value *p = B.CreatePointerCast(desc, I8PtrTy);
          p = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), p, off);
          return B.CreateLoad(T, B.CreatePointerCast(p, getUnqual(T)));
        };
        SmallVector<Value *, 6> elems;
        for (unsigned i = 0; i < 2; ++i) {
          Value *orig = call.getArgOperand(i);
          if (gutils->isConstantValue(orig))
            continue;
          Value *desc = gutils->getNewFromOriginal(orig);
          Value *sdescs = gutils->invertPointerM(orig, B);
          Value *base = field(desc, I8PtrTy, 0);
          for (unsigned w = 0; w < gutils->getWidth(); ++w) {
            Value *sdesc = gutils->getWidth() == 1
                               ? sdescs
                               : gutils->extractMeta(B, sdescs, w);
            Value *sbase = field(sdesc, I8PtrTy, 0);
            Value *notPrimal = B.CreateICmpNE(sbase, base);
            Value *notNull = B.CreateIsNotNull(sbase);
            Value *valid = B.CreateAnd(notPrimal, notNull);
            auto add = [&](Value *cond, Value *addr) {
              elems.push_back(
                  B.CreateSelect(cond, addr, Constant::getNullValue(I8PtrTy)));
            };
            if (i == 1) {
              add(valid, sbase);
              continue;
            }
            Value *extent = field(desc, IdxTy, DimOff + P);
            Value *sm = field(desc, IdxTy, DimOff + 2 * P);
            Value *has1 = B.CreateICmpSGE(extent, ConstantInt::get(IdxTy, 1));
            add(B.CreateAnd(valid, has1), sbase);
            Value *has2 = B.CreateICmpSGE(extent, ConstantInt::get(IdxTy, 2));
            Value *valid2 = B.CreateAnd(valid, has2);
            add(valid2, B.CreateInBoundsGEP(B.getInt8Ty(), sbase, sm));
          }
        }
        return elems;
      };
      auto zero = [&](IRBuilder<> &B, Value *addr) {
        auto VT = FixedVectorType::get(B.getFloatTy(), 1);
        Value *ptr = B.CreatePointerCast(addr, getUnqual(VT));
        Value *notNull = B.CreateIsNotNull(addr);
        Value *mask = B.CreateVectorSplat(1, notNull);
        B.CreateMaskedStore(Constant::getNullValue(VT), ptr, Align(4), mask);
      };

      if (Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> B(newCall->getNextNode());
        for (auto addr : shadowElements(B))
          zero(B, addr);
        return true;
      }

      unsigned N = 0;
      for (unsigned i = 0; i < 2; ++i)
        if (!gutils->isConstantValue(call.getArgOperand(i)))
          N += (i == 0 ? 2 : 1) * gutils->getWidth();
      Type *TapeTy = ArrayType::get(I8PtrTy, N);
      Value *tape;
      if (Mode == DerivativeMode::ReverseModeGradient) {
        tape = BuilderZ.CreatePHI(TapeTy, 0);
      } else {
        tape = UndefValue::get(TapeTy);
        auto elems = shadowElements(BuilderZ);
        for (unsigned i = 0; i < N; ++i)
          tape = BuilderZ.CreateInsertValue(tape, elems[i], i);
        if (auto I = dyn_cast<Instruction>(tape))
          gutils->TapesToPreventRecomputation.insert(I);
      }
      tape = gutils->cacheForReverse(
          BuilderZ, tape, getIndex(&call, CacheType::Tape, BuilderZ));

      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        tape = lookup(tape, Builder2);
        for (unsigned i = 0; i < N; ++i)
          zero(Builder2, Builder2.CreateExtractValue(tape, i));
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return true;
    }

    // Other LLVM flang runtime functions acting on descriptors (allocation,
    // pointer association, initialization, copies): the derivative is the
    // same call on the shadow descriptors. Memory that the call allocates is
    // zeroed in the shadow, and re-initialized for derived types, whose
    // components hold descriptors.
    if (auto replay = getFlangShadowReplay(funcName)) {
      if ((Mode == DerivativeMode::ForwardMode ||
           Mode == DerivativeMode::ForwardModeError) &&
          !gutils->isConstantInstruction(&call)) {
        SmallVector<int, 2> shadowed;
        for (int i : replay->shadowArgs)
          if (i >= 0 && (unsigned)i < call.arg_size() &&
              !gutils->isConstantValue(call.getArgOperand(i)))
            shadowed.push_back(i);
        unsigned expected = 0;
        for (int i : replay->shadowArgs)
          if (i >= 0 && (unsigned)i < call.arg_size())
            expected++;
        // All descriptors inactive: nothing to replay, the primal call is all
        // there is (e.g. allocating an inactive variable next to an active
        // one in the same statement).
        if (shadowed.empty())
          return true;
        // A copy from an inactive into an active descriptor would need the
        // shadow to be zeroed rather than copied into: not handled.
        if (shadowed.size() == expected) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          auto &M = *called->getParent();
          auto &C = call.getContext();

          SmallVector<Value *, 8> primalArgs;
          for (auto &op : call.args())
            primalArgs.push_back(gutils->getNewFromOriginal(op));
          SmallVector<Value *, 2> shadows;
          for (int i : shadowed)
            shadows.push_back(
                gutils->invertPointerM(call.getArgOperand(i), Builder2));

          FunctionCallee sizeFn, initFn;
          if (replay->allocates) {
            auto I8Ptr = getInt8PtrTy(C);
            auto I32 = Type::getInt32Ty(C);
            sizeFn = M.getOrInsertFunction(
                "_FortranASize", FunctionType::get(Type::getInt64Ty(C),
                                                   {I8Ptr, I8Ptr, I32}, false));
            initFn = M.getOrInsertFunction(
                "_FortranAInitialize",
                FunctionType::get(Type::getVoidTy(C), {I8Ptr, I8Ptr, I32},
                                  false));
          }

          for (unsigned w = 0; w < gutils->getWidth(); w++) {
            SmallVector<Value *, 8> args(primalArgs);
            for (unsigned k = 0; k < shadowed.size(); k++)
              args[shadowed[k]] =
                  gutils->getWidth() > 1
                      ? gutils->extractMeta(Builder2, shadows[k], w)
                      : shadows[k];
            auto dcall =
                Builder2.CreateCall(called->getFunctionType(), called, args);
            dcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
            dcall->setCallingConv(call.getCallingConv());

            if (replay->allocates) {
              // base_addr and elem_len lead every descriptor.
              Value *desc = args[0];
              auto nullp = ConstantPointerNull::get(getInt8PtrTy(C));
              auto zero32 = ConstantInt::get(Type::getInt32Ty(C), 0);
              Value *count = Builder2.CreateCall(sizeFn, {desc, nullp, zero32});
              Value *elemLen = Builder2.CreateLoad(
                  Type::getInt64Ty(C), Builder2.CreateConstInBoundsGEP1_64(
                                           Type::getInt8Ty(C), desc, 8));
              Value *base = Builder2.CreateLoad(getInt8PtrTy(C), desc);
              Builder2.CreateMemSet(base, Builder2.getInt8(0),
                                    Builder2.CreateMul(count, elemLen),
                                    MaybeAlign());
              Builder2.CreateCall(initFn, {desc, nullp, zero32});
            }
          }

          eraseIfUnused(call);
          return true;
        }
      }
    }

    /*
     * int gsl_sf_legendre_array_e(const gsl_sf_legendre_t norm,
                                   const size_t lmax,
                                   const double x,
                                   const double csphase,
                                   double result_array[]);
    */
    // d L(n, x) / dx = L(n,x) * x * (n-1) + 1
    if (funcName == "gsl_sf_legendre_array_e") {
      if (gutils->isConstantValue(call.getArgOperand(4))) {
        eraseIfUnused(call);
        return true;
      }
      if (Mode == DerivativeMode::ReverseModePrimal) {
        eraseIfUnused(call);
        return true;
      }
      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModeGradient) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        ValueType BundleTypes[5] = {ValueType::None, ValueType::None,
                                    ValueType::None, ValueType::None,
                                    ValueType::Shadow};
        auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                               /*lookup*/ true);

        Type *types[6] = {
            call.getOperand(0)->getType(), call.getOperand(1)->getType(),
            call.getOperand(2)->getType(), call.getOperand(3)->getType(),
            call.getOperand(4)->getType(), call.getOperand(4)->getType(),
        };
        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        auto F = getOrInsertPerCallingConv(*called->getParent(), called,
                                           "gsl_sf_legendre_deriv_array_e", FT);

        llvm::Value *args[6] = {
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(0)),
                            Builder2),
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(1)),
                            Builder2),
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(2)),
                            Builder2),
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(3)),
                            Builder2),
            nullptr,
            nullptr};

        Type *typesS[] = {args[1]->getType()};
        FunctionType *FTS =
            FunctionType::get(args[1]->getType(), typesS, false);
        auto FS = getOrInsertPerCallingConv(*called->getParent(), called,
                                            "gsl_sf_legendre_array_n", FTS);
        Value *alSize = Builder2.CreateCall(FS, args[1]);
        Value *tmp = CreateAllocation(Builder2, types[2], alSize);
        Value *dtmp = CreateAllocation(Builder2, types[2], alSize);
        Builder2.CreateLifetimeStart(tmp);
        Builder2.CreateLifetimeStart(dtmp);

        args[4] = Builder2.CreateBitCast(tmp, types[4]);
        args[5] = Builder2.CreateBitCast(dtmp, types[5]);

        Builder2.CreateCall(F, args, Defs);
        Builder2.CreateLifetimeEnd(tmp);
        CreateDealloc(Builder2, tmp);

        BasicBlock *currentBlock = Builder2.GetInsertBlock();

        BasicBlock *loopBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_loop");
        BasicBlock *endBlock =
            gutils->addReverseBlock(loopBlock, currentBlock->getName() + "_end",
                                    /*fork*/ true, /*push*/ false);

        Builder2.CreateCondBr(
            Builder2.CreateICmpEQ(args[1], Constant::getNullValue(types[1])),
            endBlock, loopBlock);
        Builder2.SetInsertPoint(loopBlock);

        auto idx = Builder2.CreatePHI(types[1], 2);
        idx->addIncoming(ConstantInt::get(types[1], 0, false), currentBlock);

        auto acc_idx = Builder2.CreatePHI(types[2], 2);

        Value *inc = Builder2.CreateAdd(
            idx, ConstantInt::get(types[1], 1, false), "", true, true);
        idx->addIncoming(inc, loopBlock);
        acc_idx->addIncoming(Constant::getNullValue(types[2]), currentBlock);

        Value *idxs[] = {idx};
        Value *dtmp_idx = Builder2.CreateInBoundsGEP(types[2], dtmp, idxs);
        Value *d_req = Builder2.CreateInBoundsGEP(
            types[2],
            Builder2.CreatePointerCast(
                gutils->invertPointerM(call.getOperand(4), Builder2),
                getUnqual(types[2])),
            idxs);

        auto l0 = Builder2.CreateLoad(types[2], dtmp_idx);
        auto l1 = Builder2.CreateLoad(types[2], d_req);
        auto acc = Builder2.CreateFAdd(acc_idx, Builder2.CreateFMul(l0, l1));
        Builder2.CreateStore(Constant::getNullValue(types[2]), d_req);

        acc_idx->addIncoming(acc, loopBlock);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(inc, args[1]), endBlock,
                              loopBlock);

        Builder2.SetInsertPoint(endBlock);
        {
          auto found = gutils->reverseBlockToPrimal.find(endBlock);
          assert(found != gutils->reverseBlockToPrimal.end());
          SmallVector<BasicBlock *, 4> &vec =
              gutils->reverseBlocks[found->second];
          assert(vec.size());
          vec.push_back(endBlock);
        }

        auto fin_idx = Builder2.CreatePHI(types[2], 2);
        fin_idx->addIncoming(Constant::getNullValue(types[2]), currentBlock);
        fin_idx->addIncoming(acc, loopBlock);

        Builder2.CreateLifetimeEnd(dtmp);
        CreateDealloc(Builder2, dtmp);

        ((DiffeGradientUtils *)gutils)
            ->addToDiffe(call.getOperand(2), fin_idx, Builder2, types[2]);

        return true;
      }
    }

    // Functions that only modify pointers and don't allocate memory,
    // needs to be run on shadow in primal
    if (funcName == "_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_"
                    "node_baseS0_RS_") {
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (gutils->isConstantValue(call.getArgOperand(3)))
        return true;
      SmallVector<Value *, 2> args;
      for (auto &arg : call.args()) {
        if (gutils->isConstantValue(arg))
          args.push_back(gutils->getNewFromOriginal(arg));
        else
          args.push_back(gutils->invertPointerM(arg, BuilderZ));
      }
      BuilderZ.CreateCall(called, args);
      return true;
    }

    // Functions that initialize a shadow data structure (with no
    // other arguments) needs to be run on shadow in primal.
    if (funcName == "_ZNSt8ios_baseC2Ev" || funcName == "_ZNSt8ios_baseD2Ev" ||
        funcName == "_ZNSt6localeC1Ev" || funcName == "_ZNSt6localeD1Ev" ||
        funcName == "_ZNKSt5ctypeIcE13_M_widen_initEv") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (gutils->isConstantValue(call.getArgOperand(0)))
        return true;
      Value *args[] = {gutils->invertPointerM(call.getArgOperand(0), BuilderZ)};
      BuilderZ.CreateCall(called, args);
      return true;
    }

    if (funcName == "_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_"
                    "streambufIcS1_E") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (gutils->isConstantValue(call.getArgOperand(0)))
        return true;
      Value *args[] = {gutils->invertPointerM(call.getArgOperand(0), BuilderZ),
                       gutils->invertPointerM(call.getArgOperand(1), BuilderZ)};
      BuilderZ.CreateCall(called, args);
      return true;
    }

    // if constant instruction and readonly (thus must be pointer return)
    // and shadow return recomputable from shadow arguments.
    if (funcName == "__dynamic_cast" ||
        funcName == "_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base" ||
        funcName == "jl_ptr_to_array" || funcName == "jl_ptr_to_array_1d") {
      bool shouldCache = false;
      if (gutils->knownRecomputeHeuristic.find(&call) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[&call]) {
          shouldCache = true;
        }
      }
      ValueToValueMapTy empty;
      bool lrc = gutils->legalRecompute(&call, empty, nullptr);

      if (!gutils->isConstantValue(&call)) {
        auto ifound = gutils->invertedPointers.find(&call);
        assert(ifound != gutils->invertedPointers.end());
        auto placeholder = cast<PHINode>(&*ifound->second);

        if (subretType == DIFFE_TYPE::DUP_ARG) {
          Value *shadow = placeholder;
          if (lrc || Mode == DerivativeMode::ReverseModePrimal ||
              Mode == DerivativeMode::ReverseModeCombined ||
              Mode == DerivativeMode::ForwardMode ||
              Mode == DerivativeMode::ForwardModeError) {
            if (gutils->isConstantValue(call.getArgOperand(0)))
              shadow = gutils->getNewFromOriginal(&call);
            else {
              SmallVector<Value *, 2> args;
              size_t i = 0;
              for (auto &arg : call.args()) {
                if (gutils->isConstantValue(arg) ||
                    (funcName == "__dynamic_cast" && i > 0) ||
                    (funcName == "jl_ptr_to_array_1d" && i != 1) ||
                    (funcName == "jl_ptr_to_array" && i != 1))
                  args.push_back(gutils->getNewFromOriginal(arg));
                else
                  args.push_back(gutils->invertPointerM(arg, BuilderZ));
                i++;
              }
              shadow = BuilderZ.CreateCall(called, args);
            }
          }

          bool needsReplacement = true;
          if (!lrc && (Mode == DerivativeMode::ReverseModePrimal ||
                       Mode == DerivativeMode::ReverseModeGradient)) {
            shadow = gutils->cacheForReverse(
                BuilderZ, shadow, getIndex(&call, CacheType::Shadow, BuilderZ));
            if (Mode == DerivativeMode::ReverseModeGradient)
              needsReplacement = false;
          }
          gutils->invertedPointers.erase((const Value *)&call);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&call, InvertedPointerVH(gutils, shadow)));
          if (needsReplacement) {
            assert(shadow != placeholder);
            gutils->replaceAWithB(placeholder, shadow);
            gutils->erase(placeholder);
          }
        } else {
          gutils->invertedPointers.erase((const Value *)&call);
          gutils->erase(placeholder);
        }
      }

      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        eraseIfUnused(call);
        assert(gutils->isConstantInstruction(&call));
        return true;
      }

      if (!shouldCache && !lrc) {
        std::map<UsageKey, bool> Seen =
            gutils->populateSeenFromKnownRecompute();
        bool primalNeededInReverse =
            DifferentialUseAnalysis::is_value_needed_in_reverse<
                QueryType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);
        {
          auto found = gutils->knownRecomputeHeuristic.find(&call);
          if (found != gutils->knownRecomputeHeuristic.end()) {
            if (!found->second) {
              primalNeededInReverse = true;
            }
          }
        }
        shouldCache = primalNeededInReverse;
      }

      if (shouldCache) {
        BuilderZ.SetInsertPoint(newCall->getNextNode());
        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&call, CacheType::Self, BuilderZ));
      }
      eraseIfUnused(call);
      assert(gutils->isConstantInstruction(&call));
      return true;
    }

    if (called) {
      if (funcName == "julia.write_barrier" ||
          funcName == "julia.write_barrier_binding") {
        std::map<UsageKey, bool> Seen =
            gutils->populateSeenFromKnownRecompute();
        bool backwardsShadow = false;
        bool forwardsShadow = true;
        for (auto pair : gutils->backwardsOnlyShadows) {
          if (pair.second.stores.count(&call)) {
            backwardsShadow = true;
            forwardsShadow = pair.second.primalInitialize;
            if (auto inst = dyn_cast<Instruction>(pair.first))
              if (!forwardsShadow && pair.second.LI &&
                  pair.second.LI->contains(inst->getParent()))
                backwardsShadow = false;
            break;
          }
        }

        if (Mode == DerivativeMode::ForwardMode ||
            Mode == DerivativeMode::ForwardModeError ||
            (Mode == DerivativeMode::ReverseModeCombined &&
             (forwardsShadow || backwardsShadow)) ||
            (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
            (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow)) {
          IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
          for (size_t i = 0; i < gutils->getWidth(); i++) {
            SmallVector<Value *, 1> iargs;
            bool first = true;
            for (auto &arg : call.args()) {
              if (!gutils->isConstantValue(arg)) {
                Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
                if (gutils->getWidth() > 1) {
                  ptrshadow = gutils->extractMeta(BuilderZ, ptrshadow, i);
                }
                iargs.push_back(ptrshadow);
              } else {
                if (first)
                  break;
              }
              first = false;
            }
            if (iargs.size()) {
              BuilderZ.CreateCall(called, iargs);
            }
          }
        }

        bool forceErase = false;
        if (Mode == DerivativeMode::ReverseModeGradient) {

          // Since we won't redo the store in the reverse pass, do not
          // force the write barrier.
          forceErase = true;
          for (const auto &pair : gutils->rematerializableAllocations) {
            if (!pair.second.stores.count(&call))
              continue;
            if (gutils->allocationsToBeRematerialized.count(pair.first))
              // However, if we are rematerailizing the allocation and not
              // inside the loop level rematerialization, we do still need the
              // reverse passes ``fake primal'' store and therefore write
              // barrier
              if (!pair.second.LI || !pair.second.LI->contains(&call)) {
                forceErase = false;
              }
          }
        }
        if (forceErase)
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        else
          eraseIfUnused(call);

        return true;
      }
      Intrinsic::ID ID = Intrinsic::not_intrinsic;
      if (isMemFreeLibMFunction(funcName, &ID)) {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            gutils->isConstantInstruction(&call)) {

          if (gutils->knownRecomputeHeuristic.find(&call) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[&call]) {
              gutils->cacheForReverse(
                  BuilderZ, newCall,
                  getIndex(&call, CacheType::Self, BuilderZ));
            }
          }
          eraseIfUnused(call);
          return true;
        }

        if (ID != Intrinsic::not_intrinsic) {
          SmallVector<Value *, 2> orig_ops(call.getNumOperands());
          for (unsigned i = 0; i < call.getNumOperands(); ++i) {
            orig_ops[i] = call.getOperand(i);
          }
          bool cached = handleAdjointForIntrinsic(ID, call, orig_ops);
          if (!cached) {
            if (gutils->knownRecomputeHeuristic.find(&call) !=
                gutils->knownRecomputeHeuristic.end()) {
              if (!gutils->knownRecomputeHeuristic[&call]) {
                gutils->cacheForReverse(
                    BuilderZ, newCall,
                    getIndex(&call, CacheType::Self, BuilderZ));
              }
            }
          }
          eraseIfUnused(call);
          return true;
        }
      }
    }
  }
  if (auto assembly = dyn_cast<InlineAsm>(call.getCalledOperand())) {
    if (assembly->getAsmString() == "maxpd $1, $0") {
      if (Mode == DerivativeMode::ReverseModePrimal ||
          gutils->isConstantInstruction(&call)) {

        if (gutils->knownRecomputeHeuristic.find(&call) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[&call]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(&call, CacheType::Self, BuilderZ));
          }
        }
        eraseIfUnused(call);
        return true;
      }

      SmallVector<Value *, 2> orig_ops(call.getNumOperands());
      for (unsigned i = 0; i < call.getNumOperands(); ++i) {
        orig_ops[i] = call.getOperand(i);
      }
      handleAdjointForIntrinsic(Intrinsic::maxnum, call, orig_ops);
      if (gutils->knownRecomputeHeuristic.find(&call) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[&call]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(&call, CacheType::Self, BuilderZ));
        }
      }
      eraseIfUnused(call);
      return true;
    }
  }

  if (funcName == "realloc") {
    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      if (!gutils->isConstantValue(&call)) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        auto dbgLoc = gutils->getNewFromOriginal(&call)->getDebugLoc();

        auto rule = [&](Value *ip) {
          ValueType BundleTypes[2] = {ValueType::Shadow, ValueType::Primal};

          auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                                 /*lookup*/ false);

          llvm::Value *args[2] = {
              ip, gutils->getNewFromOriginal(call.getOperand(1))};
          CallInst *CI = Builder2.CreateCall(
              call.getFunctionType(), call.getCalledFunction(), args, Defs);
          CI->setAttributes(call.getAttributes());
          CI->setCallingConv(call.getCallingConv());
          CI->setTailCallKind(call.getTailCallKind());
          CI->setDebugLoc(dbgLoc);
          return CI;
        };

        Value *CI = applyChainRule(
            call.getType(), Builder2, rule,
            gutils->invertPointerM(call.getOperand(0), Builder2));

        auto found = gutils->invertedPointers.find(&call);
        PHINode *placeholder = cast<PHINode>(&*found->second);

        gutils->invertedPointers.erase(found);
        gutils->replaceAWithB(placeholder, CI);
        gutils->erase(placeholder);
        gutils->invertedPointers.insert(
            std::make_pair(&call, InvertedPointerVH(gutils, CI)));
      }
      eraseIfUnused(call);
      return true;
    }
  }

  if (isAllocationFunction(funcName, gutils->TLI)) {

    bool constval = gutils->isConstantValue(&call);

    if (!constval) {
      auto dbgLoc = gutils->getNewFromOriginal(&call)->getDebugLoc();
      auto found = gutils->invertedPointers.find(&call);
      PHINode *placeholder = cast<PHINode>(&*found->second);
      IRBuilder<> bb(placeholder);

      SmallVector<Value *, 8> args;
      for (auto &arg : call.args()) {
        args.push_back(gutils->getNewFromOriginal(arg));
      }

      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ForwardModeSplit) {

        Value *anti = placeholder;
        // If rematerializable allocations and split mode, we can
        // simply elect to build the entire piece in the reverse
        // since it should be possible to perform any shadow stores
        // of pointers (from rematerializable property) and it does
        // not escape the function scope (lest it not be
        // rematerializable) so all input derivatives remain zero.
        bool backwardsShadow = false;
        bool forwardsShadow = true;
        bool inLoop = false;
        bool isAlloca = isa<AllocaInst>(&call);
        {
          auto found = gutils->backwardsOnlyShadows.find(&call);
          if (found != gutils->backwardsOnlyShadows.end()) {
            backwardsShadow = true;
            forwardsShadow = found->second.primalInitialize;
            // If in a loop context, maintain the same free behavior.
            if (found->second.LI &&
                found->second.LI->contains(call.getParent()))
              inLoop = true;
          }
        }
        {

          if (!forwardsShadow) {
            if (Mode == DerivativeMode::ReverseModePrimal) {
              // Needs a stronger replacement check/assertion.
              Value *replacement = getUndefinedValueForType(
                  *gutils->oldFunc->getParent(), placeholder->getType());
              gutils->replaceAWithB(placeholder, replacement);
              gutils->invertedPointers.erase(found);
              gutils->invertedPointers.insert(std::make_pair(
                  &call, InvertedPointerVH(gutils, replacement)));
              gutils->erase(placeholder);
              anti = nullptr;
              goto endAnti;
            } else if (inLoop) {
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  placeholder);
              if (hasMetadata(&call, "enzyme_fromstack"))
                isAlloca = true;
              goto endAnti;
            }
          }
          placeholder->setName("");
          if (shadowHandlers.find(funcName) != shadowHandlers.end()) {
            bb.SetInsertPoint(placeholder);

            if (Mode == DerivativeMode::ReverseModeCombined ||
                (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
                (Mode == DerivativeMode::ReverseModeGradient &&
                 backwardsShadow)) {
              anti = applyChainRule(call.getType(), bb, [&]() {
                return shadowHandlers[funcName](bb, &call, args, gutils);
              });
              if (anti->getType() != placeholder->getType()) {
                llvm::errs() << "orig: " << call << "\n";
                llvm::errs() << "placeholder: " << *placeholder << "\n";
                llvm::errs() << "anti: " << *anti << "\n";
              }
              gutils->invertedPointers.erase(found);
              bb.SetInsertPoint(placeholder);

              gutils->replaceAWithB(placeholder, anti);
              gutils->erase(placeholder);
            }

            if (auto inst = dyn_cast<Instruction>(anti))
              bb.SetInsertPoint(inst);

            if (!backwardsShadow)
              anti = gutils->cacheForReverse(
                  bb, anti, getIndex(&call, CacheType::Shadow, BuilderZ));
          } else {
            bool zeroed = false;
            uint64_t idx = 0;
            Value *prev = nullptr;
            ;
            auto rule = [&]() {
              Value *anti =
                  bb.CreateCall(call.getFunctionType(), call.getCalledOperand(),
                                args, call.getName() + "'mi");
              cast<CallInst>(anti)->setAttributes(call.getAttributes());
              cast<CallInst>(anti)->setCallingConv(call.getCallingConv());
              cast<CallInst>(anti)->setTailCallKind(call.getTailCallKind());
              cast<CallInst>(anti)->setDebugLoc(dbgLoc);

              if (anti->getType()->isPointerTy()) {
                cast<CallInst>(anti)->addAttributeAtIndex(
                    AttributeList::ReturnIndex, Attribute::NoAlias);
                cast<CallInst>(anti)->addAttributeAtIndex(
                    AttributeList::ReturnIndex, Attribute::NonNull);

                if (funcName == "malloc" || funcName == "_Znwm" ||
                    funcName == "??2@YAPAXI@Z" ||
                    funcName == "??2@YAPEAX_K@Z") {
                  if (auto ci = dyn_cast<ConstantInt>(args[0])) {
                    unsigned derefBytes = ci->getLimitedValue();
                    CallInst *cal =
                        cast<CallInst>(gutils->getNewFromOriginal(&call));
                    cast<CallInst>(anti)->addDereferenceableRetAttr(derefBytes);
                    cal->addDereferenceableRetAttr(derefBytes);
#if !defined(FLANG) && !defined(ROCM)
                    AttrBuilder B(ci->getContext());
#else
                    AttrBuilder B;
#endif
                    B.addDereferenceableOrNullAttr(derefBytes);
                    cast<CallInst>(anti)->setAttributes(
                        cast<CallInst>(anti)->getAttributes().addRetAttributes(
                            call.getContext(), B));
                    cal->setAttributes(cal->getAttributes().addRetAttributes(
                        call.getContext(), B));
                    cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                             Attribute::NoAlias);
                    cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                             Attribute::NonNull);
                  }
                }
                if (funcName == "julia.gc_alloc_obj" ||
                    funcName == "jl_gc_alloc_typed" ||
                    funcName == "ijl_gc_alloc_typed") {
                  if (EnzymeShadowAllocRewrite) {
                    bool used = unnecessaryInstructions.find(&call) ==
                                unnecessaryInstructions.end();
                    EnzymeShadowAllocRewrite(wrap(anti), gutils, wrap(&call),
                                             idx, wrap(prev), used);
                  }
                }
              }
              if (Mode == DerivativeMode::ReverseModeCombined ||
                  (Mode == DerivativeMode::ReverseModePrimal &&
                   forwardsShadow) ||
                  (Mode == DerivativeMode::ReverseModeGradient &&
                   backwardsShadow) ||
                  (Mode == DerivativeMode::ForwardModeSplit &&
                   backwardsShadow)) {
                if (!inLoop) {
                  zeroKnownAllocation(bb, anti, args, funcName, gutils->TLI,
                                      &call);
                  zeroed = true;
                }
              }
              idx++;
              prev = anti;
              return anti;
            };

            anti = applyChainRule(call.getType(), bb, rule);

            gutils->invertedPointers.erase(found);
            if (&*bb.GetInsertPoint() == placeholder)
              bb.SetInsertPoint(placeholder->getNextNode());
            gutils->replaceAWithB(placeholder, anti);
            gutils->erase(placeholder);

            if (!backwardsShadow)
              anti = gutils->cacheForReverse(
                  bb, anti, getIndex(&call, CacheType::Shadow, BuilderZ));
            else {
              if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
                isAlloca = true;
                bb.SetInsertPoint(cast<Instruction>(anti));
                Value *Size;
                if (funcName == "malloc")
                  Size = args[0];
                else if (funcName == "julia.gc_alloc_obj" ||
                         funcName == "jl_gc_alloc_typed" ||
                         funcName == "ijl_gc_alloc_typed")
                  Size = args[1];
                else
                  llvm_unreachable("Unknown allocation to upgrade");

                Type *elTy = Type::getInt8Ty(call.getContext());
                if (MD->getNumOperands() == 2) {
                  elTy = (Type *)cast<ConstantInt>(
                             cast<ConstantAsMetadata>(MD->getOperand(1))
                                 ->getValue())
                             ->getLimitedValue();
                  Value *tsize = ConstantInt::get(
                      Size->getType(), (gutils->newFunc->getParent()
                                            ->getDataLayout()
                                            .getTypeAllocSizeInBits(elTy) +
                                        7) /
                                           8);

                  Size = bb.CreateUDiv(Size, tsize, "", /*exact*/ true);
                }
                std::string name = "";
#if LLVM_VERSION_MAJOR < 17
                if (call.getContext().supportsTypedPointers()) {
                  for (auto U : call.users()) {
                    if (hasMetadata(cast<Instruction>(U), "enzyme_caststack")) {
                      if (MD->getNumOperands() == 1) {
                        elTy = U->getType()->getPointerElementType();
                        Value *tsize = ConstantInt::get(
                            Size->getType(),
                            (gutils->newFunc->getParent()
                                 ->getDataLayout()
                                 .getTypeAllocSizeInBits(elTy) +
                             7) /
                                8);

                        Size = bb.CreateUDiv(Size, tsize, "", /*exact*/ true);
                      }
                      name = (U->getName() + "'ai").str();
                      break;
                    }
                  }
                }
#endif
                auto rule = [&](Value *anti) {
                  bb.SetInsertPoint(cast<Instruction>(anti));
                  Value *replacement = bb.CreateAlloca(elTy, Size, name);
                  if (name.size() == 0)
                    replacement->takeName(anti);
                  else
                    anti->setName("");
                  auto Alignment = cast<ConstantInt>(cast<ConstantAsMetadata>(
                                                         MD->getOperand(0))
                                                         ->getValue())
                                       ->getLimitedValue();
                  if (Alignment) {
                    cast<AllocaInst>(replacement)
                        ->setAlignment(Align(Alignment));
                  }
#if LLVM_VERSION_MAJOR < 17
                  if (call.getContext().supportsTypedPointers()) {
                    if (anti->getType()->getPointerElementType() != elTy)
                      replacement = bb.CreatePointerCast(
                          replacement,
                          getUnqual(anti->getType()->getPointerElementType()));
                  }
#endif
                  auto PT = cast<PointerType>(anti->getType());
                  if (PT->getAddressSpace()) {
                    replacement = bb.CreateAddrSpaceCast(replacement, PT);
                    cast<Instruction>(replacement)
                        ->setMetadata(
                            "enzyme_backstack",
                            MDNode::get(replacement->getContext(), {}));
                  }
                  gutils->replaceAWithB(cast<Instruction>(anti), replacement);
                  bb.SetInsertPoint(cast<Instruction>(anti)->getNextNode());
                  gutils->erase(cast<Instruction>(anti));
                  return replacement;
                };

                auto replacement =
                    applyChainRule(call.getType(), bb, rule, anti);
                anti = replacement;
              }
            }

            if (Mode == DerivativeMode::ReverseModeCombined ||
                (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
                (Mode == DerivativeMode::ReverseModeGradient &&
                 backwardsShadow) ||
                (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow)) {
              if (!inLoop) {
                assert(zeroed);
              }
            }
          }
          gutils->invertedPointers.insert(
              std::make_pair(&call, InvertedPointerVH(gutils, anti)));
        }
      endAnti:;
        if (((Mode == DerivativeMode::ReverseModeCombined && shouldFree()) ||
             (Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
             (Mode == DerivativeMode::ForwardModeSplit && shouldFree())) &&
            !isAlloca) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          assert(anti);
          Value *tofree = lookup(anti, Builder2);
          assert(tofree);
          assert(tofree->getType());
          for (size_t i = 0; i < gutils->getWidth(); i++) {
            Value *tofree_i =
                gutils->getWidth() == 1
                    ? tofree
                    : GradientUtils::extractMeta(Builder2, tofree, i);

            auto CI = freeKnownAllocation(Builder2, tofree_i, funcName, dbgLoc,
                                          gutils->TLI, &call, gutils);
            if (CI) {
              CI->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                      Attribute::NonNull);
              bool combined = Mode == DerivativeMode::ReverseModeCombined;
              auto ident = MDNode::getDistinct(
                  CI->getContext(),
                  {ConstantAsMetadata::get(
                      combined ? ConstantInt::getTrue(CI->getContext())
                               : ConstantInt::getFalse(CI->getContext()))});
              Value *anti_i =
                  gutils->getWidth() == 1
                      ? anti
                      : GradientUtils::extractMeta(Builder2, anti, i);
              cast<Instruction>(anti_i)->setMetadata(
                  "enzyme_cache_alloc", MDNode::get(CI->getContext(), {ident}));
              CI->setMetadata("enzyme_cache_free",
                              MDNode::get(CI->getContext(), {ident}));
            }
          }
        }
      } else if (Mode == DerivativeMode::ForwardMode ||
                 Mode == DerivativeMode::ForwardModeError) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        SmallVector<Value *, 2> args;
        for (unsigned i = 0; i < call.arg_size(); ++i) {
          auto arg = call.getArgOperand(i);
          args.push_back(gutils->getNewFromOriginal(arg));
        }

        uint64_t idx = 0;
        Value *prev = gutils->getNewFromOriginal(&call);
        auto rule = [&]() {
          SmallVector<ValueType, 2> BundleTypes(args.size(), ValueType::Primal);

          auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                                 /*lookup*/ false);

          CallInst *CI = Builder2.CreateCall(
              call.getFunctionType(), call.getCalledFunction(), args, Defs);
          CI->setAttributes(call.getAttributes());
          CI->setCallingConv(call.getCallingConv());
          CI->setTailCallKind(call.getTailCallKind());
          CI->setDebugLoc(dbgLoc);

          if (funcName == "julia.gc_alloc_obj" ||
              funcName == "jl_gc_alloc_typed" ||
              funcName == "ijl_gc_alloc_typed") {
            if (EnzymeShadowAllocRewrite) {
              bool used = unnecessaryInstructions.find(&call) ==
                          unnecessaryInstructions.end();
              EnzymeShadowAllocRewrite(wrap(CI), gutils, wrap(&call), idx,
                                       wrap(prev), used);
            }
          }
          idx++;
          prev = CI;
          return CI;
        };

        Value *CI = applyChainRule(call.getType(), Builder2, rule);

        auto found = gutils->invertedPointers.find(&call);
        PHINode *placeholder = cast<PHINode>(&*found->second);

        gutils->invertedPointers.erase(found);
        gutils->replaceAWithB(placeholder, CI);
        gutils->erase(placeholder);
        gutils->invertedPointers.insert(
            std::make_pair(&call, InvertedPointerVH(gutils, CI)));
      }
    }

    // Cache and rematerialization irrelevant for forward mode.
    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      eraseIfUnused(call);
      return true;
    }

    std::map<UsageKey, bool> Seen = gutils->populateSeenFromKnownRecompute();
    bool primalNeededInReverse =
        Mode == DerivativeMode::ForwardMode ||
                Mode == DerivativeMode::ForwardModeError
            ? false
            : DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);

    // If we explicitly decided we need this in the reverse pass, mark it as
    // such.
    {
      auto found = gutils->knownRecomputeHeuristic.find(&call);
      if (found != gutils->knownRecomputeHeuristic.end() && !found->second) {
        primalNeededInReverse = true;
      }
    }
    bool cacheWholeAllocation = gutils->needsCacheWholeAllocation(&call);
    if (cacheWholeAllocation) {
      primalNeededInReverse = true;
    }

    auto restoreFromStack = [&](MDNode *MD) {
      IRBuilder<> B(newCall);
      Value *Size;
      if (funcName == "malloc")
        Size = call.getArgOperand(0);
      else if (funcName == "julia.gc_alloc_obj" ||
               funcName == "jl_gc_alloc_typed" ||
               funcName == "ijl_gc_alloc_typed")
        Size = call.getArgOperand(1);
      else
        llvm_unreachable("Unknown allocation to upgrade");
      Size = gutils->getNewFromOriginal(Size);

      if (isa<ConstantInt>(Size)) {
        B.SetInsertPoint(gutils->inversionAllocs);
      }
      Type *elTy = Type::getInt8Ty(call.getContext());
      if (MD->getNumOperands() == 2) {
        elTy = (Type *)cast<ConstantInt>(
                   cast<ConstantAsMetadata>(MD->getOperand(1))->getValue())
                   ->getLimitedValue();
        Value *tsize = ConstantInt::get(Size->getType(),
                                        (gutils->newFunc->getParent()
                                             ->getDataLayout()
                                             .getTypeAllocSizeInBits(elTy) +
                                         7) /
                                            8);
        Size = B.CreateUDiv(Size, tsize, "", /*exact*/ true);
      }
      Instruction *I = nullptr;
#if LLVM_VERSION_MAJOR < 17
      if (call.getContext().supportsTypedPointers()) {
        for (auto U : call.users()) {
          if (hasMetadata(cast<Instruction>(U), "enzyme_caststack")) {
            if (MD->getNumOperands() == 1) {
              elTy = U->getType()->getPointerElementType();
              Value *tsize = ConstantInt::get(
                  Size->getType(), (gutils->newFunc->getParent()
                                        ->getDataLayout()
                                        .getTypeAllocSizeInBits(elTy) +
                                    7) /
                                       8);

              Size = B.CreateUDiv(Size, tsize, "", /*exact*/ true);
            }
            I = gutils->getNewFromOriginal(cast<Instruction>(U));
            break;
          }
        }
      }
#endif
      Value *replacement = B.CreateAlloca(elTy, Size);
      for (auto MD : {"enzyme_active", "enzyme_inactive", "enzyme_type",
                      "enzymejl_allocart", "enzymejl_allocart_name",
                      "enzymejl_gc_alloc_rt"})
        if (auto M = call.getMetadata(MD))
          cast<AllocaInst>(replacement)->setMetadata(MD, M);
      if (I)
        replacement->takeName(I);
      else
        replacement->takeName(newCall);
      auto Alignment =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
              ->getLimitedValue();
      // Don't set zero alignment
      if (Alignment) {
        cast<AllocaInst>(replacement)->setAlignment(Align(Alignment));
      }
#if LLVM_VERSION_MAJOR < 17
      if (call.getContext().supportsTypedPointers()) {
        if (call.getType()->getPointerElementType() != elTy)
          replacement = B.CreatePointerCast(
              replacement, getUnqual(call.getType()->getPointerElementType()));
      }
#endif
      auto PT = cast<PointerType>(call.getType());
      if (PT->getAddressSpace()) {
        replacement = B.CreateAddrSpaceCast(replacement, PT);
        cast<Instruction>(replacement)
            ->setMetadata("enzyme_backstack",
                          MDNode::get(replacement->getContext(), {}));
      }
      gutils->replaceAWithB(newCall, replacement);
      gutils->erase(newCall);
    };

    // Don't erase any allocation that is being rematerialized.
    {
      auto found = gutils->rematerializableAllocations.find(&call);
      if (found != gutils->rematerializableAllocations.end()) {
        // If rematerializing (e.g. needed in reverse, but not needing
        //  the whole allocation):
        if (gutils->allocationsToBeRematerialized.count(&call)) {
          assert(!unnecessaryValues.count(&call));
          // if rematerialize, don't ever cache and downgrade to stack
          // allocation where possible. Note that for allocations which are
          // within a loop, we will create the rematerialized allocation in the
          // rematerialied loop. Note that what matters here is whether the
          // actual call itself here is inside the loop, not whether the
          // rematerialization is loop level. This is because one can have a
          // loop level cache, but a function level allocation (e.g. for stack
          // allocas). If we deleted it here, we would have no allocation!
          auto AllocationLoop = gutils->OrigLI->getLoopFor(call.getParent());
          // An allocation within a loop, must definitionally be a loop level
          // allocation (but not always the other way around.
          if (AllocationLoop)
            assert(found->second.LI);
          if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
            if (Mode == DerivativeMode::ReverseModeGradient && AllocationLoop) {
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  newCall);
            } else {
              restoreFromStack(MD);
            }
            return true;
          }

          // No need to free GC.
          if (EnzymeJuliaAddrLoad && isa<PointerType>(call.getType()) &&
              cast<PointerType>(call.getType())->getAddressSpace() == 10) {
            if (Mode == DerivativeMode::ReverseModeGradient && AllocationLoop)
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  newCall);
            return true;
          }

          // Otherwise if in reverse pass, free the newly created allocation.
          if (Mode == DerivativeMode::ReverseModeGradient ||
              Mode == DerivativeMode::ReverseModeCombined ||
              Mode == DerivativeMode::ForwardModeSplit) {
            IRBuilder<> Builder2(&call);
            getReverseBuilder(Builder2);
            auto dbgLoc = gutils->getNewFromOriginal(call.getDebugLoc());
            auto freecall = freeKnownAllocation(
                Builder2, lookup(newCall, Builder2), funcName, dbgLoc,
                gutils->TLI, &call, gutils);
            if (freecall) {
              auto ident = MDNode::getDistinct(
                  freecall->getContext(),
                  {ConstantAsMetadata::get(
                      ConstantInt::getTrue(freecall->getContext()))});
              newCall->setMetadata(
                  "enzyme_cache_alloc",
                  MDNode::get(freecall->getContext(), {ident}));
              freecall->setMetadata(
                  "enzyme_cache_free",
                  MDNode::get(freecall->getContext(), {ident}));
            }
            if (Mode == DerivativeMode::ReverseModeGradient && AllocationLoop)
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  newCall);
            return true;
          }
          // If in primal, do nothing (keeping the original caching behavior)
          if (Mode == DerivativeMode::ReverseModePrimal)
            return true;
        } else if (!cacheWholeAllocation) {
          if (unnecessaryValues.count(&call)) {
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
            return true;
          }
          // If not caching allocation and not needed in the reverse, we can
          // use the original freeing behavior for the function. If in the
          // reverse pass we should not recreate this allocation.
          if (Mode == DerivativeMode::ReverseModeGradient)
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          else if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
            restoreFromStack(MD);
          }
          return true;
        }
      }
    }

    // If an allocation is not needed in the reverse, maintain the original
    // free behavior and do not rematerialize this for the reverse. However,
    // this is only safe to perform for allocations with a guaranteed free
    // as can we can only guarantee that we don't erase those frees.
    bool hasPDFree = gutils->allocationsWithGuaranteedFree.count(&call);
    if (!primalNeededInReverse && hasPDFree) {
      if (unnecessaryValues.count(&call)) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
          restoreFromStack(MD);
        }
      }
      return true;
    }

    // If an object is managed by the GC do not preserve it for later free,
    // Thus it only needs caching if there is a need for it in the reverse.
    if (EnzymeJuliaAddrLoad && isa<PointerType>(call.getType()) &&
        cast<PointerType>(call.getType())->getAddressSpace() == 10) {
      if (!subretused) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (!primalNeededInReverse) {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          auto pn = BuilderZ.CreatePHI(call.getType(), 1,
                                       call.getName() + "_replacementJ");
          gutils->fictiousPHIs[pn] = &call;
          gutils->replaceAWithB(newCall, pn);
          gutils->erase(newCall);
        }
      } else if (Mode != DerivativeMode::ReverseModeCombined) {
        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&call, CacheType::Self, BuilderZ));
      }
      return true;
    }

    if (EnzymeFreeInternalAllocations)
      hasPDFree = true;

    // TODO enable this if we need to free the memory
    // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
    // TO FREE'ing
    if ((primalNeededInReverse &&
         !gutils->unnecessaryIntermediates.count(&call)) ||
        hasPDFree) {
      if (hasPDFree && Mode == DerivativeMode::ReverseModePrimal) {
        auto ident =
            MDNode::getDistinct(newCall->getContext(),
                                {ConstantAsMetadata::get(ConstantInt::getFalse(
                                    newCall->getContext()))});
        newCall->setMetadata("enzyme_cache_alloc",
                             MDNode::get(newCall->getContext(), {ident}));
      }
      Value *nop = gutils->cacheForReverse(
          BuilderZ, newCall, getIndex(&call, CacheType::Self, BuilderZ));
      if (hasPDFree &&
          ((Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
           Mode == DerivativeMode::ReverseModeCombined ||
           (Mode == DerivativeMode::ForwardModeSplit && shouldFree()))) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto dbgLoc = gutils->getNewFromOriginal(call.getDebugLoc());
        auto freecall =
            freeKnownAllocation(Builder2, lookup(nop, Builder2), funcName,
                                dbgLoc, gutils->TLI, &call, gutils);
        if (freecall) {
          bool combined = Mode == DerivativeMode::ReverseModeCombined;
          auto ident = MDNode::getDistinct(
              freecall->getContext(),
              {ConstantAsMetadata::get(
                  combined ? ConstantInt::getTrue(freecall->getContext())
                           : ConstantInt::getFalse(freecall->getContext()))});
          if (combined)
            newCall->setMetadata("enzyme_cache_alloc",
                                 MDNode::get(freecall->getContext(), {ident}));
          freecall->setMetadata("enzyme_cache_free",
                                MDNode::get(freecall->getContext(), {ident}));
        }
      }
    } else if (Mode == DerivativeMode::ReverseModeGradient ||
               Mode == DerivativeMode::ReverseModeCombined ||
               Mode == DerivativeMode::ForwardModeSplit) {
      // Note that here we cannot simply replace with null as users who
      // try to find the shadow pointer will use the shadow of null rather
      // than the true shadow of this
      auto pn = BuilderZ.CreatePHI(call.getType(), 1,
                                   call.getName() + "_replacementB");
      gutils->fictiousPHIs[pn] = &call;
      gutils->replaceAWithB(newCall, pn);
      gutils->erase(newCall);
    }

    return true;
  }

  if (funcName == "julia.gc_loaded") {
    if (gutils->isConstantValue(&call)) {
      eraseIfUnused(call);
      return true;
    }
    auto ifound = gutils->invertedPointers.find(&call);
    assert(ifound != gutils->invertedPointers.end());

    if (auto placeholder = dyn_cast<PHINode>(&*ifound->second)) {

      bool needShadow = DifferentialUseAnalysis::is_value_needed_in_reverse<
          QueryType::Shadow>(gutils, &call, Mode, oldUnreachable);
      if (!needShadow) {
        gutils->invertedPointers.erase(ifound);
        gutils->erase(placeholder);
        eraseIfUnused(call);
        return true;
      }

      gutils->invertedPointers.erase(ifound);
      auto res = gutils->invertPointerM(&call, BuilderZ);

      gutils->replaceAWithB(placeholder, res);
      gutils->erase(placeholder);
    }
    eraseIfUnused(call);

    return true;
  }

  if (funcName == "julia.pointer_from_objref") {
    if (gutils->isConstantValue(&call)) {
      eraseIfUnused(call);
      return true;
    }

    auto ifound = gutils->invertedPointers.find(&call);
    assert(ifound != gutils->invertedPointers.end());

    auto placeholder = cast<PHINode>(&*ifound->second);

    bool needShadow =
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Shadow>(
            gutils, &call, Mode, oldUnreachable);
    if (!needShadow) {
      gutils->invertedPointers.erase(ifound);
      gutils->erase(placeholder);
      eraseIfUnused(call);
      return true;
    }

    Value *ptrshadow = gutils->invertPointerM(call.getArgOperand(0), BuilderZ);

    Value *val = applyChainRule(
        call.getType(), BuilderZ,
        [&](Value *v) -> Value * { return BuilderZ.CreateCall(called, {v}); },
        ptrshadow);

    gutils->replaceAWithB(placeholder, val);
    gutils->erase(placeholder);
    eraseIfUnused(call);
    return true;
  }
  if (funcName.contains("__enzyme_todense")) {
    if (gutils->isConstantValue(&call)) {
      eraseIfUnused(call);
      return true;
    }

    auto ifound = gutils->invertedPointers.find(&call);
    assert(ifound != gutils->invertedPointers.end());

    auto placeholder = cast<PHINode>(&*ifound->second);

    bool needShadow =
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Shadow>(
            gutils, &call, Mode, oldUnreachable);
    if (!needShadow) {
      gutils->invertedPointers.erase(ifound);
      gutils->erase(placeholder);
      eraseIfUnused(call);
      return true;
    }

    // The shadow pointer is built from the shadows of the extra arguments.
    // An inactive argument (e.g. an index) is passed as is to every lane.
    SmallVector<Value *, 3> args;
    SmallVector<bool, 3> argIsShadow;
    for (size_t i = 0; i < 2; i++) {
      args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
      argIsShadow.push_back(false);
    }
    for (size_t i = 2; i < call.arg_size(); ++i) {
      auto arg = call.getArgOperand(i);
      if (gutils->isConstantValue(arg)) {
        args.push_back(gutils->getNewFromOriginal(arg));
        argIsShadow.push_back(false);
      } else {
        args.push_back(gutils->invertPointerM(arg, BuilderZ));
        argIsShadow.push_back(true);
      }
    }

    Value *res = UndefValue::get(gutils->getShadowType(call.getType()));
    if (gutils->getWidth() == 1) {
      res = BuilderZ.CreateCall(called, args);
    } else {
      for (size_t w = 0; w < gutils->getWidth(); ++w) {
        SmallVector<Value *, 3> targs = {args[0], args[1]};
        for (size_t i = 2; i < call.arg_size(); ++i)
          targs.push_back(argIsShadow[i]
                              ? GradientUtils::extractMeta(BuilderZ, args[i], w)
                              : args[i]);

        auto tres = BuilderZ.CreateCall(called, targs);
        res = BuilderZ.CreateInsertValue(res, tres, w);
      }
    }

    gutils->replaceAWithB(placeholder, res);
    gutils->erase(placeholder);
    eraseIfUnused(call);
    return true;
  }

  if (funcName == "memcpy" || funcName == "memmove") {
    auto ID = (funcName == "memcpy") ? Intrinsic::memcpy : Intrinsic::memmove;
    visitMemTransferCommon(ID, /*srcAlign*/ MaybeAlign(1),
                           /*dstAlign*/ MaybeAlign(1), call,
                           call.getArgOperand(0), call.getArgOperand(1),
                           gutils->getNewFromOriginal(call.getArgOperand(2)),
                           ConstantInt::getFalse(call.getContext()));
    return true;
  }
  if (funcName == "memset" || funcName == "memset_pattern16" ||
      funcName == "__memset_chk") {
    visitMemSetCommon(call);
    return true;
  }
  if (funcName == "enzyme_zerotype") {
    IRBuilder<> BuilderZ(&call);
    getForwardBuilder(BuilderZ);

    bool backwardsShadow = false;
    bool forwardsShadow = true;
    for (auto pair : gutils->backwardsOnlyShadows) {
      if (pair.second.stores.count(&call)) {
        backwardsShadow = true;
        forwardsShadow = pair.second.primalInitialize;
        if (auto inst = dyn_cast<Instruction>(pair.first))
          if (!forwardsShadow && pair.second.LI &&
              pair.second.LI->contains(inst->getParent()))
            backwardsShadow = false;
      }
    }

    bool forceErase =
        !((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeCombined && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
          (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow));

    if (forceErase)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    else
      eraseIfUnused(call);

    Value *orig_op0 = call.getArgOperand(0);

    // If constant destination then no operation needs doing
    if (gutils->isConstantValue(orig_op0)) {
      return true;
    }

    if (!forceErase) {
      Value *op0 = gutils->invertPointerM(orig_op0, BuilderZ);
      Value *op1 = gutils->getNewFromOriginal(call.getArgOperand(1));
      Value *op2 = gutils->getNewFromOriginal(call.getArgOperand(2));
      auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::Shadow, ValueType::Primal, ValueType::Primal},
          BuilderZ, /*lookup*/ false);

      applyChainRule(
          BuilderZ,
          [&](Value *op0) {
            SmallVector<Value *, 4> args = {op0, op1, op2};
            auto cal =
                BuilderZ.CreateCall(call.getCalledFunction(), args, Defs);
            llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
            ToCopy2.push_back(LLVMContext::MD_noalias);
            cal->copyMetadata(call, ToCopy2);
            cal->setAttributes(call.getAttributes());
            if (auto m = hasMetadata(&call, "enzyme_zerostack"))
              cal->setMetadata("enzyme_zerostack", m);
            cal->setCallingConv(call.getCallingConv());
            cal->setTailCallKind(call.getTailCallKind());
            cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
          },
          op0);
    }
    return true;
  }
  if (funcName == "cuStreamCreate") {
    Value *val = nullptr;
    llvm::Type *PT = getInt8PtrTy(call.getContext());
#if LLVM_VERSION_MAJOR < 17
    if (call.getContext().supportsTypedPointers()) {
      if (isa<PointerType>(call.getArgOperand(0)->getType()))
        PT = call.getArgOperand(0)->getType()->getPointerElementType();
    }
#endif
    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      val = gutils->getNewFromOriginal(call.getOperand(0));
      if (!isa<PointerType>(val->getType()))
        val = BuilderZ.CreateIntToPtr(val, getUnqual(PT));
      val = BuilderZ.CreateLoad(PT, val);
      val = gutils->cacheForReverse(BuilderZ, val,
                                    getIndex(&call, CacheType::Tape, BuilderZ));

    } else if (Mode == DerivativeMode::ReverseModeGradient) {
      PHINode *toReplace =
          BuilderZ.CreatePHI(PT, 1, call.getName() + "_psxtmp");
      val = gutils->cacheForReverse(BuilderZ, toReplace,
                                    getIndex(&call, CacheType::Tape, BuilderZ));
    }
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      val = gutils->lookupM(val, Builder2);
      auto FreeFunc = getOrInsertPerCallingConv(
          *gutils->newFunc->getParent(), called, "cuStreamDestroy",
          FunctionType::get(call.getType(), {PT}, false));
      Value *nargs[] = {val};
      Builder2.CreateCall(FreeFunc, nargs);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }
  if (funcName == "cuStreamDestroy") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }
  if (funcName == "cuStreamSynchronize") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      Value *nargs[] = {gutils->lookupM(
          gutils->getNewFromOriginal(call.getOperand(0)), Builder2)};
      auto callval = call.getCalledOperand();
      Builder2.CreateCall(call.getFunctionType(), callval, nargs);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }
  if (funcName == "posix_memalign" || funcName == "cuMemAllocAsync" ||
      funcName == "cuMemAlloc" || funcName == "cuMemAlloc_v2" ||
      funcName == "cudaMalloc" || funcName == "cudaMallocAsync" ||
      funcName == "cudaMallocHost" || funcName == "cudaMallocFromPoolAsync") {
    bool constval = gutils->isConstantInstruction(&call);

    Value *val;
    llvm::Type *PT = getInt8PtrTy(call.getContext());
#if LLVM_VERSION_MAJOR < 17
    if (call.getContext().supportsTypedPointers()) {
      if (isa<PointerType>(call.getArgOperand(0)->getType()))
        PT = call.getArgOperand(0)->getType()->getPointerElementType();
    }
#endif
    if (!constval) {
      Value *stream = nullptr;
      if (funcName == "cuMemAllocAsync")
        stream = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocAsync")
        stream = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocFromPoolAsync")
        stream = gutils->getNewFromOriginal(call.getArgOperand(3));

      auto M = gutils->newFunc->getParent();

      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        Value *ptrshadow =
            gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
        SmallVector<Value *, 1> args;
        SmallVector<ValueType, 1> valtys;
        args.push_back(ptrshadow);
        valtys.push_back(ValueType::Shadow);
        for (size_t i = 1; i < call.arg_size(); ++i) {
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
          valtys.push_back(ValueType::Primal);
        }

        auto Defs = gutils->getInvertedBundles(&call, valtys, BuilderZ,
                                               /*lookup*/ false);

        val = applyChainRule(
            PT, BuilderZ,
            [&](Value *ptrshadow) {
              args[0] = ptrshadow;

              BuilderZ.CreateCall(called, args, Defs);
              if (!isa<PointerType>(ptrshadow->getType()))
                ptrshadow = BuilderZ.CreateIntToPtr(ptrshadow, getUnqual(PT));
              Value *val = BuilderZ.CreateLoad(PT, ptrshadow);

              auto dst_arg =
                  BuilderZ.CreateBitCast(val, getInt8PtrTy(call.getContext()));

              auto val_arg =
                  ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
              auto len_arg = gutils->getNewFromOriginal(
                  call.getArgOperand((funcName == "posix_memalign") ? 2 : 1));

              if (funcName == "posix_memalign" ||
                  funcName == "cudaMallocHost") {
                BuilderZ.CreateMemSet(dst_arg, val_arg, len_arg, MaybeAlign());
              } else if (funcName == "cudaMalloc") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called, "cudaMemset",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cudaMallocAsync" ||
                         funcName == "cudaMallocFromPoolAsync") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                               stream->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called, "cudaMemsetAsync",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cuMemAllocAsync") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                               stream->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called, "cuMemsetD8Async",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cuMemAlloc" ||
                         funcName == "cuMemAlloc_v2") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called,
                    funcName == "cuMemAlloc_v2" ? "cuMemsetD8_v2"
                                                : "cuMemsetD8",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else {
                llvm_unreachable("unhandled allocation");
              }
              return val;
            },
            ptrshadow);

        if (Mode != DerivativeMode::ForwardMode &&
            Mode != DerivativeMode::ForwardModeError)
          val = gutils->cacheForReverse(
              BuilderZ, val, getIndex(&call, CacheType::Tape, BuilderZ));
      } else if (Mode == DerivativeMode::ReverseModeGradient) {
        PHINode *toReplace = BuilderZ.CreatePHI(gutils->getShadowType(PT), 1,
                                                call.getName() + "_psxtmp");
        val = gutils->cacheForReverse(
            BuilderZ, toReplace, getIndex(&call, CacheType::Tape, BuilderZ));
      }

      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModeGradient) {
        if (shouldFree()) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          Value *tofree = gutils->lookupM(val, Builder2, ValueToValueMapTy(),
                                          /*tryLegalRecompute*/ false);

          Type *VoidTy = Type::getVoidTy(M->getContext());
          Type *IntPtrTy = getInt8PtrTy(M->getContext());

          Value *streamL = nullptr;
          if (stream)
            streamL = gutils->lookupM(stream, Builder2);

          applyChainRule(
              BuilderZ,
              [&](Value *tofree) {
                if (funcName == "posix_memalign") {
                  auto FreeFunc =
                      M->getOrInsertFunction("free", VoidTy, IntPtrTy);
                  Builder2.CreateCall(FreeFunc, tofree);
                } else if (funcName == "cuMemAllocAsync") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cuMemFreeAsync",
                      FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()},
                                        false));
                  Value *nargs[] = {tofree, streamL};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cuMemAlloc" ||
                           funcName == "cuMemAlloc_v2") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cuMemFree",
                      FunctionType::get(VoidTy, {IntPtrTy}, false));
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMalloc") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cudaFree",
                      FunctionType::get(VoidTy, {IntPtrTy}, false));
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMallocAsync" ||
                           funcName == "cudaMallocFromPoolAsync") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cudaFreeAsync",
                      FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()},
                                        false));
                  Value *nargs[] = {tofree, streamL};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMallocHost") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cudaFreeHost",
                      FunctionType::get(VoidTy, {IntPtrTy}, false));
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else
                  llvm_unreachable("unknown function to free");
              },
              tofree);
        }
      }
    }

    // TODO enable this if we need to free the memory
    // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
    // TO FREE'ing
    if (Mode == DerivativeMode::ReverseModeGradient) {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    } else if (Mode == DerivativeMode::ReverseModePrimal) {
      // if (is_value_needed_in_reverse<Primal>(
      //        TR, gutils, orig, /*topLevel*/ Mode ==
      //        DerivativeMode::Both))
      //        {

      //  gutils->cacheForReverse(BuilderZ, newCall,
      //                          getIndex(orig, CacheType::Self, BuilderZ));
      //} else if (Mode != DerivativeMode::Forward) {
      // Note that here we cannot simply replace with null as users who try
      // to find the shadow pointer will use the shadow of null rather than
      // the true shadow of this
      //}
    } else if (Mode == DerivativeMode::ReverseModeCombined && shouldFree()) {
      IRBuilder<> Builder2(newCall->getNextNode());
      auto ptrv = gutils->getNewFromOriginal(call.getOperand(0));
      if (!isa<PointerType>(ptrv->getType()))
        ptrv = BuilderZ.CreateIntToPtr(ptrv, getUnqual(PT));
      auto load = Builder2.CreateLoad(PT, ptrv, "posix_preread");
      Builder2.SetInsertPoint(&call);
      getReverseBuilder(Builder2);
      auto tofree = gutils->lookupM(load, Builder2, ValueToValueMapTy(),
                                    /*tryLegal*/ false);
      Value *streamL = nullptr;
      if (funcName == "cuMemAllocAsync")
        streamL = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocAsync")
        streamL = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocFromPoolAsync")
        streamL = gutils->getNewFromOriginal(call.getArgOperand(3));
      if (streamL)
        streamL = gutils->lookupM(streamL, Builder2);

      auto M = gutils->newFunc->getParent();
      Type *VoidTy = Type::getVoidTy(M->getContext());
      Type *IntPtrTy = getInt8PtrTy(M->getContext());

      if (funcName == "posix_memalign") {
        auto FreeFunc = M->getOrInsertFunction("free", VoidTy, IntPtrTy);
        Builder2.CreateCall(FreeFunc, tofree);
      } else if (funcName == "cuMemAllocAsync") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cuMemFreeAsync",
            FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()}, false));
        Value *nargs[] = {tofree, streamL};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cuMemAlloc" || funcName == "cuMemAlloc_v2") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cuMemFree",
            FunctionType::get(VoidTy, {IntPtrTy}, false));
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMalloc") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cudaFree",
            FunctionType::get(VoidTy, {IntPtrTy}, false));
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMallocAsync" ||
                 funcName == "cudaMallocFromPoolAsync") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cudaFreeAsync",
            FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()}, false));
        Value *nargs[] = {tofree, streamL};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMallocHost") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cudaFreeHost",
            FunctionType::get(VoidTy, {IntPtrTy}, false));
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else
        llvm_unreachable("unknown function to free");
    }

    return true;
  }

  // Remove free's in forward pass so the memory can be used in the reverse
  // pass
  if (isDeallocationFunction(funcName, gutils->TLI)) {
    assert(gutils->invertedPointers.find(&call) ==
           gutils->invertedPointers.end());

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      if (!gutils->isConstantValue(call.getArgOperand(0))) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        auto origfree = call.getArgOperand(0);
        auto newfree = gutils->getNewFromOriginal(call.getArgOperand(0));
        auto tofree = gutils->invertPointerM(origfree, Builder2);

        Function *free = getOrInsertCheckedFree(
            *call.getModule(), &call, newfree->getType(), gutils->getWidth());

        bool used = true;
        if (auto instArg = dyn_cast<Instruction>(call.getArgOperand(0)))
          used = unnecessaryInstructions.find(instArg) ==
                 unnecessaryInstructions.end();

        SmallVector<Value *, 3> args;
        if (used)
          args.push_back(newfree);
        else
          args.push_back(
              Constant::getNullValue(call.getArgOperand(0)->getType()));

        auto rule = [&args](Value *tofree) { args.push_back(tofree); };
        applyChainRule(Builder2, rule, tofree);

        for (size_t i = 1; i < call.arg_size(); i++) {
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
        }

        auto frees = Builder2.CreateCall(free->getFunctionType(), free, args);
        frees->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));

        eraseIfUnused(call);
        return true;
      }
      eraseIfUnused(call);
    }
    auto callval = call.getCalledOperand();

    for (auto rmat : gutils->backwardsOnlyShadows) {
      if (gutils->allocationsToBeRematerialized.count(rmat.first) &&
          rmat.second.frees.count(&call)) {
        bool shouldFree = false;
        if (rmat.second.primalInitialize) {
          if (Mode == DerivativeMode::ReverseModePrimal)
            shouldFree = true;
        }

        if (shouldFree) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          auto origfree = call.getArgOperand(0);
          auto tofree = gutils->invertPointerM(origfree, Builder2);
          if (tofree != origfree) {
            SmallVector<Value *, 2> args = {tofree};
            CallInst *CI =
                Builder2.CreateCall(call.getFunctionType(), callval, args);
            CI->setAttributes(call.getAttributes());
          }
        }
        break;
      }
    }

    // If a rematerializable allocation.
    for (auto rmat : gutils->rematerializableAllocations) {
      if (gutils->allocationsToBeRematerialized.count(rmat.first) &&
          rmat.second.frees.count(&call)) {
        // Leave the original free behavior since this won't be used
        // in the reverse pass in split mode
        if (Mode == DerivativeMode::ReverseModePrimal) {
          eraseIfUnused(call);
          return true;
        } else if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return true;
        } else {
          assert(Mode == DerivativeMode::ReverseModeCombined);
          std::map<UsageKey, bool> Seen =
              gutils->populateSeenFromKnownRecompute();
          bool primalNeededInReverse =
              DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, rmat.first, Mode, Seen,
                                     oldUnreachable);
          {
            auto found = gutils->knownRecomputeHeuristic.find(rmat.first);
            if (found != gutils->knownRecomputeHeuristic.end()) {
              if (!found->second) {
                primalNeededInReverse = true;
              }
            }
          }
          bool cacheWholeAllocation =
              gutils->needsCacheWholeAllocation(rmat.first);
          if (cacheWholeAllocation) {
            primalNeededInReverse = true;
          }
          // If in a loop context, maintain the same free behavior, unless
          // caching whole allocation.
          if (!cacheWholeAllocation) {
            eraseIfUnused(call);
            return true;
          }
          assert(!unnecessaryValues.count(rmat.first));
          (void)primalNeededInReverse;
          assert(primalNeededInReverse);
        }
      }
    }

    if (gutils->forwardDeallocations.count(&call)) {
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else
        eraseIfUnused(call);
      return true;
    }

    if (gutils->postDominatingFrees.count(&call)) {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return true;
    }

    if (call.getMetadata("enzyme_cache_free")) {
      bool hasGuaranteedFree = false;
      for (const auto &pair : gutils->allocationsWithGuaranteedFree) {
        if (pair.second.count(&call)) {
          hasGuaranteedFree = true;
          break;
        }
      }
      if (!hasGuaranteedFree) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
    }

    llvm::Value *val = getBaseObject(call.getArgOperand(0));
    if (isa<ConstantPointerNull>(val)) {
      llvm::errs() << "removing free of null pointer\n";
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return true;
    }

    // TODO HANDLE FREE
    llvm::errs() << "freeing without malloc " << *val << " in " << call << "\n";
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }

  if (call.hasFnAttr("enzyme_sample")) {
    if (Mode != DerivativeMode::ReverseModeCombined &&
        Mode != DerivativeMode::ReverseModeGradient)
      return true;

    bool constval = gutils->isConstantInstruction(&call);

    if (constval)
      return true;

    IRBuilder<> Builder2(&call);
    getReverseBuilder(Builder2);

    auto trace = call.getArgOperand(call.arg_size() - 1);
    auto address = call.getArgOperand(0);

    auto dtrace = lookup(gutils->getNewFromOriginal(trace), Builder2);
    auto daddress = lookup(gutils->getNewFromOriginal(address), Builder2);

    Value *dchoice;
    if (TR.query(&call)[{-1}].isPossiblePointer()) {
      dchoice = gutils->invertPointerM(&call, Builder2);
    } else {
      dchoice = diffe(&call, Builder2);
    }

    if (call.hasMetadata("enzyme_gradient_setter")) {
      auto gradient_setter = cast<Function>(
          cast<ValueAsMetadata>(
              call.getMetadata("enzyme_gradient_setter")->getOperand(0).get())
              ->getValue());

      TraceUtils::InsertChoiceGradient(
          Builder2, gradient_setter->getFunctionType(), gradient_setter,
          daddress, dchoice, dtrace);
    }

    return true;
  }

  if (call.hasFnAttr("enzyme_insert_argument")) {
    IRBuilder<> Builder2(&call);
    getReverseBuilder(Builder2);

    auto name = call.getArgOperand(0);
    auto arg = call.getArgOperand(1);
    auto trace = call.getArgOperand(2);

    auto gradient_setter = cast<Function>(
        cast<ValueAsMetadata>(
            call.getMetadata("enzyme_gradient_setter")->getOperand(0).get())
            ->getValue());

    auto dtrace = lookup(gutils->getNewFromOriginal(trace), Builder2);
    auto dname = lookup(gutils->getNewFromOriginal(name), Builder2);
    Value *darg;

    if (TR.query(arg)[{-1}].isPossiblePointer()) {
      darg = gutils->invertPointerM(arg, Builder2);
    } else {
      darg = diffe(arg, Builder2);
    }

    TraceUtils::InsertArgumentGradient(Builder2,
                                       gradient_setter->getFunctionType(),
                                       gradient_setter, dname, darg, dtrace);
    return true;
  }

  return false;
}
