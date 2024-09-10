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

using namespace llvm;

extern "C" {
void (*EnzymeShadowAllocRewrite)(LLVMValueRef, void *) = nullptr;
}

void AdjointGenerator::handleMPI(llvm::CallInst &call, llvm::Function *called,
                                 llvm::StringRef funcName) {
  using namespace llvm;

  assert(called);
  assert(gutils->getWidth() == 1);

  IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
  BuilderZ.setFastMathFlags(getFast());

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
              d_req, PointerType::getUnqual(getInt8PtrTy(call.getContext())));
        }

        auto i64 = Type::getInt64Ty(call.getContext());
        auto impi = getMPIHelper(call.getContext());

        Value *impialloc =
            CreateAllocation(BuilderZ, impi, ConstantInt::get(i64, 1));
        BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));

        d_req = BuilderZ.CreateBitCast(
            d_req, PointerType::getUnqual(impialloc->getType()));
        Value *d_req_prev = BuilderZ.CreateLoad(impialloc->getType(), d_req);
        BuilderZ.CreateStore(
            BuilderZ.CreatePointerCast(d_req_prev,
                                       getInt8PtrTy(call.getContext())),
            getMPIMemberPtr<MPI_Elem::Old>(BuilderZ, impialloc, impi));
        BuilderZ.CreateStore(impialloc, d_req);

        if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
          Value *tysize =
              MPI_TYPE_SIZE(gutils->getNewFromOriginal(call.getOperand(2)),
                            BuilderZ, call.getType());

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
        if (Function *recvfn = called->getParent()->getFunction("PMPI_Wait")) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        }
        if (Function *recvfn = called->getParent()->getFunction("MPI_Wait")) {
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
        Type *helperTy = llvm::PointerType::getUnqual(impi);
        Value *helper =
            Builder2.CreatePointerCast(d_req, PointerType::getUnqual(helperTy));
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

        Builder2.CreateStore(
            prev, Builder2.CreatePointerCast(
                      d_req, PointerType::getUnqual(prev->getType())));

        assert(shouldFree());

        assert(tysize);
        tysize = MPI_TYPE_SIZE(tysize, Builder2, call.getType());

        Value *args[] = {/*req*/ req,
                         /*status*/ IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(statusType)};
        FunctionCallee waitFunc = nullptr;
        for (auto name : {"PMPI_Wait", "MPI_Wait"})
          if (Function *recvfn = called->getParent()->getFunction(name)) {
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
          waitFunc = called->getParent()->getOrInsertFunction("MPI_Wait", FT);
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
              Intrinsic::getDeclaration(called->getParent(), Intrinsic::memset,
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

        Value *buf = gutils->invertPointerM(call.getOperand(0), Builder2);
        Value *count = gutils->getNewFromOriginal(call.getOperand(1));
        Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
        Value *source = gutils->getNewFromOriginal(call.getOperand(3));
        Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
        Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
        Value *request = gutils->invertPointerM(call.getOperand(6), Builder2);

        Value *args[] = {
            /*buf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*source*/ source,
            /*tag*/ tag,
            /*comm*/ comm,
            /*request*/ request,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow},
            Builder2, /*lookup*/ false);

        auto callval = call.getCalledOperand();

        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
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
            req, PointerType::getUnqual(getInt8PtrTy(call.getContext())));
      }

      Value *isNull = nullptr;
      if (auto GV = gutils->newFunc->getParent()->getNamedValue(
              "ompi_request_null")) {
        Value *reql = BuilderZ.CreatePointerCast(
            req, PointerType::getUnqual(GV->getType()));
        reql = BuilderZ.CreateLoad(GV->getType(), reql);
        isNull = BuilderZ.CreateICmpEQ(reql, GV);
      }

      if (d_req->getType()->isIntegerTy()) {
        d_req = BuilderZ.CreateIntToPtr(
            d_req, PointerType::getUnqual(getInt8PtrTy(call.getContext())));
      }

      d_reqp = BuilderZ.CreateLoad(
          PointerType::getUnqual(impi),
          BuilderZ.CreatePointerCast(
              d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));
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
        d_reqp = BuilderZ.CreatePHI(PointerType::getUnqual(impi), 0);
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
          *called->getParent(), types, call.getOperand(0)->getType());

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

      Value *request = gutils->invertPointerM(call.getArgOperand(0), Builder2);
      Value *status = gutils->invertPointerM(call.getArgOperand(1), Builder2);

      if (request->getType()->isIntegerTy()) {
        request = Builder2.CreateIntToPtr(
            request, PointerType::getUnqual(getInt8PtrTy(call.getContext())));
      }

      Value *args[] = {/*request*/ request,
                       /*status*/ status};

      auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::Shadow, ValueType::Shadow}, Builder2,
          /*lookup*/ false);

      auto callval = call.getCalledOperand();

      Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
      return;
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
    Value *d_reqp = nullptr;
    auto impi = getMPIHelper(call.getContext());
    PointerType *reqType = PointerType::getUnqual(impi);
    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      Value *count = gutils->getNewFromOriginal(call.getOperand(0));
      Value *req = gutils->getNewFromOriginal(call.getOperand(1));
      Value *d_req = gutils->invertPointerM(call.getOperand(1), BuilderZ);

      if (req->getType()->isIntegerTy()) {
        req = BuilderZ.CreateIntToPtr(
            req, PointerType::getUnqual(getInt8PtrTy(call.getContext())));
      }

      if (d_req->getType()->isIntegerTy()) {
        d_req = BuilderZ.CreateIntToPtr(
            d_req, PointerType::getUnqual(getInt8PtrTy(call.getContext())));
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
        d_reqp = BuilderZ.CreatePHI(PointerType::getUnqual(reqType), 0);
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
          PointerType::getUnqual(impi),
          Builder2.CreatePointerCast(
              d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));

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
      Function *dwait = getOrInsertDifferentialMPI_Wait(*called->getParent(),
                                                        types, req->getType());
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

      assert(!gutils->isConstantValue(call.getOperand(1)));

      Value *count = gutils->getNewFromOriginal(call.getOperand(0));
      Value *array_of_requests =
          gutils->invertPointerM(call.getOperand(1), Builder2);
      if (array_of_requests->getType()->isIntegerTy()) {
        array_of_requests = Builder2.CreateIntToPtr(
            array_of_requests,
            PointerType::getUnqual(getInt8PtrTy(call.getContext())));
      }

      Value *args[] = {
          /*count*/ count,
          /*array_of_requests*/ array_of_requests,
      };

      auto Defs = gutils->getInvertedBundles(
          &call,
          {ValueType::None, ValueType::None, ValueType::None, ValueType::None,
           ValueType::None, ValueType::None, ValueType::Shadow},
          Builder2, /*lookup*/ false);

      auto callval = call.getCalledOperand();

      Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
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
        if (Function *recvfn = called->getParent()->getFunction("MPI_Recv")) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        } else if (Function *recvfn =
                       called->getParent()->getFunction("PMPI_Recv")) {
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
        Value *args[] = {
            /*buf*/ shadow,
            /*count*/ count,
            /*datatype*/ datatype,
            /*dest*/ src,
            /*tag*/ tag,
            /*comm*/ comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ false);

        auto callval = call.getCalledOperand();
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
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

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

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
          called->getParent()->getOrInsertFunction("MPI_Recv", FT), args);
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
        Mode == DerivativeMode::ReverseModeCombined) {
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

      Value *args[] = {
          shadow, count, datatype, source, tag, comm,
      };

      auto Defs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal, ValueType::Primal, ValueType::Primal,
           ValueType::None},
          Builder2, /*lookup*/ !forwardMode);

      if (forwardMode) {
        auto callval = call.getCalledOperand();

        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
        return;
      }

      Type *types[sizeof(args) / sizeof(*args)];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
        types[i] = args[i]->getType();
      FunctionType *FT = FunctionType::get(call.getType(), types, false);

      auto fcall = Builder2.CreateCall(
          called->getParent()->getOrInsertFunction("MPI_Send", FT), args, Defs);
      fcall->setCallingConv(call.getCallingConv());

      auto dst_arg =
          Builder2.CreateBitCast(args[0], getInt8PtrTy(call.getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto len_arg = Builder2.CreateZExtOrTrunc(
          args[1], Type::getInt64Ty(call.getContext()));
      auto tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());
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
          Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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
  if (funcName == "MPI_Bcast") {
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

      ConcreteType CT = TR.firstPointer(1, call.getOperand(0), &call);
      auto MPI_OP_type = getInt8PtrTy(call.getContext());
      Type *MPI_OP_Ptr_type = PointerType::getUnqual(MPI_OP_type);

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
        Value *args[] = {
            /*buffer*/ shadow,
            /*count*/ count,
            /*datatype*/ datatype,
            /*root*/ root,
            /*comm*/ comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ false);

        auto callval = call.getCalledOperand();
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
        return;
      }

      Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

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
            getOrInsertOpFloatSum(*gutils->newFunc->getParent(),
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
            called->getParent()->getOrInsertFunction("MPI_Reduce", FT), args,
            BufferDefs);
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

        auto memcpyF = Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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
          Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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

  if (funcName == "MPI_Reduce" || funcName == "PMPI_Reduce") {
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

      // Get the operations from MPI_Receive
      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_recvbuf = call.getOperand(1);
      Value *orig_count = call.getOperand(2);
      Value *orig_datatype = call.getOperand(3);
      Value *orig_op = call.getOperand(4);
      Value *orig_root = call.getOperand(5);
      Value *orig_comm = call.getOperand(6);

      bool isSum = false;
      if (Constant *C = dyn_cast<Constant>(orig_op)) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_op_sum") {
            isSum = true;
          }
        }
        // MPICH
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 1476395011) {
            isSum = true;
          }
        }
      }
      if (!isSum) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << " call: " << call << "\n";
        ss << " unhandled mpi_reduce op: " << *orig_op << "\n";
        EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
        return;
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

      // Need to preserve the shadow send/recv buffers.
      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
           ValueType::Primal, ValueType::Primal, ValueType::Primal,
           ValueType::Primal},
          Builder2, /*lookup*/ !forwardMode);

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

      Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());

      if (forwardMode) {
        Value *args[] = {
            /*sendbuf*/ shadow_sendbuf,
            /*recvbuf*/ shadow_recvbuf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*op*/ op,
            /*root*/ root,
            /*comm*/ comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal,
             ValueType::Primal},
            Builder2, /*lookup*/ false);

        auto callval = call.getCalledOperand();
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
        return;
      }

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

      // Get the length for the allocation of the intermediate buffer
      auto len_arg = Builder2.CreateZExtOrTrunc(
          count, Type::getInt64Ty(call.getContext()));
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

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        {
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *nargs[] = {buf, shadow_recvbuf, len_arg, volatile_arg};

          Type *tys[] = {nargs[0]->getType(), nargs[1]->getType(),
                         len_arg->getType()};

          auto memcpyF = Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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
        Value *args[] = {
            /*buf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*int root*/ root,
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Bcast", FT), args,
            BufferDefs);
      }

      // 3. if root, Zero diff(recvbuffer) [memset to 0]
      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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

  if (funcName == "MPI_Allreduce") {
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

      // Get the operations from MPI_Receive
      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_recvbuf = call.getOperand(1);
      Value *orig_count = call.getOperand(2);
      Value *orig_datatype = call.getOperand(3);
      Value *orig_op = call.getOperand(4);
      Value *orig_comm = call.getOperand(5);

      bool isSum = false;
      if (Constant *C = dyn_cast<Constant>(orig_op)) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_op_sum") {
            isSum = true;
          }
        }
        // MPICH
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 1476395011) {
            isSum = true;
          }
        }
      }
      if (!isSum) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << " call: " << call << "\n";
        ss << " unhandled mpi_allreduce op: " << *orig_op << "\n";
        EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
        return;
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

      // Need to preserve the shadow send/recv buffers.
      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
           ValueType::Primal, ValueType::Primal, ValueType::Primal},
          Builder2, /*lookup*/ !forwardMode);

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

      if (forwardMode) {
        Value *args[] = {
            /*sendbuf*/ shadow_sendbuf,
            /*recvbuf*/ shadow_recvbuf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*op*/ op,
            /*comm*/ comm,
        };

        auto callval = call.getCalledOperand();
        Builder2.CreateCall(call.getFunctionType(), callval, args, BufferDefs);

        return;
      }

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

      // Get the length for the allocation of the intermediate buffer
      auto len_arg = Builder2.CreateZExtOrTrunc(
          count, Type::getInt64Ty(call.getContext()));
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
        Value *args[] = {
            /*sendbuf*/ shadow_recvbuf,
            /*recvbuf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*op*/ op,
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Allreduce", FT), args,
            BufferDefs);
      }

      // 3. Zero diff(recvbuffer) [memset to 0]
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto volatile_arg = ConstantInt::getFalse(call.getContext());
      Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
      Type *tys[] = {args[0]->getType(), args[2]->getType()};
      auto memset = cast<CallInst>(Builder2.CreateCall(
          Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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

  if (funcName == "MPI_Gather") {
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

      Value *root = gutils->getNewFromOriginal(orig_root);
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

      if (forwardMode) {
        Value *args[] = {
            /*sendbuf*/ shadow_sendbuf,
            /*sendcount*/ sendcount,
            /*sendtype*/ sendtype,
            /*recvbuf*/ shadow_recvbuf,
            /*recvcount*/ recvcount,
            /*recvtype*/ recvtype,
            /*root*/ root,
            /*comm*/ comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ false);

        auto callval = call.getCalledOperand();
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
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
           ValueType::Primal, ValueType::Primal},
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
        Value *args[] = {
            /*sendbuf*/ shadow_recvbuf,
            /*sendcount*/ recvcount,
            /*sendtype*/ recvtype,
            /*recvbuf*/ buf,
            /*recvcount*/ sendcount,
            /*recvtype*/ sendtype,
            /*op*/ root,
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Scatter", FT), args,
            BufferDefs);
      }

      // 3. if root, Zero diff(recvbuffer) [memset to 0]
      {

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);
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
                MPI_COMM_SIZE(comm, Builder2, root->getType()),
                Type::getInt64Ty(call.getContext())),
            "", true, true);

        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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
  if (funcName == "MPI_Scatter") {
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

      Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

      if (forwardMode) {
        Value *args[] = {
            /*sendbuf*/ shadow_sendbuf,
            /*sendcount*/ sendcount,
            /*sendtype*/ sendtype,
            /*recvbuf*/ shadow_recvbuf,
            /*recvcount*/ recvcount,
            /*recvtype*/ recvtype,
            /*root*/ root,
            /*comm*/ comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ false);

        auto callval = call.getCalledOperand();
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
        return;
      }
      // Get the length for the allocation of the intermediate buffer
      auto recvlen_arg = Builder2.CreateZExtOrTrunc(
          recvcount, Type::getInt64Ty(call.getContext()));
      recvlen_arg =
          Builder2.CreateMul(recvlen_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // Need to preserve the shadow send/recv buffers.
      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal, ValueType::Primal},
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

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcount, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        sendlen_arg = Builder2.CreateMul(
            sendlen_arg,
            Builder2.CreateZExtOrTrunc(
                MPI_COMM_SIZE(comm, Builder2, root->getType()),
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
        Value *args[] = {
            /*sendbuf*/ shadow_recvbuf,
            /*sendcount*/ recvcount,
            /*sendtype*/ recvtype,
            /*recvbuf*/ buf,
            /*recvcount*/ sendcount,
            /*recvtype*/ sendtype,
            /*root*/ root,
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Gather", FT), args,
            BufferDefs);
      }

      // 3. Zero diff(recvbuffer) [memset to 0]
      {
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
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

  if (funcName == "MPI_Allgather") {
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

      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

      if (forwardMode) {
        Value *args[] = {
            /*sendbuf*/ shadow_sendbuf,
            /*sendcount*/ sendcount,
            /*sendtype*/ sendtype,
            /*recvbuf*/ shadow_recvbuf,
            /*recvcount*/ recvcount,
            /*recvtype*/ recvtype,
            /*comm*/ comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal},
            Builder2, /*lookup*/ false);

        auto callval = call.getCalledOperand();
        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
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

      ConcreteType CT = TR.firstPointer(1, orig_sendbuf, &call);
      auto MPI_OP_type = getInt8PtrTy(call.getContext());
      Type *MPI_OP_Ptr_type = PointerType::getUnqual(MPI_OP_type);

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
            getOrInsertOpFloatSum(*gutils->newFunc->getParent(),
                                  MPI_OP_Ptr_type, MPI_OP_type, CT,
                                  call.getType(), Builder2),
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(called->getParent()->getOrInsertFunction(
                                "MPI_Reduce_scatter_block", FT),
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
                MPI_COMM_SIZE(comm, Builder2, call.getType()),
                Type::getInt64Ty(call.getContext())),
            "", true, true);
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
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
  if (funcName == "MPI_Barrier") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      auto callval = call.getCalledOperand();
      Value *args[] = {
          lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2)};
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
          called->getParent()->getOrInsertFunction("MPI_Comm_free", FT), args);
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

bool AdjointGenerator::handleKnownCallDerivatives(
    CallInst &call, Function *called, StringRef funcName,
    const std::vector<bool> &overwritten_args, CallInst *const newCall) {
  bool subretused = false;
  bool shadowReturnUsed = false;
  DIFFE_TYPE subretType =
      gutils->getReturnDiffeType(&call, &subretused, &shadowReturnUsed);

  IRBuilder<> BuilderZ(newCall);
  BuilderZ.setFastMathFlags(getFast());

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

  if ((startsWith(funcName, "MPI_") || startsWith(funcName, "PMPI_")) &&
      (!gutils->isConstantInstruction(&call) || funcName == "MPI_Barrier" ||
       funcName == "MPI_Comm_free" || funcName == "MPI_Comm_disconnect" ||
       MPIInactiveCommAllocators.find(funcName) !=
           MPIInactiveCommAllocators.end())) {
    handleMPI(call, called, funcName);
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

    if (startsWith(funcName, "__kmpc") &&
        funcName != "__kmpc_global_thread_num") {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << call << "\n";
      assert(0 && "unhandled openmp function");
      llvm_unreachable("unhandled openmp function");
    }

    auto mod = call.getParent()->getParent()->getParent();
#include "CallDerivatives.inc"

    if (funcName == "llvm.julia.gc_preserve_end") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {

        auto begin_call = cast<CallInst>(call.getOperand(0));

        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        SmallVector<Value *, 1> args;
        for (auto &arg : begin_call->args())
        {
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
      for (auto &arg : call.args())
      {
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
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                "llvm.julia.gc_preserve_end",
                FunctionType::get(Builder2.getVoidTy(), call.getType(), false)),
            placeholder);
      }
      return true;
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
        auto F = called->getParent()->getOrInsertFunction(
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
        auto FS = called->getParent()->getOrInsertFunction(
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
                PointerType::getUnqual(types[2])),
            idxs);

        auto acc = Builder2.CreateFAdd(
            acc_idx,
            Builder2.CreateFMul(Builder2.CreateLoad(types[2], dtmp_idx),
                                Builder2.CreateLoad(types[2], d_req)));
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
      for (auto &arg : call.args())
      {
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
              for (auto &arg : call.args())
              {
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
        std::map<UsageKey, bool> Seen;
        for (auto pair : gutils->knownRecomputeHeuristic)
          Seen[UsageKey(pair.first, QueryType::Primal)] = false;
        bool primalNeededInReverse =
            DifferentialUseAnalysis::is_value_needed_in_reverse<
                QueryType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);
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

        std::map<UsageKey, bool> Seen;
        for (auto pair : gutils->knownRecomputeHeuristic)
          if (!pair.second)
            Seen[UsageKey(pair.first, QueryType::Primal)] = false;

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
          SmallVector<Value *, 1> iargs;
          IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
          for (auto &arg : call.args())
          {
            if (!gutils->isConstantValue(arg)) {
              Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
              applyChainRule(
                  BuilderZ,
                  [&](Value *ptrshadow) { iargs.push_back(ptrshadow); },
                  ptrshadow);
            }
          }
          if (iargs.size()) {
            BuilderZ.CreateCall(called, iargs);
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
            bool primalNeededInReverse =
                Mode == DerivativeMode::ForwardMode ||
                        Mode == DerivativeMode::ForwardModeError
                    ? false
                    : DifferentialUseAnalysis::is_value_needed_in_reverse<
                          QueryType::Primal>(gutils, pair.first, Mode, Seen,
                                             oldUnreachable);

            bool cacheWholeAllocation =
                gutils->needsCacheWholeAllocation(pair.first);
            if (cacheWholeAllocation) {
              primalNeededInReverse = true;
            }

            if (primalNeededInReverse && !cacheWholeAllocation)
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
      for (auto &arg : call.args())
      {
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
                  if (EnzymeShadowAllocRewrite)
                    EnzymeShadowAllocRewrite(wrap(anti), gutils);
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
                std::string name = "";
#if LLVM_VERSION_MAJOR < 17
                if (call.getContext().supportsTypedPointers()) {
                  for (auto U : call.users()) {
                    if (hasMetadata(cast<Instruction>(U), "enzyme_caststack")) {
                      elTy = U->getType()->getPointerElementType();
                      Value *tsize = ConstantInt::get(
                          Size->getType(), (gutils->newFunc->getParent()
                                                ->getDataLayout()
                                                .getTypeAllocSizeInBits(elTy) +
                                            7) /
                                               8);
                      Size = bb.CreateUDiv(Size, tsize, "", /*exact*/ true);
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
                          PointerType::getUnqual(
                              anti->getType()->getPointerElementType()));
                  }
#endif
                  if (int AS = cast<PointerType>(anti->getType())
                                   ->getAddressSpace()) {
                    llvm::PointerType *PT;
#if LLVM_VERSION_MAJOR < 17
                    if (call.getContext().supportsTypedPointers()) {
                      PT = PointerType::get(
                          anti->getType()->getPointerElementType(), AS);
#endif
#if LLVM_VERSION_MAJOR < 17
                    } else {
#endif
                      PT = PointerType::get(anti->getContext(), AS);
#if LLVM_VERSION_MAJOR < 17
                    }
#endif
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
          auto rule = [&](Value *tofree) {
            auto CI = freeKnownAllocation(Builder2, tofree, funcName, dbgLoc,
                                          gutils->TLI, &call, gutils);
            if (CI)
              CI->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                      Attribute::NonNull);
          };
          applyChainRule(Builder2, rule, tofree);
        }
      } else if (Mode == DerivativeMode::ForwardMode ||
                 Mode == DerivativeMode::ForwardModeError) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        SmallVector<Value *, 2> args;
        for (unsigned i = 0; i < call.arg_size(); ++i)
        {
          auto arg = call.getArgOperand(i);
          args.push_back(gutils->getNewFromOriginal(arg));
        }

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

    std::map<UsageKey, bool> Seen;
    for (auto pair : gutils->knownRecomputeHeuristic)
      if (!pair.second)
        Seen[UsageKey(pair.first, QueryType::Primal)] = false;
    bool primalNeededInReverse =
        Mode == DerivativeMode::ForwardMode ||
                Mode == DerivativeMode::ForwardModeError
            ? false
            : DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);

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
      Instruction *I = nullptr;
#if LLVM_VERSION_MAJOR < 17
      if (call.getContext().supportsTypedPointers()) {
        for (auto U : call.users()) {
          if (hasMetadata(cast<Instruction>(U), "enzyme_caststack")) {
            elTy = U->getType()->getPointerElementType();
            Value *tsize = ConstantInt::get(Size->getType(),
                                            (gutils->newFunc->getParent()
                                                 ->getDataLayout()
                                                 .getTypeAllocSizeInBits(elTy) +
                                             7) /
                                                8);
            Size = B.CreateUDiv(Size, tsize, "", /*exact*/ true);
            I = gutils->getNewFromOriginal(cast<Instruction>(U));
            break;
          }
        }
      }
#endif
      Value *replacement = B.CreateAlloca(elTy, Size);
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
              replacement,
              PointerType::getUnqual(call.getType()->getPointerElementType()));

      }
#endif
      if (int AS = cast<PointerType>(call.getType())->getAddressSpace()) {
        llvm::PointerType *PT;
#if LLVM_VERSION_MAJOR < 17
        if (call.getContext().supportsTypedPointers()) {
          PT = PointerType::get(call.getType()->getPointerElementType(), AS);
#endif
#if LLVM_VERSION_MAJOR < 17
        } else {
#endif
          PT = PointerType::get(call.getContext(), AS);
#if LLVM_VERSION_MAJOR < 17
        }
#endif
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
        if (primalNeededInReverse && !cacheWholeAllocation) {
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
          if (funcName == "ijl_alloc_array_1d" ||
              funcName == "ijl_alloc_array_2d" ||
              funcName == "ijl_alloc_array_3d" ||
              funcName == "ijl_array_copy" || funcName == "jl_alloc_array_1d" ||
              funcName == "jl_alloc_array_2d" ||
              funcName == "jl_alloc_array_3d" || funcName == "jl_array_copy" ||
              funcName == "julia.gc_alloc_obj" ||
              funcName == "jl_gc_alloc_typed" ||
              funcName == "ijl_gc_alloc_typed") {
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
            freeKnownAllocation(Builder2, lookup(newCall, Builder2), funcName,
                                dbgLoc, gutils->TLI, &call, gutils);
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
    if (funcName == "jl_alloc_array_1d" || funcName == "jl_alloc_array_2d" ||
        funcName == "jl_alloc_array_3d" || funcName == "jl_array_copy" ||
        funcName == "ijl_alloc_array_1d" || funcName == "ijl_alloc_array_2d" ||
        funcName == "ijl_alloc_array_3d" || funcName == "ijl_array_copy" ||
        funcName == "julia.gc_alloc_obj" || funcName == "jl_gc_alloc_typed" ||
        funcName == "ijl_gc_alloc_typed") {
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
      Value *nop = gutils->cacheForReverse(
          BuilderZ, newCall, getIndex(&call, CacheType::Self, BuilderZ));
      if (hasPDFree &&
          ((Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
           Mode == DerivativeMode::ReverseModeCombined ||
           (Mode == DerivativeMode::ForwardModeSplit && shouldFree()))) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto dbgLoc = gutils->getNewFromOriginal(call.getDebugLoc());
        freeKnownAllocation(Builder2, lookup(nop, Builder2), funcName, dbgLoc,
                            gutils->TLI, &call, gutils);
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

    Value *ptr0shadow = gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
    Value *ptr1shadow = gutils->invertPointerM(call.getArgOperand(1), BuilderZ);

    Value *val = applyChainRule(
        call.getType(), BuilderZ,
        [&](Value *v1, Value *v2) -> Value * {
          Value *args[2] = {v1, v2};
          return BuilderZ.CreateCall(called, args);
        },
        ptr0shadow, ptr1shadow);

    gutils->replaceAWithB(placeholder, val);
    gutils->erase(placeholder);
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

    SmallVector<Value *, 3> args;
    for (size_t i = 0; i < 2; i++)
      args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
    for (size_t i = 2; i < call.arg_size(); ++i)
      args.push_back(gutils->invertPointerM(call.getArgOperand(0), BuilderZ));

    Value *res = UndefValue::get(gutils->getShadowType(call.getType()));
    if (gutils->getWidth() == 1) {
      res = BuilderZ.CreateCall(called, args);
    } else {
      for (size_t w = 0; w < gutils->getWidth(); ++w) {
        SmallVector<Value *, 3> targs = {args[0], args[1]};
        for (size_t i = 2; i < call.arg_size(); ++i)
          targs.push_back(GradientUtils::extractMeta(BuilderZ, args[i], w));

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

    bool forceErase = Mode == DerivativeMode::ReverseModeGradient ||
                      Mode == DerivativeMode::ForwardModeSplit;

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
        val = BuilderZ.CreateIntToPtr(val, PointerType::getUnqual(PT));
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
      auto FreeFunc = gutils->newFunc->getParent()->getOrInsertFunction(
          "cuStreamDestroy", call.getType(), PT);
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
        for (size_t i = 1; i < call.arg_size(); ++i)
        {
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
                ptrshadow = BuilderZ.CreateIntToPtr(ptrshadow,
                                                    PointerType::getUnqual(PT));
              Value *val = BuilderZ.CreateLoad(PT, ptrshadow);

              auto dst_arg =
                  BuilderZ.CreateBitCast(val, getInt8PtrTy(call.getContext()));

              auto val_arg =
                  ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
              auto len_arg = gutils->getNewFromOriginal(
                  call.getArgOperand((funcName == "posix_memalign") ? 2 : 1));

              if (funcName == "posix_memalign" ||
                  funcName == "cudaMallocHost") {
                auto volatile_arg = ConstantInt::getFalse(call.getContext());

                Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};

                Type *tys[] = {dst_arg->getType(), len_arg->getType()};

                auto memset = cast<CallInst>(BuilderZ.CreateCall(
                    Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                              Intrinsic::memset, tys),
                    nargs));
                // memset->addParamAttr(0,
                // Attribute::getWithAlignment(Context,
                // inst->getAlignment()));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cudaMalloc") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                auto F = M->getOrInsertFunction(
                    "cudaMemset",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cudaMallocAsync" ||
                         funcName == "cudaMallocFromPoolAsync") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                               stream->getType()};
                auto F = M->getOrInsertFunction(
                    "cudaMemsetAsync",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cuMemAllocAsync") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                               stream->getType()};
                auto F = M->getOrInsertFunction(
                    "cuMemsetD8Async",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cuMemAlloc" ||
                         funcName == "cuMemAlloc_v2") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                auto F = M->getOrInsertFunction(
                    "cuMemsetD8",
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
                  auto FreeFunc = M->getOrInsertFunction(
                      "cuMemFreeAsync", VoidTy, IntPtrTy, streamL->getType());
                  Value *nargs[] = {tofree, streamL};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cuMemAlloc" ||
                           funcName == "cuMemAlloc_v2") {
                  auto FreeFunc =
                      M->getOrInsertFunction("cuMemFree", VoidTy, IntPtrTy);
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMalloc") {
                  auto FreeFunc =
                      M->getOrInsertFunction("cudaFree", VoidTy, IntPtrTy);
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMallocAsync" ||
                           funcName == "cudaMallocFromPoolAsync") {
                  auto FreeFunc = M->getOrInsertFunction(
                      "cudaFreeAsync", VoidTy, IntPtrTy, streamL->getType());
                  Value *nargs[] = {tofree, streamL};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMallocHost") {
                  auto FreeFunc =
                      M->getOrInsertFunction("cudaFreeHost", VoidTy, IntPtrTy);
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
        ptrv = BuilderZ.CreateIntToPtr(ptrv, PointerType::getUnqual(PT));
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
        auto FreeFunc = M->getOrInsertFunction("cuMemFreeAsync", VoidTy,
                                               IntPtrTy, streamL->getType());
        Value *nargs[] = {tofree, streamL};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cuMemAlloc" || funcName == "cuMemAlloc_v2") {
        auto FreeFunc = M->getOrInsertFunction("cuMemFree", VoidTy, IntPtrTy);
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMalloc") {
        auto FreeFunc = M->getOrInsertFunction("cudaFree", VoidTy, IntPtrTy);
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMallocAsync" ||
                 funcName == "cudaMallocFromPoolAsync") {
        auto FreeFunc = M->getOrInsertFunction("cudaFreeAsync", VoidTy,
                                               IntPtrTy, streamL->getType());
        Value *nargs[] = {tofree, streamL};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMallocHost") {
        auto FreeFunc =
            M->getOrInsertFunction("cudaFreeHost", VoidTy, IntPtrTy);
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

        for (size_t i = 1; i < call.arg_size(); i++)
        {
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
      if (rmat.second.frees.count(&call)) {
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
      if (rmat.second.frees.count(&call)) {
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
          std::map<UsageKey, bool> Seen;
          for (auto pair : gutils->knownRecomputeHeuristic)
            if (!pair.second)
              Seen[UsageKey(pair.first, QueryType::Primal)] = false;
          bool primalNeededInReverse =
              DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, rmat.first, Mode, Seen,
                                     oldUnreachable);
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

    llvm::Value *val = getBaseObject(call.getArgOperand(0));
    if (isa<ConstantPointerNull>(val)) {
      llvm::errs() << "removing free of null pointer\n";
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return true;
    }

    // TODO HANDLE FREE
    llvm::errs() << "freeing without malloc " << *val << "\n";
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
