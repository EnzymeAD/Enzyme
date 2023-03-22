#include "Runtime.h"
#include "Utils.h"

#include "llvm-c/Core.h"

using namespace llvm;

// MARK: Options

extern "C" {
void (*CustomErrorHandler)(const char *, LLVMValueRef, ErrorType,
                           const void *) = nullptr;
LLVMValueRef (*CustomAllocator)(LLVMBuilderRef, LLVMTypeRef,
                                /*Count*/ LLVMValueRef,
                                /*Align*/ LLVMValueRef, uint8_t,
                                LLVMValueRef *) = nullptr;
void (*CustomZero)(LLVMBuilderRef, LLVMTypeRef,
                   /*Ptr*/ LLVMValueRef, uint8_t) = nullptr;
LLVMValueRef (*CustomDeallocator)(LLVMBuilderRef, LLVMValueRef) = nullptr;
void (*CustomRuntimeInactiveError)(LLVMBuilderRef, LLVMValueRef,
                                   LLVMValueRef) = nullptr;
LLVMValueRef *(*EnzymePostCacheStore)(LLVMValueRef, LLVMBuilderRef,
                                      uint64_t *size) = nullptr;
LLVMTypeRef (*EnzymeDefaultTapeType)(LLVMContextRef) = nullptr;
}

// MARK: Helpers

/// Convert a floating type to a string
static inline std::string tofltstr(Type *T) {
  switch (T->getTypeID()) {
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::X86_FP80TyID:
    return "x87d";
  case Type::FP128TyID:
    return "quad";
  case Type::PPC_FP128TyID:
    return "ppcddouble";
  default:
    llvm_unreachable("Invalid floating type");
  }
}

Constant *getString(Module &M, StringRef Str) {
  llvm::Constant *s = llvm::ConstantDataArray::getString(M.getContext(), Str);
  auto *gv = new llvm::GlobalVariable(
      M, s->getType(), true, llvm::GlobalValue::PrivateLinkage, s, ".str");
  gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  Value *Idxs[2] = {ConstantInt::get(Type::getInt32Ty(M.getContext()), 0),
                    ConstantInt::get(Type::getInt32Ty(M.getContext()), 0)};
  return ConstantExpr::getInBoundsGetElementPtr(s->getType(), gv, Idxs);
}

// MARK: Runtime

void ZeroMemory(llvm::IRBuilder<> &Builder, llvm::Type *T, llvm::Value *obj,
                bool isTape) {
  if (CustomZero) {
    CustomZero(wrap(&Builder), wrap(T), wrap(obj), isTape);
  } else {
    Builder.CreateStore(Constant::getNullValue(T), obj);
  }
}

llvm::SmallVector<llvm::Instruction *, 2> PostCacheStore(llvm::StoreInst *SI,
                                                         llvm::IRBuilder<> &B) {
  SmallVector<llvm::Instruction *, 2> res;
  if (EnzymePostCacheStore) {
    uint64_t size = 0;
    auto ptr = EnzymePostCacheStore(wrap(SI), wrap(&B), &size);
    for (size_t i = 0; i < size; i++) {
      res.push_back(cast<Instruction>(unwrap(ptr[i])));
    }
    free(ptr);
  }
  return res;
}

llvm::PointerType *getDefaultAnonymousTapeType(llvm::LLVMContext &C) {
  if (EnzymeDefaultTapeType)
    return cast<PointerType>(unwrap(EnzymeDefaultTapeType(wrap(&C))));
  return Type::getInt8PtrTy(C);
}

Function *getOrInsertExponentialAllocator(Module &M, Function *newFunc,
                                          bool ZeroInit, llvm::Type *RT) {
  bool custom = true;
  llvm::PointerType *allocType;
  {
    auto i64 = Type::getInt64Ty(newFunc->getContext());
    BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", newFunc);
    IRBuilder<> B(BB);
    auto P = B.CreatePHI(i64, 1);
    CallInst *malloccall;
    Instruction *SubZero = nullptr;
    CreateAllocation(B, RT, P, "tapemem", &malloccall, &SubZero)->getType();
    if (auto F = getFunctionFromCall(malloccall)) {
      custom = F->getName() != "malloc";
    }
    allocType = cast<PointerType>(malloccall->getType());
    BB->eraseFromParent();
  }

  Type *types[] = {allocType, Type::getInt64Ty(M.getContext()),
                   Type::getInt64Ty(M.getContext())};
  std::string name = "__enzyme_exponentialallocation";
  if (ZeroInit)
    name += "zero";
  if (custom)
    name += ".custom@" + std::to_string((size_t)RT);

  FunctionType *FT = FunctionType::get(allocType, types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addFnAttr(Attribute::NoUnwind);
  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *grow = BasicBlock::Create(M.getContext(), "grow", F);
  BasicBlock *ok = BasicBlock::Create(M.getContext(), "ok", F);

  IRBuilder<> B(entry);

  Argument *ptr = F->arg_begin();
  ptr->setName("ptr");
  Argument *size = ptr + 1;
  size->setName("size");
  Argument *tsize = size + 1;
  tsize->setName("tsize");

  Value *hasOne = B.CreateICmpNE(
      B.CreateAnd(size, ConstantInt::get(size->getType(), 1, false)),
      ConstantInt::get(size->getType(), 0, false));
  auto popCnt = Intrinsic::getDeclaration(&M, Intrinsic::ctpop, {types[1]});

  B.CreateCondBr(
      B.CreateAnd(B.CreateICmpULT(B.CreateCall(popCnt, {size}),
                                  ConstantInt::get(types[1], 3, false)),
                  hasOne),
      grow, ok);

  B.SetInsertPoint(grow);

  auto lz =
      B.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::ctlz, {types[1]}),
                   {size, ConstantInt::getTrue(M.getContext())});
  Value *next =
      B.CreateShl(tsize, B.CreateSub(ConstantInt::get(types[1], 64, false), lz,
                                     "", true, true));

  Value *gVal;

  Value *prevSize =
      B.CreateSelect(B.CreateICmpEQ(size, ConstantInt::get(size->getType(), 1)),
                     ConstantInt::get(next->getType(), 0),
                     B.CreateLShr(next, ConstantInt::get(next->getType(), 1)));

  if (!custom) {
    auto reallocF = M.getOrInsertFunction("realloc", allocType, allocType,
                                          Type::getInt64Ty(M.getContext()));

    Value *args[] = {B.CreatePointerCast(ptr, allocType), next};
    gVal = B.CreateCall(reallocF, args);
  } else {
    Value *tsize = ConstantInt::get(
        next->getType(),
        newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(RT) / 8);
    auto elSize = B.CreateUDiv(next, tsize, "", /*isExact*/ true);
    Instruction *SubZero = nullptr;
    gVal = CreateAllocation(B, RT, elSize, "", nullptr, &SubZero);

    Type *bTy =
        PointerType::get(Type::getInt8Ty(gVal->getContext()),
                         cast<PointerType>(gVal->getType())->getAddressSpace());
    gVal = B.CreatePointerCast(gVal, bTy);
    auto pVal = B.CreatePointerCast(ptr, gVal->getType());

    Value *margs[] = {gVal, pVal, prevSize,
                      ConstantInt::getFalse(M.getContext())};
    Type *tys[] = {margs[0]->getType(), margs[1]->getType(),
                   margs[2]->getType()};
    auto memsetF = Intrinsic::getDeclaration(&M, Intrinsic::memcpy, tys);
    B.CreateCall(memsetF, margs);
    if (SubZero) {
      ZeroInit = false;
      IRBuilder<> BB(SubZero);
      Value *zeroSize = BB.CreateSub(next, prevSize);
      Value *tmp = SubZero->getOperand(0);
      Type *tmpT = tmp->getType();
      tmp = BB.CreatePointerCast(tmp, bTy);

#if LLVM_VERSION_MAJOR > 7
      tmp = BB.CreateInBoundsGEP(tmp->getType()->getPointerElementType(), tmp,
                                 prevSize);
#else
      tmp = BB.CreateInBoundsGEP(tmp, prevSize);
#endif
      tmp = BB.CreatePointerCast(tmp, tmpT);
      SubZero->setOperand(0, tmp);
      SubZero->setOperand(2, zeroSize);
    }
  }

  if (ZeroInit) {
    Value *zeroSize = B.CreateSub(next, prevSize);

    Value *margs[] = {
#if LLVM_VERSION_MAJOR > 7
      B.CreateInBoundsGEP(gVal->getType()->getPointerElementType(), gVal,
                          prevSize),
#else
      B.CreateInBoundsGEP(gVal, prevSize),
#endif
      ConstantInt::get(Type::getInt8Ty(M.getContext()), 0),
      zeroSize,
      ConstantInt::getFalse(M.getContext())
    };
    Type *tys[] = {margs[0]->getType(), margs[2]->getType()};
    auto memsetF = Intrinsic::getDeclaration(&M, Intrinsic::memset, tys);
    B.CreateCall(memsetF, margs);
  }
  gVal = B.CreatePointerCast(gVal, ptr->getType());

  B.CreateBr(ok);
  B.SetInsertPoint(ok);
  auto phi = B.CreatePHI(ptr->getType(), 2);
  phi->addIncoming(gVal, grow);
  phi->addIncoming(ptr, entry);
  B.CreateRet(phi);
  return F;
}

llvm::Value *CreateReAllocation(llvm::IRBuilder<> &B, llvm::Value *prev,
                                llvm::Type *T, llvm::Value *OuterCount,
                                llvm::Value *InnerCount, llvm::Twine Name,
                                llvm::CallInst **caller, bool ZeroMem) {
  auto newFunc = B.GetInsertBlock()->getParent();

  Value *tsize = ConstantInt::get(
      InnerCount->getType(),
      newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(T) / 8);

  Value *idxs[] = {
      /*ptr*/
      prev,
      /*incrementing value to increase when it goes past a power of two*/
      OuterCount,
      /*buffer size (element x subloops)*/
      B.CreateMul(tsize, InnerCount, "", /*NUW*/ true,
                  /*NSW*/ true)};

  auto realloccall =
      B.CreateCall(getOrInsertExponentialAllocator(*newFunc->getParent(),
                                                   newFunc, ZeroMem, T),
                   idxs, Name);
  if (caller)
    *caller = realloccall;
  return realloccall;
}

Value *CreateAllocation(IRBuilder<> &Builder, llvm::Type *T, Value *Count,
                        Twine Name, CallInst **caller, Instruction **ZeroMem,
                        bool isDefault) {
  Value *res;
  auto &M = *Builder.GetInsertBlock()->getParent()->getParent();
  auto AlignI = M.getDataLayout().getTypeAllocSizeInBits(T) / 8;
  auto Align = ConstantInt::get(Count->getType(), AlignI);
  CallInst *malloccall = nullptr;
  if (CustomAllocator) {
    LLVMValueRef wzeromem = nullptr;
    res = unwrap(CustomAllocator(wrap(&Builder), wrap(T), wrap(Count),
                                 wrap(Align), isDefault,
                                 ZeroMem ? &wzeromem : nullptr));
    if (auto I = dyn_cast<Instruction>(res))
      I->setName(Name);

    malloccall = dyn_cast<CallInst>(res);
    if (malloccall == nullptr) {
      malloccall = cast<CallInst>(cast<Instruction>(res)->getOperand(0));
    }
    if (ZeroMem) {
      *ZeroMem = cast_or_null<Instruction>(unwrap(wzeromem));
      ZeroMem = nullptr;
    }
  } else {
    if (Builder.GetInsertPoint() == Builder.GetInsertBlock()->end()) {
      res = CallInst::CreateMalloc(Builder.GetInsertBlock(), Count->getType(),
                                   T, Align, Count, nullptr, Name);
      Builder.SetInsertPoint(Builder.GetInsertBlock());
    } else {
      res = CallInst::CreateMalloc(&*Builder.GetInsertPoint(), Count->getType(),
                                   T, Align, Count, nullptr, Name);
    }
    if (!cast<Instruction>(res)->getParent())
      Builder.Insert(cast<Instruction>(res));

    malloccall = dyn_cast<CallInst>(res);
    if (malloccall == nullptr) {
      malloccall = cast<CallInst>(cast<Instruction>(res)->getOperand(0));
    }

    // Assert computation of size of array doesn't wrap
    if (auto BI = dyn_cast<BinaryOperator>(malloccall->getArgOperand(0))) {
      if (BI->getOpcode() == BinaryOperator::Mul) {
        if ((BI->getOperand(0) == Align && BI->getOperand(1) == Count) ||
            (BI->getOperand(1) == Align && BI->getOperand(0) == Count))
          BI->setHasNoSignedWrap(true);
        BI->setHasNoUnsignedWrap(true);
      }
    }

    if (auto ci = dyn_cast<ConstantInt>(Count)) {
#if LLVM_VERSION_MAJOR >= 14
      malloccall->addDereferenceableRetAttr(ci->getLimitedValue() * AlignI);
#if !defined(FLANG) && !defined(ROCM)
      AttrBuilder B(ci->getContext());
#else
      AttrBuilder B;
#endif
      B.addDereferenceableOrNullAttr(ci->getLimitedValue() * AlignI);
      malloccall->setAttributes(malloccall->getAttributes().addRetAttributes(
          malloccall->getContext(), B));
#else
      malloccall->addDereferenceableAttr(llvm::AttributeList::ReturnIndex,
                                         ci->getLimitedValue() * AlignI);
      malloccall->addDereferenceableOrNullAttr(llvm::AttributeList::ReturnIndex,
                                               ci->getLimitedValue() * AlignI);
#endif
      // malloccall->removeAttribute(llvm::AttributeList::ReturnIndex,
      // Attribute::DereferenceableOrNull);
    }
#if LLVM_VERSION_MAJOR >= 14
    malloccall->addAttributeAtIndex(AttributeList::ReturnIndex,
                                    Attribute::NoAlias);
    malloccall->addAttributeAtIndex(AttributeList::ReturnIndex,
                                    Attribute::NonNull);
#else
    malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
    malloccall->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
#endif
  }
  if (caller) {
    *caller = malloccall;
  }
  if (ZeroMem) {
    auto PT = cast<PointerType>(malloccall->getType());
    Value *tozero = malloccall;

    bool needsCast = false;
#if LLVM_VERSION_MAJOR >= 15
    if (PT->getContext().supportsTypedPointers()) {
#endif
      needsCast = !PT->getPointerElementType()->isIntegerTy(8);
#if LLVM_VERSION_MAJOR >= 15
    }
#endif
    if (needsCast)
      tozero = Builder.CreatePointerCast(
          tozero, PointerType::get(Type::getInt8Ty(PT->getContext()),
                                   PT->getAddressSpace()));
    Value *args[] = {
        tozero, ConstantInt::get(Type::getInt8Ty(malloccall->getContext()), 0),
        Builder.CreateMul(Align, Count, "", true, true),
        ConstantInt::getFalse(malloccall->getContext())};
    Type *tys[] = {args[0]->getType(), args[2]->getType()};

    *ZeroMem = Builder.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::memset, tys), args);
  }
  return res;
}

CallInst *CreateDealloc(llvm::IRBuilder<> &Builder, llvm::Value *ToFree) {
  CallInst *res = nullptr;

  if (CustomDeallocator) {
    res = dyn_cast_or_null<CallInst>(
        unwrap(CustomDeallocator(wrap(&Builder), wrap(ToFree))));
  } else {

    ToFree = Builder.CreatePointerCast(
        ToFree, Type::getInt8PtrTy(ToFree->getContext()));
    if (Builder.GetInsertPoint() == Builder.GetInsertBlock()->end()) {
      res = cast<CallInst>(
          CallInst::CreateFree(ToFree, Builder.GetInsertBlock()));
      Builder.SetInsertPoint(Builder.GetInsertBlock());
    } else {
      res = cast<CallInst>(
          CallInst::CreateFree(ToFree, &*Builder.GetInsertPoint()));
    }
    if (!cast<Instruction>(res)->getParent())
      Builder.Insert(cast<Instruction>(res));
#if LLVM_VERSION_MAJOR >= 14
    res->addAttributeAtIndex(AttributeList::FirstArgIndex, Attribute::NonNull);
#else
    res->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
  }
  return res;
}

/// Create function for type that is equivalent to memcpy but adds to
/// destination rather than a direct copy; dst, src, numelems
Function *getOrInsertDifferentialFloatMemcpy(Module &M, Type *elementType,
                                             unsigned dstalign,
                                             unsigned srcalign,
                                             unsigned dstaddr,
                                             unsigned srcaddr) {
  assert(elementType->isFloatingPointTy());
  std::string name = "__enzyme_memcpyadd_" + tofltstr(elementType) + "da" +
                     std::to_string(dstalign) + "sa" + std::to_string(srcalign);
  if (dstaddr)
    name += "dadd" + std::to_string(dstaddr);
  if (srcaddr)
    name += "sadd" + std::to_string(srcaddr);
  FunctionType *FT = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {PointerType::get(elementType, dstaddr),
                                        PointerType::get(elementType, srcaddr),
                                        Type::getInt64Ty(M.getContext())},
                                       false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoCapture);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto dst = F->arg_begin();
  dst->setName("dst");
  auto src = dst + 1;
  src->setName("src");
  auto num = src + 1;
  num->setName("num");

  {
    IRBuilder<> B(entry);
    B.CreateCondBr(B.CreateICmpEQ(num, ConstantInt::get(num->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(num->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), entry);

#if LLVM_VERSION_MAJOR > 7
    Value *dsti = B.CreateInBoundsGEP(elementType, dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(elementType, dsti, "dst.i.l");
#else
    Value *dsti = B.CreateInBoundsGEP(dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(dsti, "dst.i.l");
#endif
    StoreInst *dsts = B.CreateStore(Constant::getNullValue(elementType), dsti);
    if (dstalign) {
#if LLVM_VERSION_MAJOR >= 10
      dstl->setAlignment(Align(dstalign));
      dsts->setAlignment(Align(dstalign));
#else
      dstl->setAlignment(dstalign);
      dsts->setAlignment(dstalign);
#endif
    }

#if LLVM_VERSION_MAJOR > 7
    Value *srci = B.CreateInBoundsGEP(elementType, src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(elementType, srci, "src.i.l");
#else
    Value *srci = B.CreateInBoundsGEP(src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(srci, "src.i.l");
#endif
    StoreInst *srcs = B.CreateStore(B.CreateFAdd(srcl, dstl), srci);
    if (srcalign) {
#if LLVM_VERSION_MAJOR >= 10
      srcl->setAlignment(Align(srcalign));
      srcs->setAlignment(Align(srcalign));
#else
      srcl->setAlignment(srcalign);
      srcs->setAlignment(srcalign);
#endif
    }

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(num->getType(), 1), "idx.next");
    idx->addIncoming(next, body);
    B.CreateCondBr(B.CreateICmpEQ(num, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }
  return F;
}

Function *getOrInsertMemcpyStrided(Module &M, PointerType *T, Type *IT,
                                   unsigned dstalign, unsigned srcalign) {
  Type *elementType = T->getPointerElementType();
  assert(elementType->isFloatingPointTy());
  std::string name = "__enzyme_memcpy_" + tofltstr(elementType) + "_" +
                     std::to_string(cast<IntegerType>(IT)->getBitWidth()) +
                     "_da" + std::to_string(dstalign) + "sa" +
                     std::to_string(srcalign) + "stride";
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()), {T, T, IT, IT}, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(0, Attribute::WriteOnly);
  F->addParamAttr(1, Attribute::ReadOnly);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto dst = F->arg_begin();
  dst->setName("dst");
  auto src = dst + 1;
  src->setName("src");
  auto num = src + 1;
  num->setName("num");
  auto stride = num + 1;
  stride->setName("stride");

  {
    IRBuilder<> B(entry);
    B.CreateCondBr(B.CreateICmpEQ(num, ConstantInt::get(num->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(num->getType(), 2, "idx");
    PHINode *sidx = B.CreatePHI(num->getType(), 2, "sidx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), entry);
    sidx->addIncoming(ConstantInt::get(num->getType(), 0), entry);

#if LLVM_VERSION_MAJOR > 7
    Value *dsti = B.CreateInBoundsGEP(elementType, dst, idx, "dst.i");
    Value *srci = B.CreateInBoundsGEP(elementType, src, sidx, "src.i");
    LoadInst *srcl = B.CreateLoad(elementType, srci, "src.i.l");
#else
    Value *dsti = B.CreateInBoundsGEP(dst, idx, "dst.i");
    Value *srci = B.CreateInBoundsGEP(src, sidx, "src.i");
    LoadInst *srcl = B.CreateLoad(srci, "src.i.l");
#endif

    StoreInst *dsts = B.CreateStore(srcl, dsti);

    if (dstalign) {
#if LLVM_VERSION_MAJOR >= 10
      dsts->setAlignment(Align(dstalign));
#else
      dsts->setAlignment(dstalign);
#endif
    }
    if (srcalign) {
#if LLVM_VERSION_MAJOR >= 10
      srcl->setAlignment(Align(srcalign));
#else
      srcl->setAlignment(srcalign);
#endif
    }

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(num->getType(), 1), "idx.next");
    Value *snext = B.CreateNUWAdd(sidx, stride, "sidx.next");
    idx->addIncoming(next, body);
    sidx->addIncoming(snext, body);
    B.CreateCondBr(B.CreateICmpEQ(num, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }

  return F;
}

// TODO implement differential memmove
Function *getOrInsertDifferentialFloatMemmove(Module &M, Type *T,
                                              unsigned dstalign,
                                              unsigned srcalign,
                                              unsigned dstaddr,
                                              unsigned srcaddr) {
  llvm::errs() << "warning: didn't implement memmove, using memcpy as fallback "
                  "which can result in errors\n";
  return getOrInsertDifferentialFloatMemcpy(M, T, dstalign, srcalign, dstaddr,
                                            srcaddr);
}

Function *getOrInsertCheckedFree(Module &M, CallInst *call, Type *Ty,
                                 unsigned width) {
  FunctionType *FreeTy = call->getFunctionType();
#if LLVM_VERSION_MAJOR >= 11
  Value *Free = call->getCalledOperand();
#else
  Value *Free = call->getCalledValue();
#endif
  AttributeList FreeAttributes = call->getAttributes();
  CallingConv::ID CallingConvention = call->getCallingConv();
  DebugLoc DebugLoc = call->getDebugLoc();

  std::string name = "__enzyme_checked_free_" + std::to_string(width);

  SmallVector<Type *, 3> types;
  types.push_back(Ty);
  for (unsigned i = 0; i < width; i++) {
    types.push_back(Ty);
  }

  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()), types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *free0 = BasicBlock::Create(M.getContext(), "free0", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "end", F);

  IRBuilder<> EntryBuilder(entry);
  IRBuilder<> Free0Builder(free0);
  IRBuilder<> EndBuilder(end);

  auto primal = F->arg_begin();
  Argument *first_shadow = F->arg_begin() + 1;
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoCapture);

  Value *isNotEqual = EntryBuilder.CreateICmpNE(primal, first_shadow);
  EntryBuilder.CreateCondBr(isNotEqual, free0, end);

  CallInst *CI = Free0Builder.CreateCall(FreeTy, Free, {first_shadow});
  CI->setAttributes(FreeAttributes);
  CI->setCallingConv(CallingConvention);
  CI->setDebugLoc(DebugLoc);

  if (width > 1) {
    Value *checkResult = nullptr;
    BasicBlock *free1 = BasicBlock::Create(M.getContext(), "free1", F);
    IRBuilder<> Free1Builder(free1);

    for (unsigned i = 0; i < width; i++) {
      F->addParamAttr(i + 1, Attribute::NoCapture);
      Argument *shadow = F->arg_begin() + i + 1;

      if (i < width - 1) {
        Argument *nextShadow = F->arg_begin() + i + 2;
        Value *isNotEqual = Free0Builder.CreateICmpNE(shadow, nextShadow);
        checkResult = checkResult
                          ? Free0Builder.CreateAnd(isNotEqual, checkResult)
                          : isNotEqual;

        CallInst *CI = Free1Builder.CreateCall(FreeTy, Free, {nextShadow});
        CI->setAttributes(FreeAttributes);
        CI->setCallingConv(CallingConvention);
        CI->setDebugLoc(DebugLoc);
      }
    }
    Free0Builder.CreateCondBr(checkResult, free1, end);
    Free1Builder.CreateBr(end);
  } else {
    Free0Builder.CreateBr(end);
  }

  EndBuilder.CreateRetVoid();

  return F;
}

llvm::Function *getOrInsertDifferentialMPI_Wait(llvm::Module &M,
                                                ArrayRef<llvm::Type *> T,
                                                Type *reqType) {
  llvm::SmallVector<llvm::Type *, 4> types(T.begin(), T.end());
  types.push_back(reqType);
  std::string name = "__enzyme_differential_mpi_wait";
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()), types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *isend = BasicBlock::Create(M.getContext(), "invertISend", F);
  BasicBlock *irecv = BasicBlock::Create(M.getContext(), "invertIRecv", F);

#if 0
  /*0 */Type::getInt8PtrTy(call.getContext())
  /*1 */i64
  /*2 */Type::getInt8PtrTy(call.getContext())
  /*3 */i64
  /*4 */i64
  /*5 */Type::getInt8PtrTy(call.getContext())
  /*6 */Type::getInt8Ty(call.getContext())
#endif

  auto buff = F->arg_begin();
  buff->setName("buf");
  Value *buf = buff;
  Value *count = buff + 1;
  count->setName("count");
  Value *datatype = buff + 2;
  datatype->setName("datatype");
  Value *source = buff + 3;
  source->setName("source");
  Value *tag = buff + 4;
  tag->setName("tag");
  Value *comm = buff + 5;
  comm->setName("comm");
  Value *fn = buff + 6;
  fn->setName("fn");
  Value *d_req = buff + 7;
  d_req->setName("d_req");

  bool pmpi = true;
  auto isendfn = M.getFunction("PMPI_Isend");
  if (!isendfn) {
    isendfn = M.getFunction("MPI_Isend");
    pmpi = false;
  }
  assert(isendfn);
  auto irecvfn = M.getFunction("PMPI_Irecv");
  if (!irecvfn)
    irecvfn = M.getFunction("MPI_Irecv");
  if (!irecvfn) {
    FunctionType *FuT = isendfn->getFunctionType();
    std::string name = pmpi ? "PMPI_Irecv" : "MPI_Irecv";
#if LLVM_VERSION_MAJOR >= 9
    irecvfn = cast<Function>(M.getOrInsertFunction(name, FuT).getCallee());

#else
    irecvfn = cast<Function>(M.getOrInsertFunction(name, FuT));
#endif
  }
  assert(irecvfn);

  IRBuilder<> B(entry);
  auto arg = isendfn->arg_begin();
  if (arg->getType()->isIntegerTy())
    buf = B.CreatePtrToInt(buf, arg->getType());
  arg++;
  count = B.CreateZExtOrTrunc(count, arg->getType());
  arg++;
  datatype = B.CreatePointerCast(datatype, arg->getType());
  arg++;
  source = B.CreateZExtOrTrunc(source, arg->getType());
  arg++;
  tag = B.CreateZExtOrTrunc(tag, arg->getType());
  arg++;
  comm = B.CreatePointerCast(comm, arg->getType());
  arg++;
  if (arg->getType()->isIntegerTy())
    d_req = B.CreatePtrToInt(d_req, arg->getType());
  Value *args[] = {
      buf, count, datatype, source, tag, comm, d_req,
  };

  B.CreateCondBr(B.CreateICmpEQ(fn, ConstantInt::get(fn->getType(),
                                                     (int)MPI_CallType::ISEND)),
                 isend, irecv);

  {
    B.SetInsertPoint(isend);
    auto fcall = B.CreateCall(irecvfn, args);
    fcall->setCallingConv(isendfn->getCallingConv());
    B.CreateRetVoid();
  }

  {
    B.SetInsertPoint(irecv);
    auto fcall = B.CreateCall(isendfn, args);
    fcall->setCallingConv(isendfn->getCallingConv());
    B.CreateRetVoid();
  }
  return F;
}

llvm::Function *getOrInsertDifferentialWaitallSave(llvm::Module &M,
                                                   ArrayRef<llvm::Type *> T,
                                                   PointerType *reqType) {
  std::string name = "__enzyme_differential_waitall_save";
  FunctionType *FT =
      FunctionType::get(PointerType::getUnqual(reqType), T, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);

  auto buff = F->arg_begin();
  buff->setName("count");
  Value *count = buff;
  Value *req = buff + 1;
  req->setName("req");
  Value *dreq = buff + 2;
  dreq->setName("dreq");

  IRBuilder<> B(entry);
  count = B.CreateZExtOrTrunc(count, Type::getInt64Ty(entry->getContext()));

  auto ret = CreateAllocation(B, reqType, count);

  BasicBlock *loopBlock = BasicBlock::Create(M.getContext(), "loop", F);
  BasicBlock *endBlock = BasicBlock::Create(M.getContext(), "end", F);

  B.CreateCondBr(B.CreateICmpEQ(count, ConstantInt::get(count->getType(), 0)),
                 endBlock, loopBlock);

  B.SetInsertPoint(loopBlock);
  auto idx = B.CreatePHI(count->getType(), 2);
  idx->addIncoming(ConstantInt::get(count->getType(), 0), entry);
  auto inc = B.CreateAdd(idx, ConstantInt::get(count->getType(), 1));
  idx->addIncoming(inc, loopBlock);

  Value *idxs[] = {idx};
#if LLVM_VERSION_MAJOR > 7
  Value *ireq =
      B.CreateInBoundsGEP(req->getType()->getPointerElementType(), req, idxs);
  Value *idreq =
      B.CreateInBoundsGEP(req->getType()->getPointerElementType(), dreq, idxs);
  Value *iout = B.CreateInBoundsGEP(reqType, ret, idxs);
#else
  Value *ireq = B.CreateInBoundsGEP(req, idxs);
  Value *idreq = B.CreateInBoundsGEP(dreq, idxs);
  Value *iout = B.CreateInBoundsGEP(ret, idxs);
#endif
  Value *isNull = nullptr;
  if (auto GV = M.getNamedValue("ompi_request_null")) {
    Value *reql =
        B.CreatePointerCast(ireq, PointerType::getUnqual(GV->getType()));
#if LLVM_VERSION_MAJOR > 7
    reql = B.CreateLoad(GV->getType(), reql);
#else
    reql = B.CreateLoad(reql);
#endif
    isNull = B.CreateICmpEQ(reql, GV);
  }

  idreq = B.CreatePointerCast(idreq, PointerType::getUnqual(reqType));
#if LLVM_VERSION_MAJOR > 7
  Value *d_reqp = B.CreateLoad(reqType, idreq);
#else
  Value *d_reqp = B.CreateLoad(idreq);
#endif
  if (isNull)
    d_reqp = B.CreateSelect(isNull, Constant::getNullValue(d_reqp->getType()),
                            d_reqp);

  B.CreateStore(d_reqp, iout);

  B.CreateCondBr(B.CreateICmpEQ(inc, count), endBlock, loopBlock);

  B.SetInsertPoint(endBlock);
  B.CreateRet(ret);
  return F;
}

void ErrorIfRuntimeInactive(llvm::IRBuilder<> &B, llvm::Value *primal,
                            llvm::Value *shadow, const char *Message,
                            llvm::DebugLoc &&loc, llvm::Instruction *orig) {
  Module &M = *B.GetInsertBlock()->getParent()->getParent();
  std::string name = "__enzyme_runtimeinactiveerr";
  if (CustomRuntimeInactiveError) {
    static int count = 0;
    name += std::to_string(count);
    count++;
  }
  FunctionType *FT = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       {Type::getInt8PtrTy(M.getContext()),
                                        Type::getInt8PtrTy(M.getContext()),
                                        Type::getInt8PtrTy(M.getContext())},
                                       false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT));
#endif

  if (F->empty()) {
    F->setLinkage(Function::LinkageTypes::InternalLinkage);
    F->addFnAttr(Attribute::AlwaysInline);
    F->addParamAttr(0, Attribute::NoCapture);
    F->addParamAttr(1, Attribute::NoCapture);

    BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
    BasicBlock *error = BasicBlock::Create(M.getContext(), "error", F);
    BasicBlock *end = BasicBlock::Create(M.getContext(), "end", F);

    auto prim = F->arg_begin();
    prim->setName("primal");
    auto shadow = prim + 1;
    shadow->setName("shadow");
    auto msg = prim + 2;
    msg->setName("msg");

    IRBuilder<> EB(entry);
    EB.CreateCondBr(EB.CreateICmpEQ(prim, shadow), error, end);

    EB.SetInsertPoint(error);

    if (CustomRuntimeInactiveError) {
      CustomRuntimeInactiveError(wrap(&EB), wrap(msg), wrap(orig));
    } else {
      FunctionType *FT =
          FunctionType::get(Type::getInt32Ty(M.getContext()),
                            {Type::getInt8PtrTy(M.getContext())}, false);

#if LLVM_VERSION_MAJOR >= 9
      auto PutsF = M.getOrInsertFunction("puts", FT);
#else
      auto PutsF = M.getOrInsertFunction("puts", FT);
#endif
      EB.CreateCall(PutsF, msg);

      FunctionType *FT2 =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            {Type::getInt32Ty(M.getContext())}, false);

#if LLVM_VERSION_MAJOR >= 9
      auto ExitF = M.getOrInsertFunction("exit", FT2);
#else
      auto ExitF = M.getOrInsertFunction("exit", FT2);
#endif
      EB.CreateCall(ExitF,
                    ConstantInt::get(Type::getInt32Ty(M.getContext()), 1));
    }
    EB.CreateUnreachable();

    EB.SetInsertPoint(end);
    EB.CreateRetVoid();
  }

  Value *args[] = {
      B.CreatePointerCast(primal, Type::getInt8PtrTy(M.getContext())),
      B.CreatePointerCast(shadow, Type::getInt8PtrTy(M.getContext())),
      getString(M, Message)};
  auto call = B.CreateCall(F, args);
  call->setDebugLoc(loc);
}
