//===- Utils.cpp - Definition of miscellaneous utilities ------------------===//
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
// This file defines miscellaneous utilities that are used as part of the
// AD process.
//
//===----------------------------------------------------------------------===//
#include "Utils.h"
#include "TypeAnalysis/TypeAnalysis.h"

#if LLVM_VERSION_MAJOR >= 16
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "TypeAnalysis/TBAA.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "llvm-c/Core.h"

#include "LibraryFuncs.h"

using namespace llvm;

extern "C" {
LLVMValueRef (*CustomErrorHandler)(const char *, LLVMValueRef, ErrorType,
                                   const void *, LLVMValueRef,
                                   LLVMBuilderRef) = nullptr;
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
LLVMValueRef (*EnzymeUndefinedValueForType)(LLVMTypeRef, uint8_t) = nullptr;

LLVMValueRef (*EnzymeSanitizeDerivatives)(LLVMValueRef, LLVMValueRef toset,
                                          LLVMBuilderRef,
                                          LLVMValueRef) = nullptr;

extern llvm::cl::opt<bool> EnzymeZeroCache;

// default to false because lacpy is slow
llvm::cl::opt<bool>
    EnzymeLapackCopy("enzyme-lapack-copy", cl::init(false), cl::Hidden,
                     cl::desc("Use blas copy calls to cache matrices"));
llvm::cl::opt<bool>
    EnzymeBlasCopy("enzyme-blas-copy", cl::init(true), cl::Hidden,
                   cl::desc("Use blas copy calls to cache vectors"));
llvm::cl::opt<bool>
    EnzymeFastMath("enzyme-fast-math", cl::init(true), cl::Hidden,
                   cl::desc("Use fast math on derivative compuation"));
llvm::cl::opt<bool>
    EnzymeStrongZero("enzyme-strong-zero", cl::init(false), cl::Hidden,
                     cl::desc("Use additional checks to ensure correct "
                              "behavior when handling functions with inf"));
}

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
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

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
      tmp = BB.CreateInBoundsGEP(Type::getInt8Ty(tmp->getContext()), tmp,
                                 prevSize);
      tmp = BB.CreatePointerCast(tmp, tmpT);
      SubZero->setOperand(0, tmp);
      SubZero->setOperand(2, zeroSize);
    }
  }

  if (ZeroInit) {
    Value *zeroSize = B.CreateSub(next, prevSize);

    Value *margs[] = {B.CreateInBoundsGEP(B.getInt8Ty(), gVal, prevSize),
                      B.getInt8(0), zeroSize, B.getFalse()};
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
                                llvm::Value *InnerCount,
                                const llvm::Twine &Name,
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
                        const Twine &Name, CallInst **caller,
                        Instruction **ZeroMem, bool isDefault) {
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
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
    if (PT->getContext().supportsTypedPointers()) {
#endif
      needsCast = !PT->getPointerElementType()->isIntegerTy(8);
#if LLVM_VERSION_MAJOR >= 15
    }
#endif
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

EnzymeFailure::EnzymeFailure(const llvm::Twine &RemarkName,
                             const llvm::DiagnosticLocation &Loc,
                             const llvm::Instruction *CodeRegion)
    : DiagnosticInfoUnsupported(*CodeRegion->getParent()->getParent(),
                                RemarkName, Loc) {}

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

  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

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

      auto PutsF = M.getOrInsertFunction("puts", FT);
      EB.CreateCall(PutsF, msg);

      FunctionType *FT2 =
          FunctionType::get(Type::getVoidTy(M.getContext()),
                            {Type::getInt32Ty(M.getContext())}, false);

      auto ExitF = M.getOrInsertFunction("exit", FT2);
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

/// Create function for type that is equivalent to memcpy but adds to
/// destination rather than a direct copy; dst, src, numelems
Function *getOrInsertDifferentialFloatMemcpy(Module &M, Type *elementType,
                                             unsigned dstalign,
                                             unsigned srcalign,
                                             unsigned dstaddr, unsigned srcaddr,
                                             unsigned bitwidth) {
  assert(elementType->isFloatingPointTy());
  std::string name = "__enzyme_memcpy";
  if (bitwidth != 64)
    name += std::to_string(bitwidth);
  name += "add_" + tofltstr(elementType) + "da" + std::to_string(dstalign) +
          "sa" + std::to_string(srcalign);
  if (dstaddr)
    name += "dadd" + std::to_string(dstaddr);
  if (srcaddr)
    name += "sadd" + std::to_string(srcaddr);
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()),
                        {PointerType::get(elementType, dstaddr),
                         PointerType::get(elementType, srcaddr),
                         IntegerType::get(M.getContext(), bitwidth)},
                        false);

  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
#if LLVM_VERSION_MAJOR >= 16
  F->setOnlyAccessesArgMemory();
#else
  F->addFnAttr(Attribute::ArgMemOnly);
#endif
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

    Value *dsti = B.CreateInBoundsGEP(elementType, dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(elementType, dsti, "dst.i.l");
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

    Value *srci = B.CreateInBoundsGEP(elementType, src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(elementType, srci, "src.i.l");
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

void callMemcpyStridedBlas(llvm::IRBuilder<> &B, llvm::Module &M, BlasInfo blas,
                           llvm::ArrayRef<llvm::Value *> args,
                           llvm::ArrayRef<llvm::OperandBundleDef> bundles) {
  std::string copy_name =
      (blas.prefix + blas.floatType + "copy" + blas.suffix).str();

  SmallVector<Type *, 1> tys;
  for (auto arg : args)
    tys.push_back(arg->getType());

  auto FT = FunctionType::get(Type::getVoidTy(M.getContext()), tys, false);
  auto fn = M.getOrInsertFunction(copy_name, FT);

  B.CreateCall(fn, args, bundles);
}

void callMemcpyStridedLapack(llvm::IRBuilder<> &B, llvm::Module &M,
                             BlasInfo blas, llvm::ArrayRef<llvm::Value *> args,
                             llvm::ArrayRef<llvm::OperandBundleDef> bundles) {
  std::string copy_name = (blas.floatType + "lacpy" + blas.suffix).str();

  SmallVector<Type *, 1> tys;
  for (auto arg : args)
    tys.push_back(arg->getType());

  auto FT = FunctionType::get(Type::getVoidTy(M.getContext()), tys, false);
  auto fn = M.getOrInsertFunction(copy_name, FT);

  B.CreateCall(fn, args, bundles);
}

void callSPMVDiagUpdate(IRBuilder<> &B, Module &M, BlasInfo blas,
                        IntegerType *IT, Type *BlasCT,
                        Type *BlasFPT, Type *BlasPT,
                        Type *BlasIT, Type *fpTy,
                        ArrayRef<Value *> args,
                        ArrayRef<OperandBundleDef> bundles,
                        bool byRef, bool julia_decl) {
  // add spmv diag update call if not already present
  std::string fnc_name =
      ("__enzyme_spmv_diag" + blas.floatType + blas.suffix).str();

  //  spmvDiagHelper(uplo, n, alpha, x, incx, ya, incy, APa)
  auto FDiagUpdateT = FunctionType::get(
      B.getVoidTy(),
      {BlasCT, BlasIT, BlasFPT, BlasPT, BlasIT, BlasPT, BlasIT, BlasPT}, false);
  Function *F =
      cast<Function>(M.getOrInsertFunction(fnc_name, FDiagUpdateT).getCallee());

  if (!F->empty()) {
    B.CreateCall(F, args, bundles);
    return;
  }

  // now add the implementation for the call
  F->setLinkage(Function::LinkageTypes::InternalLinkage);
#if LLVM_VERSION_MAJOR >= 16
  F->setOnlyAccessesArgMemory();
#else
  F->addFnAttr(Attribute::ArgMemOnly);
#endif
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  if (!julia_decl) {
    F->addParamAttr(3, Attribute::NoCapture);
    F->addParamAttr(5, Attribute::NoCapture);
    F->addParamAttr(7, Attribute::NoCapture);
    F->addParamAttr(3, Attribute::NoAlias);
    F->addParamAttr(5, Attribute::NoAlias);
    F->addParamAttr(7, Attribute::NoAlias);
    F->addParamAttr(3, Attribute::ReadOnly);
    F->addParamAttr(5, Attribute::ReadOnly);
    if (byRef) {
      F->addParamAttr(2, Attribute::NoCapture);
      F->addParamAttr(2, Attribute::NoAlias);
      F->addParamAttr(2, Attribute::ReadOnly);
    }
  }

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *init = BasicBlock::Create(M.getContext(), "init", F);
  BasicBlock *uper_code = BasicBlock::Create(M.getContext(), "uper", F);
  BasicBlock *lower_code = BasicBlock::Create(M.getContext(), "lower", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  //  spmvDiagHelper(uplo, n, alpha, x, incx, ya, incy, APa)
  auto blasuplo = F->arg_begin();
  blasuplo->setName("blasuplo");
  auto blasn = blasuplo + 1;
  blasn->setName("blasn");
  auto blasalpha = blasn + 1;
  blasalpha->setName("blasalpha");
  auto blasx = blasalpha + 1;
  blasx->setName("blasx");
  auto blasincx = blasx + 1;
  blasincx->setName("blasincx");
  auto blasdy = blasx + 1;
  blasdy->setName("blasdy");
  auto blasincy = blasdy + 1;
  blasincy->setName("blasincy");
  auto blasdAP = blasincy + 1;
  blasdAP->setName("blasdAP");

  // TODO: consider cblas_layout

  // https://dl.acm.org/doi/pdf/10.1145/3382191
  // Following example is Fortran based, thus 1 indexed
  // if(uplo == 'u' .or. uplo == 'U') then
  //   k = 0
  //   do i = 1,n
  //     k = k+i
  //     APa(k) = APa(k) - alpha*x(1 + (i-1)*incx)*ya(1 + (i-1)*incy)
  //   end do
  // else
  //   k = 1
  //   do i = 1,n
  //     APa(k) = APa(k) - alpha*x(1 + (i-1)*incx)*ya(1 + (i-1)*incy)
  //     k = k+n-i+1
  //   end do
  // end if
  {
    IRBuilder<> B1(entry);
    Value *n = load_if_ref(B1, IT, blasn, byRef);
    Value *incx = load_if_ref(B1, IT, blasincx, byRef);
    Value *incy = load_if_ref(B1, IT, blasincy, byRef);
    Value *alpha = blasalpha;
    if (byRef) {
      auto VP = B1.CreatePointerCast(
          blasalpha,
          PointerType::get(
              fpTy,
              cast<PointerType>(blasalpha->getType())->getAddressSpace()));
      alpha = B1.CreateLoad(fpTy, VP);
    }
    Value *is_u = is_uper(B1, blasuplo, byRef);
    Value *k = B1.CreateSelect(is_u, ConstantInt::get(IT, 0),
                               ConstantInt::get(IT, 1), "k");
    B1.CreateCondBr(B1.CreateICmpEQ(n, ConstantInt::get(IT, 0)), end, init);

    IRBuilder<> B2(init);
    Value *xfloat = B2.CreatePointerCast(
        blasx,
        PointerType::get(
            fpTy, cast<PointerType>(blasx->getType())->getAddressSpace()));
    Value *dyfloat = B2.CreatePointerCast(
        blasdy,
        PointerType::get(
            fpTy, cast<PointerType>(blasdy->getType())->getAddressSpace()));
    Value *dAPfloat = B2.CreatePointerCast(
        blasdAP,
        PointerType::get(
            fpTy, cast<PointerType>(blasdAP->getType())->getAddressSpace()));
    B2.CreateCondBr(is_u, uper_code, lower_code);

    IRBuilder<> B3(uper_code);
    B3.setFastMathFlags(getFast());
    {
      PHINode *iter = B3.CreatePHI(IT, 2, "iteration");
      PHINode *kval = B3.CreatePHI(IT, 2, "k");
      iter->addIncoming(ConstantInt::get(IT, 0), init);
      kval->addIncoming(ConstantInt::get(IT, 0), init);
      Value *iternext =
          B3.CreateAdd(iter, ConstantInt::get(IT, 1), "iter.next");
      // 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, ... are diag elements
      Value *kvalnext = B3.CreateAdd(kval, iternext, "k.next");
      iter->addIncoming(iternext, uper_code);
      kval->addIncoming(kvalnext, uper_code);

      Value *xidx = B3.CreateNUWMul(iter, incx, "x.idx");
      Value *yidx = B3.CreateNUWMul(iter, incy, "y.idx");
      Value *x = B3.CreateInBoundsGEP(fpTy, xfloat, xidx, "x.ptr");
      Value *y = B3.CreateInBoundsGEP(fpTy, dyfloat, yidx, "y.ptr");
      Value *xval = B3.CreateLoad(fpTy, x, "x.val");
      Value *yval = B3.CreateLoad(fpTy, y, "y.val");
      Value *xy = B3.CreateFMul(xval, yval, "xy");
      Value *xyalpha = B3.CreateFMul(xy, alpha, "xy.alpha");
      Value *kptr = B3.CreateInBoundsGEP(fpTy, dAPfloat, kval, "k.ptr");
      Value *kvalloaded = B3.CreateLoad(fpTy, kptr, "k.val");
      Value *kvalnew = B3.CreateFSub(kvalloaded, xyalpha, "k.val.new");
      B3.CreateStore(kvalnew, kptr);

      B3.CreateCondBr(B3.CreateICmpEQ(iternext, n), end, uper_code);
    }

    IRBuilder<> B4(lower_code);
    B4.setFastMathFlags(getFast());
    {
      PHINode *iter = B4.CreatePHI(IT, 2, "iteration");
      PHINode *kval = B4.CreatePHI(IT, 2, "k");
      iter->addIncoming(ConstantInt::get(IT, 0), init);
      kval->addIncoming(ConstantInt::get(IT, 0), init);
      Value *iternext =
          B4.CreateAdd(iter, ConstantInt::get(IT, 1), "iter.next");
      Value *ktmp = B4.CreateAdd(n, ConstantInt::get(IT, 1), "tmp.val");
      Value *ktmp2 = B4.CreateSub(ktmp, iternext, "tmp.val.other");
      Value *kvalnext = B4.CreateAdd(kval, ktmp2, "k.next");
      iter->addIncoming(iternext, lower_code);
      kval->addIncoming(kvalnext, lower_code);

      Value *xidx = B4.CreateNUWMul(iter, incx, "x.idx");
      Value *yidx = B4.CreateNUWMul(iter, incy, "y.idx");
      Value *x = B4.CreateInBoundsGEP(fpTy, xfloat, xidx, "x.ptr");
      Value *y = B4.CreateInBoundsGEP(fpTy, dyfloat, yidx, "y.ptr");
      Value *xval = B4.CreateLoad(fpTy, x, "x.val");
      Value *yval = B4.CreateLoad(fpTy, y, "y.val");
      Value *xy = B4.CreateFMul(xval, yval, "xy");
      Value *xyalpha = B4.CreateFMul(xy, alpha, "xy.alpha");
      Value *kptr = B4.CreateInBoundsGEP(fpTy, dAPfloat, kval, "k.ptr");
      Value *kvalloaded = B4.CreateLoad(fpTy, kptr, "k.val");
      Value *kvalnew = B4.CreateFSub(kvalloaded, xyalpha, "k.val.new");
      B4.CreateStore(kvalnew, kptr);

      B4.CreateCondBr(B4.CreateICmpEQ(iternext, n), end, lower_code);
    }

    IRBuilder<> B5(end);
    B5.CreateRetVoid();
  }
  B.CreateCall(F, args, bundles);
  return;
}

llvm::CallInst *
getorInsertInnerProd(llvm::IRBuilder<> &B, llvm::Module &M, BlasInfo blas,
                     IntegerType *IT, Type *BlasPT, Type *BlasIT, Type *fpTy,
                     llvm::ArrayRef<llvm::Value *> args,
                     const llvm::ArrayRef<llvm::OperandBundleDef> bundles,
                     bool byRef, bool julia_decl) {
  assert(fpTy->isFloatingPointTy());

  // add inner_prod call if not already present
  std::string prod_name =
      ("__enzyme_inner_prod" + blas.floatType + blas.suffix).str();
  auto FInnerProdT =
      FunctionType::get(fpTy, {BlasIT, BlasIT, BlasPT, BlasIT, BlasPT}, false);
  Function *F =
      cast<Function>(M.getOrInsertFunction(prod_name, FInnerProdT).getCallee());

  if (!F->empty())
    return B.CreateCall(F, args, bundles);

  // add dot call if not already present
  std::string dot_name =
      (blas.prefix + blas.floatType + "dot" + blas.suffix).str();
  auto FDotT =
      FunctionType::get(fpTy, {BlasIT, BlasPT, BlasIT, BlasPT, BlasIT}, false);
  Function *FDot =
      cast<Function>(M.getOrInsertFunction(dot_name, FDotT).getCallee());

  // now add the implementation for the inner_prod call
  F->setLinkage(Function::LinkageTypes::InternalLinkage);
#if LLVM_VERSION_MAJOR >= 16
  F->setOnlyAccessesArgMemory();
  F->setOnlyReadsMemory();
#else
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::ReadOnly);
#endif
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  if (!julia_decl) {
    F->addParamAttr(2, Attribute::NoCapture);
    F->addParamAttr(4, Attribute::NoCapture);
    F->addParamAttr(2, Attribute::NoAlias);
    F->addParamAttr(4, Attribute::NoAlias);
    F->addParamAttr(2, Attribute::ReadOnly);
    F->addParamAttr(4, Attribute::ReadOnly);
  }

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *init = BasicBlock::Create(M.getContext(), "init.idx", F);
  BasicBlock *fastPath = BasicBlock::Create(M.getContext(), "fast.path", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  // This is the .td declaration which we need to match
  // No need to support ld for the second matrix, as it will
  // always be based on a matrix which we allocated (contiguous)
  //(FrobInnerProd<> $m, $n, adj<"C">, $ldc, use<"AB">)

  auto blasm = F->arg_begin();
  blasm->setName("blasm");
  auto blasn = blasm + 1;
  blasn->setName("blasn");
  auto matA = blasn + 1;
  matA->setName("A");
  auto blaslda = matA + 1;
  blaslda->setName("lda");
  auto matB = blaslda + 1;
  matB->setName("B");

  {
    IRBuilder<> B1(entry);
    Value *blasOne = to_blas_callconv(B1, ConstantInt::get(IT, 1), byRef, IT,
                                      B1, "constant.one");
    Value *m = load_if_ref(B1, IT, blasm, byRef);
    Value *n = load_if_ref(B1, IT, blasn, byRef);
    Value *size = B1.CreateNUWMul(m, n, "mat.size");
    Value *blasSize = to_blas_callconv(B1, size, byRef, IT, B1, "mat.size");
    B1.CreateCondBr(B1.CreateICmpEQ(size, ConstantInt::get(IT, 0)), end, init);

    IRBuilder<> B2(init);
    B2.setFastMathFlags(getFast());
    Value *lda = load_if_ref(B2, IT, blaslda, byRef);
    Value *Afloat = B2.CreatePointerCast(
        matA, PointerType::get(
                  fpTy, cast<PointerType>(matA->getType())->getAddressSpace()));
    Value *Bfloat = B2.CreatePointerCast(
        matB, PointerType::get(
                  fpTy, cast<PointerType>(matB->getType())->getAddressSpace()));
    B2.CreateCondBr(B2.CreateICmpEQ(m, lda), fastPath, body);

    // our second matrix is always continuos, by construction.
    // If our first matrix is continuous too (lda == m), then we can
    // use a single dot call.
    IRBuilder<> B3(fastPath);
    B3.setFastMathFlags(getFast());
    Value *blasA = B3.CreatePointerCast(matA, BlasPT);
    Value *blasB = B3.CreatePointerCast(matB, BlasPT);
    Value *fastSum = B3.CreateCall(
        FDot, {blasSize, blasA, blasOne, blasB, blasOne}, bundles);
    B3.CreateBr(end);

    IRBuilder<> B4(body);
    B4.setFastMathFlags(getFast());
    PHINode *Aidx = B4.CreatePHI(IT, 2, "Aidx");
    PHINode *Bidx = B4.CreatePHI(IT, 2, "Bidx");
    PHINode *iter = B4.CreatePHI(IT, 2, "iteration");
    PHINode *sum = B4.CreatePHI(fpTy, 2, "sum");
    Aidx->addIncoming(ConstantInt::get(IT, 0), init);
    Bidx->addIncoming(ConstantInt::get(IT, 0), init);
    iter->addIncoming(ConstantInt::get(IT, 0), init);
    sum->addIncoming(ConstantFP::get(fpTy, 0.0), init);

    Value *Ai = B4.CreateInBoundsGEP(fpTy, Afloat, Aidx, "A.i");
    Value *Bi = B4.CreateInBoundsGEP(fpTy, Bfloat, Bidx, "B.i");
    Value *AiDot = B4.CreatePointerCast(Ai, BlasPT);
    Value *BiDot = B4.CreatePointerCast(Bi, BlasPT);
    Value *newDot =
        B4.CreateCall(FDot, {blasm, AiDot, blasOne, BiDot, blasOne}, bundles);

    Value *Anext = B4.CreateNUWAdd(Aidx, lda, "Aidx.next");
    Value *Bnext = B4.CreateNUWAdd(Aidx, m, "Bidx.next");
    Value *iternext = B4.CreateAdd(iter, ConstantInt::get(IT, 1), "iter.next");
    Value *sumnext = B4.CreateFAdd(sum, newDot);

    iter->addIncoming(iternext, body);
    Aidx->addIncoming(Anext, body);
    Bidx->addIncoming(Bnext, body);
    sum->addIncoming(sumnext, body);

    B4.CreateCondBr(B4.CreateICmpEQ(iter, n), end, body);

    IRBuilder<> B5(end);
    PHINode *res = B5.CreatePHI(fpTy, 3, "res");
    res->addIncoming(ConstantFP::get(fpTy, 0.0), entry);
    res->addIncoming(sum, body);
    res->addIncoming(fastSum, fastPath);
    B5.CreateRet(res);
  }

  return B.CreateCall(F, args, bundles);
}

Function *getOrInsertMemcpyStrided(Module &M, Type *elementType, PointerType *T,
                                   Type *IT, unsigned dstalign,
                                   unsigned srcalign) {
  assert(elementType->isFloatingPointTy());
  std::string name = "__enzyme_memcpy_" + tofltstr(elementType) + "_" +
                     std::to_string(cast<IntegerType>(IT)->getBitWidth()) +
                     "_da" + std::to_string(dstalign) + "sa" +
                     std::to_string(srcalign) + "stride";
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M.getContext()), {T, T, IT, IT}, false);

  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
#if LLVM_VERSION_MAJOR >= 16
  F->setOnlyAccessesArgMemory();
#else
  F->addFnAttr(Attribute::ArgMemOnly);
#endif
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(0, Attribute::NoAlias);
  F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoAlias);
  F->addParamAttr(0, Attribute::WriteOnly);
  F->addParamAttr(1, Attribute::ReadOnly);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *init = BasicBlock::Create(M.getContext(), "init.idx", F);
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
                   end, init);
  }

  {
    IRBuilder<> B2(init);
    B2.setFastMathFlags(getFast());
    Value *a = B2.CreateNSWSub(ConstantInt::get(num->getType(), 1), num, "a");
    Value *negidx = B2.CreateNSWMul(a, stride, "negidx");
    // Value *negidx =
    //     B2.CreateNSWAdd(b, ConstantInt::get(num->getType(), 1),
    //     "negidx");
    Value *isneg =
        B2.CreateICmpSLT(stride, ConstantInt::get(num->getType(), 0), "is.neg");
    Value *startidx = B2.CreateSelect(
        isneg, negidx, ConstantInt::get(num->getType(), 0), "startidx");
    B2.CreateBr(body);
    //}

    //{
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(num->getType(), 2, "idx");
    PHINode *sidx = B.CreatePHI(num->getType(), 2, "sidx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), init);
    sidx->addIncoming(startidx, init);

    Value *dsti = B.CreateInBoundsGEP(elementType, dst, idx, "dst.i");
    Value *srci = B.CreateInBoundsGEP(elementType, src, sidx, "src.i");
    LoadInst *srcl = B.CreateLoad(elementType, srci, "src.i.l");
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
        B.CreateNSWAdd(idx, ConstantInt::get(num->getType(), 1), "idx.next");
    Value *snext = B.CreateNSWAdd(sidx, stride, "sidx.next");
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

Function *getOrInsertMemcpyMat(Module &Mod, Type *elementType, PointerType *PT,
                               IntegerType *IT, unsigned dstalign,
                               unsigned srcalign) {
  assert(elementType->isFloatingPointTy());
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
  if (Mod.getContext().supportsTypedPointers()) {
#endif
    assert(PT->getPointerElementType() == elementType);
#if LLVM_VERSION_MAJOR >= 15
  }
#endif
#endif
  std::string name = "__enzyme_memcpy_" + tofltstr(elementType) + "_mat_" +
                     std::to_string(cast<IntegerType>(IT)->getBitWidth());
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Mod.getContext()),
                                       {PT, PT, IT, IT, IT}, false);

  Function *F = cast<Function>(Mod.getOrInsertFunction(name, FT).getCallee());

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
#if LLVM_VERSION_MAJOR >= 16
  F->setOnlyAccessesArgMemory();
#else
  F->addFnAttr(Attribute::ArgMemOnly);
#endif
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(0, Attribute::NoAlias);
  F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::NoAlias);
  F->addParamAttr(0, Attribute::WriteOnly);
  F->addParamAttr(1, Attribute::ReadOnly);

  BasicBlock *entry = BasicBlock::Create(F->getContext(), "entry", F);
  BasicBlock *init = BasicBlock::Create(F->getContext(), "init.idx", F);
  BasicBlock *body = BasicBlock::Create(F->getContext(), "for.body", F);
  BasicBlock *initend = BasicBlock::Create(F->getContext(), "init.end", F);
  BasicBlock *end = BasicBlock::Create(F->getContext(), "for.end", F);

  auto dst = F->arg_begin();
  dst->setName("dst");
  auto src = dst + 1;
  src->setName("src");
  auto M = src + 1;
  M->setName("M");
  auto N = M + 1;
  N->setName("N");
  auto LDA = N + 1;
  LDA->setName("LDA");

  {
    IRBuilder<> B(entry);
    Value *l = B.CreateAdd(M, N, "mul", true, true);
    // Don't copy a 0*0 matrix
    B.CreateCondBr(B.CreateICmpEQ(l, ConstantInt::get(IT, 0)), end, init);
  }

  PHINode *j;
  {
    IRBuilder<> B(init);
    j = B.CreatePHI(IT, 2, "j");
    j->addIncoming(ConstantInt::get(IT, 0), entry);
    B.CreateBr(body);
  }

  {
    IRBuilder<> B(body);
    PHINode *i = B.CreatePHI(IT, 2, "i");
    i->addIncoming(ConstantInt::get(IT, 0), init);

    Value *dsti = B.CreateInBoundsGEP(
        elementType, dst,
        B.CreateAdd(i, B.CreateMul(j, M, "", true, true), "", true, true),
        "dst.i");
    Value *srci = B.CreateInBoundsGEP(
        elementType, src,
        B.CreateAdd(i, B.CreateMul(j, LDA, "", true, true), "", true, true),
        "dst.i");
    LoadInst *srcl = B.CreateLoad(elementType, srci, "src.i.l");

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

    Value *nexti =
        B.CreateAdd(i, ConstantInt::get(IT, 1), "i.next", true, true);
    i->addIncoming(nexti, body);
    B.CreateCondBr(B.CreateICmpEQ(nexti, M), initend, body);
  }

  {
    IRBuilder<> B(initend);
    Value *nextj =
        B.CreateAdd(j, ConstantInt::get(IT, 1), "j.next", true, true);
    j->addIncoming(nextj, initend);
    B.CreateCondBr(B.CreateICmpEQ(nextj, N), end, init);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }

  return F;
}

// TODO implement differential memmove
Function *
getOrInsertDifferentialFloatMemmove(Module &M, Type *T, unsigned dstalign,
                                    unsigned srcalign, unsigned dstaddr,
                                    unsigned srcaddr, unsigned bitwidth) {
  llvm::errs() << "warning: didn't implement memmove, using memcpy as fallback "
                  "which can result in errors\n";
  return getOrInsertDifferentialFloatMemcpy(M, T, dstalign, srcalign, dstaddr,
                                            srcaddr, bitwidth);
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

  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

  if (!F->empty())
    return F;

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
#if LLVM_VERSION_MAJOR >= 16
  F->setOnlyAccessesArgMemory();
#else
  F->addFnAttr(Attribute::ArgMemOnly);
#endif
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

/// Create function to computer nearest power of two
llvm::Value *nextPowerOfTwo(llvm::IRBuilder<> &B, llvm::Value *V) {
  assert(V->getType()->isIntegerTy());
  IntegerType *T = cast<IntegerType>(V->getType());
  V = B.CreateAdd(V, ConstantInt::get(T, -1));
  for (size_t i = 1; i < T->getBitWidth(); i *= 2) {
    V = B.CreateOr(V, B.CreateLShr(V, ConstantInt::get(T, i)));
  }
  V = B.CreateAdd(V, ConstantInt::get(T, 1));
  return V;
}

llvm::Function *getOrInsertDifferentialWaitallSave(llvm::Module &M,
                                                   ArrayRef<llvm::Type *> T,
                                                   PointerType *reqType) {
  std::string name = "__enzyme_differential_waitall_save";
  FunctionType *FT =
      FunctionType::get(PointerType::getUnqual(reqType), T, false);
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

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

  Type *reqT = reqType; // req->getType()->getPointerElementType();
  Value *idxs[] = {idx};
  Value *ireq = B.CreateInBoundsGEP(reqT, req, idxs);
  Value *idreq = B.CreateInBoundsGEP(reqT, dreq, idxs);
  Value *iout = B.CreateInBoundsGEP(reqType, ret, idxs);
  Value *isNull = nullptr;
  if (auto GV = M.getNamedValue("ompi_request_null")) {
    Value *reql =
        B.CreatePointerCast(ireq, PointerType::getUnqual(GV->getType()));
    reql = B.CreateLoad(GV->getType(), reql);
    isNull = B.CreateICmpEQ(reql, GV);
  }

  idreq = B.CreatePointerCast(idreq, PointerType::getUnqual(reqType));
  Value *d_reqp = B.CreateLoad(reqType, idreq);
  if (isNull)
    d_reqp = B.CreateSelect(isNull, Constant::getNullValue(d_reqp->getType()),
                            d_reqp);

  B.CreateStore(d_reqp, iout);

  B.CreateCondBr(B.CreateICmpEQ(inc, count), endBlock, loopBlock);

  B.SetInsertPoint(endBlock);
  B.CreateRet(ret);
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
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

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
    irecvfn = cast<Function>(M.getOrInsertFunction(name, FuT).getCallee());
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

llvm::Value *getOrInsertOpFloatSum(llvm::Module &M, llvm::Type *OpPtr,
                                   llvm::Type *OpType, ConcreteType CT,
                                   llvm::Type *intType, IRBuilder<> &B2) {
  std::string name = "__enzyme_mpi_sum" + CT.str();
  assert(CT.isFloat());
  auto FlT = CT.isFloat();

  if (auto Glob = M.getGlobalVariable(name)) {
    return B2.CreateLoad(Glob->getValueType(), Glob);
  }

  llvm::Type *types[] = {PointerType::getUnqual(FlT),
                         PointerType::getUnqual(FlT),
                         PointerType::getUnqual(intType), OpPtr};
  FunctionType *FuT =
      FunctionType::get(Type::getVoidTy(M.getContext()), types, false);
  Function *F =
      cast<Function>(M.getOrInsertFunction(name + "_run", FuT).getCallee());

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
#if LLVM_VERSION_MAJOR >= 16
  F->setOnlyAccessesArgMemory();
#else
  F->addFnAttr(Attribute::ArgMemOnly);
#endif
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(0, Attribute::ReadOnly);
  F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(2, Attribute::NoCapture);
  F->addParamAttr(2, Attribute::ReadOnly);
  F->addParamAttr(3, Attribute::NoCapture);
  F->addParamAttr(3, Attribute::ReadNone);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto src = F->arg_begin();
  src->setName("src");
  auto dst = src + 1;
  dst->setName("dst");
  auto lenp = dst + 1;
  lenp->setName("lenp");
  Value *len;
  // TODO consider using datatype arg and asserting same size as assumed
  // by type analysis

  {
    IRBuilder<> B(entry);
    len = B.CreateLoad(intType, lenp);
    B.CreateCondBr(B.CreateICmpEQ(len, ConstantInt::get(len->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(len->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(len->getType(), 0), entry);

    Value *dsti = B.CreateInBoundsGEP(FlT, dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(FlT, dsti, "dst.i.l");

    Value *srci = B.CreateInBoundsGEP(FlT, src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(FlT, srci, "src.i.l");
    B.CreateStore(B.CreateFAdd(srcl, dstl), dsti);

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(len->getType(), 1), "idx.next");
    idx->addIncoming(next, body);
    B.CreateCondBr(B.CreateICmpEQ(len, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }

  llvm::Type *rtypes[] = {Type::getInt8PtrTy(M.getContext()), intType, OpPtr};
  FunctionType *RFT = FunctionType::get(intType, rtypes, false);

  Constant *RF = M.getNamedValue("MPI_Op_create");
  if (!RF) {
    RF =
        cast<Function>(M.getOrInsertFunction("MPI_Op_create", RFT).getCallee());
  } else {
    RF = ConstantExpr::getBitCast(RF, PointerType::getUnqual(RFT));
  }

  GlobalVariable *GV =
      new GlobalVariable(M, OpType, false, GlobalVariable::InternalLinkage,
                         UndefValue::get(OpType), name);

  Type *i1Ty = Type::getInt1Ty(M.getContext());
  GlobalVariable *initD = new GlobalVariable(
      M, i1Ty, false, GlobalVariable::InternalLinkage,
      ConstantInt::getFalse(M.getContext()), name + "_initd");

  // Finish initializing mpi sum
  // https://www.mpich.org/static/docs/v3.2/www3/MPI_Op_create.html
  FunctionType *IFT = FunctionType::get(Type::getVoidTy(M.getContext()),
                                        ArrayRef<Type *>(), false);
  Function *initializerFunction = cast<Function>(
      M.getOrInsertFunction(name + "initializer", IFT).getCallee());

  initializerFunction->setLinkage(Function::LinkageTypes::InternalLinkage);
  initializerFunction->addFnAttr(Attribute::NoUnwind);

  {
    BasicBlock *entry =
        BasicBlock::Create(M.getContext(), "entry", initializerFunction);
    BasicBlock *run =
        BasicBlock::Create(M.getContext(), "run", initializerFunction);
    BasicBlock *end =
        BasicBlock::Create(M.getContext(), "end", initializerFunction);
    IRBuilder<> B(entry);

    B.CreateCondBr(B.CreateLoad(initD->getValueType(), initD), end, run);

    B.SetInsertPoint(run);
    Value *args[] = {ConstantExpr::getPointerCast(F, rtypes[0]),
                     ConstantInt::get(rtypes[1], 1, false),
                     ConstantExpr::getPointerCast(GV, rtypes[2])};
    B.CreateCall(RFT, RF, args);
    B.CreateStore(ConstantInt::getTrue(M.getContext()), initD);
    B.CreateBr(end);
    B.SetInsertPoint(end);
    B.CreateRetVoid();
  }

  B2.CreateCall(M.getFunction(name + "initializer"));
  return B2.CreateLoad(GV->getValueType(), GV);
}

void mayExecuteAfter(llvm::SmallVectorImpl<llvm::Instruction *> &results,
                     llvm::Instruction *inst,
                     const llvm::SmallPtrSetImpl<Instruction *> &stores,
                     const llvm::Loop *region) {
  using namespace llvm;
  std::map<BasicBlock *, SmallVector<Instruction *, 1>> maybeBlocks;
  BasicBlock *instBlk = inst->getParent();
  for (auto store : stores) {
    BasicBlock *storeBlk = store->getParent();
    if (instBlk == storeBlk) {
      // if store doesn't come before, exit.

      if (store != inst) {
        BasicBlock::const_iterator It = storeBlk->begin();
        for (; &*It != store && &*It != inst; ++It)
          /*empty*/;
        // if inst comes first (e.g. before store) in the
        // block, return true
        if (&*It == inst) {
          results.push_back(store);
        }
      }
      maybeBlocks[storeBlk].push_back(store);
    } else {
      maybeBlocks[storeBlk].push_back(store);
    }
  }

  if (maybeBlocks.size() == 0)
    return;

  llvm::SmallVector<BasicBlock *, 2> todo;
  for (auto B : successors(instBlk)) {
    if (region && region->getHeader() == B) {
      continue;
    }
    todo.push_back(B);
  }

  SmallPtrSet<BasicBlock *, 2> seen;
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (seen.count(cur))
      continue;
    seen.insert(cur);
    auto found = maybeBlocks.find(cur);
    if (found != maybeBlocks.end()) {
      for (auto store : found->second)
        results.push_back(store);
      maybeBlocks.erase(found);
    }
    for (auto B : successors(cur)) {
      if (region && region->getHeader() == B) {
        continue;
      }
      todo.push_back(B);
    }
  }
}

bool overwritesToMemoryReadByLoop(
    llvm::ScalarEvolution &SE, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
    llvm::Instruction *maybeReader, const llvm::SCEV *LoadStart,
    const llvm::SCEV *LoadEnd, llvm::Instruction *maybeWriter,
    const llvm::SCEV *StoreStart, const llvm::SCEV *StoreEnd,
    llvm::Loop *scope) {
  // The store may either occur directly after the load in the current loop
  // nest, or prior to the load in a subsequent iteration of the loop nest
  // Generally:
  // L0 -> scope -> L1 -> L2 -> L3 -> load_L4 -> load_L5 ...  Load
  //                               \-> store_L4 -> store_L5 ... Store
  // We begin by finding the common ancestor of the two loops, which may
  // be none.
  Loop *anc = getAncestor(LI.getLoopFor(maybeReader->getParent()),
                          LI.getLoopFor(maybeWriter->getParent()));

  // The surrounding scope must contain the ancestor
  if (scope) {
    assert(anc);
    assert(scope == anc || scope->contains(anc));
  }

  // Consider the case where the load and store don't share any common loops.
  // That is to say, there's no loops in [scope, ancestor) we need to consider
  // having a store in a  later iteration overwrite the load of a previous
  // iteration.
  //
  // An example of this overwriting would be a "left shift"
  //   for (int j = 1; j<N; j++) {
  //      load A[j]
  //      store A[j-1]
  //    }
  //
  // Ignoring such ancestors, if we compare the two regions to have no direct
  // overlap we can return that it doesn't overwrite memory if the regions
  // don't overlap at any level of region expansion. That is to say, we can
  // expand the start or end, for any loop to be the worst case scenario
  // given the loop bounds.
  //
  // However, now let us consider the case where there are surrounding loops.
  // If the storing boundary is represented by an induction variable of one
  // of these common loops, we must conseratively expand it all the way to the
  // end. We will also mark the loops we may expand. If we encounter all
  // intervening loops in this fashion, and it is proven safe in these cases,
  // the region does not overlap. However, if we don't encounter all surrounding
  // loops in our induction expansion, we may simply be repeating the write
  // which we should also ensure we say the region may overlap (due to the
  // repetition).
  //
  // Since we also have a Loop scope, we can ignore any common loops at the
  // scope level or above

  /// We force all ranges for all loops in range ... [scope, anc], .... cur
  /// to expand the number of iterations

  SmallPtrSet<const Loop *, 1> visitedAncestors;
  auto skipLoop = [&](const Loop *L) {
    assert(L);
    if (scope && L->contains(scope))
      return false;

    if (anc && (anc == L || anc->contains(L))) {
      visitedAncestors.insert(L);
      return true;
    }
    return false;
  };

  // Check the boounds  of an [... endprev][startnext ...] for potential
  // overlaps. The boolean EndIsStore is true of the EndPev represents
  // the store and should have its loops expanded, or if that should
  // apply to StartNed.
  auto hasOverlap = [&](const SCEV *EndPrev, const SCEV *StartNext,
                        bool EndIsStore) {
    for (auto slim = StartNext; slim != SE.getCouldNotCompute();) {
      bool sskip = false;
      if (!EndIsStore)
        if (auto startL = dyn_cast<SCEVAddRecExpr>(slim))
          if (skipLoop(startL->getLoop()) &&
              SE.isKnownNonPositive(startL->getStepRecurrence(SE))) {
            sskip = true;
          }

      if (!sskip)
        for (auto elim = EndPrev; elim != SE.getCouldNotCompute();) {
          {

            bool eskip = false;
            if (EndIsStore)
              if (auto endL = dyn_cast<SCEVAddRecExpr>(elim)) {
                if (skipLoop(endL->getLoop()) &&
                    SE.isKnownNonNegative(endL->getStepRecurrence(SE))) {
                  eskip = true;
                }
              }

            // Moreover because otherwise SE cannot "groupScevByComplexity"
            // we need to ensure that if both slim/elim are AddRecv
            // they must be in the same loop, or one loop must dominate
            // the other.
            if (!eskip) {

              if (auto endL = dyn_cast<SCEVAddRecExpr>(elim)) {
                auto EH = endL->getLoop()->getHeader();
                if (auto startL = dyn_cast<SCEVAddRecExpr>(slim)) {
                  auto SH = startL->getLoop()->getHeader();
                  if (EH != SH && !DT.dominates(EH, SH) &&
                      !DT.dominates(SH, EH))
                    eskip = true;
                }
              }
            }
            if (!eskip) {
              auto sub = SE.getMinusSCEV(slim, elim);
              if (sub != SE.getCouldNotCompute() && SE.isKnownNonNegative(sub))
                return false;
            }
          }

          if (auto endL = dyn_cast<SCEVAddRecExpr>(elim)) {
            if (SE.isKnownNonPositive(endL->getStepRecurrence(SE))) {
              elim = endL->getStart();
              continue;
            } else if (SE.isKnownNonNegative(endL->getStepRecurrence(SE))) {
#if LLVM_VERSION_MAJOR >= 12
              auto ebd = SE.getSymbolicMaxBackedgeTakenCount(endL->getLoop());
#else
              auto ebd = SE.getBackedgeTakenCount(endL->getLoop());
#endif
              if (ebd == SE.getCouldNotCompute())
                break;
              elim = endL->evaluateAtIteration(ebd, SE);
              continue;
            }
          }
          break;
        }

      if (auto startL = dyn_cast<SCEVAddRecExpr>(slim)) {
        if (SE.isKnownNonNegative(startL->getStepRecurrence(SE))) {
          slim = startL->getStart();
          continue;
        } else if (SE.isKnownNonPositive(startL->getStepRecurrence(SE))) {
#if LLVM_VERSION_MAJOR >= 12
          auto sbd = SE.getSymbolicMaxBackedgeTakenCount(startL->getLoop());
#else
          auto sbd = SE.getBackedgeTakenCount(startL->getLoop());
#endif
          if (sbd == SE.getCouldNotCompute())
            break;
          slim = startL->evaluateAtIteration(sbd, SE);
          continue;
        }
      }
      break;
    }
    return true;
  };

  // There is no overwrite if either the stores all occur before the loads
  // [S, S+Size][start load, L+Size]
  visitedAncestors.clear();
  if (!hasOverlap(StoreEnd, LoadStart, /*EndIsStore*/ true)) {
    // We must have seen all common loops as induction variables
    // to be legal, lest we have a repetition of the store.
    bool legal = true;
    for (const Loop *L = anc; anc != scope; anc = anc->getParentLoop()) {
      if (!visitedAncestors.count(L))
        legal = false;
    }
    if (legal)
      return false;
  }

  // There is no overwrite if either the loads all occur before the stores
  // [start load, L+Size] [S, S+Size]
  visitedAncestors.clear();
  if (!hasOverlap(LoadEnd, StoreStart, /*EndIsStore*/ false)) {
    // We must have seen all common loops as induction variables
    // to be legal, lest we have a repetition of the store.
    bool legal = true;
    for (const Loop *L = anc; anc != scope; anc = anc->getParentLoop()) {
      if (!visitedAncestors.count(L))
        legal = false;
    }
    if (legal)
      return false;
  }
  return true;
}

bool overwritesToMemoryReadBy(llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI,
                              ScalarEvolution &SE, llvm::LoopInfo &LI,
                              llvm::DominatorTree &DT,
                              llvm::Instruction *maybeReader,
                              llvm::Instruction *maybeWriter,
                              llvm::Loop *scope) {
  using namespace llvm;
  if (!writesToMemoryReadBy(AA, TLI, maybeReader, maybeWriter))
    return false;
  const SCEV *LoadBegin = SE.getCouldNotCompute();
  const SCEV *LoadEnd = SE.getCouldNotCompute();

  const SCEV *StoreBegin = SE.getCouldNotCompute();
  const SCEV *StoreEnd = SE.getCouldNotCompute();

  if (auto LI = dyn_cast<LoadInst>(maybeReader)) {
    LoadBegin = SE.getSCEV(LI->getPointerOperand());
    if (LoadBegin != SE.getCouldNotCompute()) {
      auto &DL = maybeWriter->getModule()->getDataLayout();
      auto width = cast<IntegerType>(DL.getIndexType(LoadBegin->getType()))
                       ->getBitWidth();
#if LLVM_VERSION_MAJOR >= 10
      auto TS = SE.getConstant(
          APInt(width, DL.getTypeStoreSize(LI->getType()).getFixedSize()));
#else
      auto TS =
          SE.getConstant(APInt(width, DL.getTypeStoreSize(LI->getType())));
#endif
      LoadEnd = SE.getAddExpr(LoadBegin, TS);
    }
  }
  if (auto SI = dyn_cast<StoreInst>(maybeWriter)) {
    StoreBegin = SE.getSCEV(SI->getPointerOperand());
    if (StoreBegin != SE.getCouldNotCompute()) {
      auto &DL = maybeWriter->getModule()->getDataLayout();
      auto width = cast<IntegerType>(DL.getIndexType(StoreBegin->getType()))
                       ->getBitWidth();
#if LLVM_VERSION_MAJOR >= 10
      auto TS = SE.getConstant(
          APInt(width, DL.getTypeStoreSize(SI->getValueOperand()->getType())
                           .getFixedSize()));
#else
      auto TS = SE.getConstant(
          APInt(width, DL.getTypeStoreSize(SI->getValueOperand()->getType())));
#endif
      StoreEnd = SE.getAddExpr(StoreBegin, TS);
    }
  }
  if (auto MS = dyn_cast<MemSetInst>(maybeWriter)) {
    StoreBegin = SE.getSCEV(MS->getArgOperand(0));
    if (StoreBegin != SE.getCouldNotCompute()) {
      if (auto Len = dyn_cast<ConstantInt>(MS->getArgOperand(2))) {
        auto &DL = MS->getModule()->getDataLayout();
        auto width = cast<IntegerType>(DL.getIndexType(StoreBegin->getType()))
                         ->getBitWidth();
        auto TS =
            SE.getConstant(APInt(width, Len->getValue().getLimitedValue()));
        StoreEnd = SE.getAddExpr(StoreBegin, TS);
      }
    }
  }
  if (auto MS = dyn_cast<MemTransferInst>(maybeWriter)) {
    StoreBegin = SE.getSCEV(MS->getArgOperand(0));
    if (StoreBegin != SE.getCouldNotCompute()) {
      if (auto Len = dyn_cast<ConstantInt>(MS->getArgOperand(2))) {
        auto &DL = MS->getModule()->getDataLayout();
        auto width = cast<IntegerType>(DL.getIndexType(StoreBegin->getType()))
                         ->getBitWidth();
        auto TS =
            SE.getConstant(APInt(width, Len->getValue().getLimitedValue()));
        StoreEnd = SE.getAddExpr(StoreBegin, TS);
      }
    }
  }
  if (auto MS = dyn_cast<MemTransferInst>(maybeReader)) {
    LoadBegin = SE.getSCEV(MS->getArgOperand(1));
    if (LoadBegin != SE.getCouldNotCompute()) {
      if (auto Len = dyn_cast<ConstantInt>(MS->getArgOperand(2))) {
        auto &DL = MS->getModule()->getDataLayout();
        auto width = cast<IntegerType>(DL.getIndexType(LoadBegin->getType()))
                         ->getBitWidth();
        auto TS =
            SE.getConstant(APInt(width, Len->getValue().getLimitedValue()));
        LoadEnd = SE.getAddExpr(LoadBegin, TS);
      }
    }
  }

  if (!overwritesToMemoryReadByLoop(SE, LI, DT, maybeReader, LoadBegin, LoadEnd,
                                    maybeWriter, StoreBegin, StoreEnd, scope))
    return false;

  return true;
}

/// Return whether maybeReader can read from memory written to by maybeWriter
bool writesToMemoryReadBy(llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI,
                          llvm::Instruction *maybeReader,
                          llvm::Instruction *maybeWriter) {
  assert(maybeReader->getParent()->getParent() ==
         maybeWriter->getParent()->getParent());
  using namespace llvm;
  if (isa<StoreInst>(maybeReader))
    return false;
  if (auto call = dyn_cast<CallInst>(maybeWriter)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isCertainPrint(funcName) || isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }
    if (funcName == "jl_array_copy" || funcName == "ijl_array_copy")
      return false;

    // Isend only writes to inaccessible mem only
    if (funcName == "MPI_Send" || funcName == "PMPI_Send") {
      return false;
    }
    // Wait only overwrites memory in the status and request.
    if (funcName == "MPI_Wait" || funcName == "PMPI_Wait" ||
        funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
#if LLVM_VERSION_MAJOR > 11
      auto loc = LocationSize::afterPointer();
#else
      auto loc = MemoryLocation::UnknownSize;
#endif
      size_t off = (funcName == "MPI_Wait" || funcName == "PMPI_Wait") ? 0 : 1;
      // No alias with status
      if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(off + 1),
                                     loc))) {
        // No alias with request
        if (!isRefSet(AA.getModRefInfo(maybeReader,
                                       call->getArgOperand(off + 0), loc)))
          return false;
        auto R = parseTBAA(*maybeReader, maybeReader->getParent()
                                             ->getParent()
                                             ->getParent()
                                             ->getDataLayout())[{-1}];
        // Could still conflict with the mpi_request unless a non pointer
        // type.
        if (R != BaseType::Unknown && R != BaseType::Anything &&
            R != BaseType::Pointer)
          return false;
      }
    }
    // Isend only writes to inaccessible mem and request.
    if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
      auto R = parseTBAA(*maybeReader, maybeReader->getParent()
                                           ->getParent()
                                           ->getParent()
                                           ->getDataLayout())[{-1}];
      // Could still conflict with the mpi_request, unless either
      // synchronous, or a non pointer type.
      if (R != BaseType::Unknown && R != BaseType::Anything &&
          R != BaseType::Pointer)
        return false;
#if LLVM_VERSION_MAJOR > 11
      if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                     LocationSize::afterPointer())))
        return false;
#else
      if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                     MemoryLocation::UnknownSize)))
        return false;
#endif
      return false;
    }
    if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv" ||
        funcName == "MPI_Recv" || funcName == "PMPI_Recv") {
      ConcreteType type(BaseType::Unknown);
      if (Constant *C = dyn_cast<Constant>(call->getArgOperand(2))) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_double") {
            type = ConcreteType(Type::getDoubleTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_float") {
            type = ConcreteType(Type::getFloatTy(C->getContext()));
          }
        }
      }
      if (type.isKnown()) {
        auto R = parseTBAA(*maybeReader, maybeReader->getParent()
                                             ->getParent()
                                             ->getParent()
                                             ->getDataLayout())[{-1}];
        if (R.isKnown() && type != R) {
          // Could still conflict with the mpi_request, unless either
          // synchronous, or a non pointer type.
          if (funcName == "MPI_Recv" || funcName == "PMPI_Recv" ||
              (R != BaseType::Anything && R != BaseType::Pointer))
            return false;
#if LLVM_VERSION_MAJOR > 11
          if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                         LocationSize::afterPointer())))
            return false;
#else
          if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                         MemoryLocation::UnknownSize)))
            return false;
#endif
        }
      }
    }
    if (auto II = dyn_cast<IntrinsicInst>(call)) {
      if (II->getIntrinsicID() == Intrinsic::stacksave)
        return false;
      if (II->getIntrinsicID() == Intrinsic::stackrestore)
        return false;
      if (II->getIntrinsicID() == Intrinsic::trap)
        return false;
#if LLVM_VERSION_MAJOR >= 13
      if (II->getIntrinsicID() == Intrinsic::experimental_noalias_scope_decl)
        return false;
#endif
    }

#if LLVM_VERSION_MAJOR >= 11
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand()))
#else
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue()))
#endif
    {
      if (StringRef(iasm->getAsmString()).contains("exit"))
        return false;
    }
  }
  if (auto call = dyn_cast<CallInst>(maybeReader)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }

    if (auto II = dyn_cast<IntrinsicInst>(call)) {
      if (II->getIntrinsicID() == Intrinsic::stacksave)
        return false;
      if (II->getIntrinsicID() == Intrinsic::stackrestore)
        return false;
      if (II->getIntrinsicID() == Intrinsic::trap)
        return false;
#if LLVM_VERSION_MAJOR >= 13
      if (II->getIntrinsicID() == Intrinsic::experimental_noalias_scope_decl)
        return false;
#endif
    }
  }
  if (auto call = dyn_cast<InvokeInst>(maybeWriter)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }
    if (funcName == "jl_array_copy" || funcName == "ijl_array_copy")
      return false;

#if LLVM_VERSION_MAJOR >= 11
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand()))
#else
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue()))
#endif
    {
      if (StringRef(iasm->getAsmString()).contains("exit"))
        return false;
    }
  }
  if (auto call = dyn_cast<InvokeInst>(maybeReader)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }
  }
  assert(maybeWriter->mayWriteToMemory());
  assert(maybeReader->mayReadFromMemory());

  if (auto li = dyn_cast<LoadInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(li)));
  }
  if (auto rmw = dyn_cast<AtomicRMWInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(rmw)));
  }
  if (auto xch = dyn_cast<AtomicCmpXchgInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(xch)));
  }
  if (auto mti = dyn_cast<MemTransferInst>(maybeReader)) {
    return isModSet(
        AA.getModRefInfo(maybeWriter, MemoryLocation::getForSource(mti)));
  }

  if (auto si = dyn_cast<StoreInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(si)));
  }
  if (auto rmw = dyn_cast<AtomicRMWInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(rmw)));
  }
  if (auto xch = dyn_cast<AtomicCmpXchgInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(xch)));
  }
  if (auto mti = dyn_cast<MemIntrinsic>(maybeWriter)) {
    return isRefSet(
        AA.getModRefInfo(maybeReader, MemoryLocation::getForDest(mti)));
  }

  if (auto cb = dyn_cast<CallInst>(maybeReader)) {
    return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
  }
  if (auto cb = dyn_cast<InvokeInst>(maybeReader)) {
    return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
  }
  llvm::errs() << " maybeReader: " << *maybeReader
               << " maybeWriter: " << *maybeWriter << "\n";
  llvm_unreachable("unknown inst2");
}

Function *GetFunctionFromValue(Value *fn) {
  while (!isa<Function>(fn)) {
    if (auto ci = dyn_cast<CastInst>(fn)) {
      fn = ci->getOperand(0);
      continue;
    }
    if (auto ci = dyn_cast<ConstantExpr>(fn)) {
      if (ci->isCast()) {
        fn = ci->getOperand(0);
        continue;
      }
    }
    if (auto ci = dyn_cast<BlockAddress>(fn)) {
      fn = ci->getFunction();
      continue;
    }
    if (auto *GA = dyn_cast<GlobalAlias>(fn)) {
      fn = GA->getAliasee();
      continue;
    }
    if (auto *Call = dyn_cast<CallInst>(fn)) {
      if (auto F = Call->getCalledFunction()) {
        SmallPtrSet<Value *, 1> ret;
        for (auto &BB : *F) {
          if (auto RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
            ret.insert(RI->getReturnValue());
          }
        }
        if (ret.size() == 1) {
          auto val = *ret.begin();
          if (isa<Constant>(val)) {
            fn = val;
            continue;
          }
          if (auto arg = dyn_cast<Argument>(val)) {
            fn = Call->getArgOperand(arg->getArgNo());
            continue;
          }
        }
      }
    }
    if (auto *Call = dyn_cast<InvokeInst>(fn)) {
      if (auto F = Call->getCalledFunction()) {
        SmallPtrSet<Value *, 1> ret;
        for (auto &BB : *F) {
          if (auto RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
            ret.insert(RI->getReturnValue());
          }
        }
        if (ret.size() == 1) {
          auto val = *ret.begin();
          if (isa<Constant>(val)) {
            fn = val;
            continue;
          }
          if (auto arg = dyn_cast<Argument>(val)) {
            fn = Call->getArgOperand(arg->getArgNo());
            continue;
          }
        }
      }
    }
    break;
  }

  return dyn_cast<Function>(fn);
}

#if LLVM_VERSION_MAJOR >= 16
std::optional<BlasInfo> extractBLAS(llvm::StringRef in)
#else
llvm::Optional<BlasInfo> extractBLAS(llvm::StringRef in)
#endif
{
  llvm::Twine floatType[] = {"s", "d"}; // c, z
  llvm::Twine extractable[] = {"dot", "scal", "axpy", "gemv", "gemm", "spmv"};
  llvm::Twine prefixes[] = {"" /*Fortran*/, "cblas_", "cublas_"};
  llvm::Twine suffixes[] = {"", "_", "64_", "_64_"};
  for (auto t : floatType) {
    for (auto f : extractable) {
      for (auto p : prefixes) {
        for (auto s : suffixes) {
          if (in == (p + t + f + s).str()) {
            return BlasInfo{
                t.getSingleStringRef(),
                p.getSingleStringRef(),
                s.getSingleStringRef(),
                f.getSingleStringRef(),
            };
          }
        }
      }
    }
  }
  return {};
}

llvm::Constant *getUndefinedValueForType(llvm::Type *T, bool forceZero) {
  if (EnzymeUndefinedValueForType)
    return cast<Constant>(
        unwrap(EnzymeUndefinedValueForType(wrap(T), forceZero)));
  else if (EnzymeZeroCache || forceZero)
    return Constant::getNullValue(T);
  else
    return UndefValue::get(T);
}

llvm::Value *SanitizeDerivatives(llvm::Value *val, llvm::Value *toset,
                                 llvm::IRBuilder<> &BuilderM,
                                 llvm::Value *mask) {
  if (EnzymeSanitizeDerivatives)
    return unwrap(EnzymeSanitizeDerivatives(wrap(val), wrap(toset),
                                            wrap(&BuilderM), wrap(mask)));
  return toset;
}

llvm::FastMathFlags getFast() {
  llvm::FastMathFlags f;
  if (EnzymeFastMath)
    f.set();
  return f;
}

void addValueToCache(llvm::Value *arg, bool cache_arg, llvm::Type *ty,
                     llvm::SmallVectorImpl<llvm::Value *> &cacheValues,
                     llvm::IRBuilder<> &BuilderZ, const Twine &name) {
  if (!arg->getType()->isPointerTy()) {
    assert(arg->getType() == ty);
    cacheValues.push_back(arg);
    return;
  }
  if (!cache_arg)
    return;
#if LLVM_VERSION_MAJOR < 18
  auto PT = cast<PointerType>(arg->getType());
#if LLVM_VERSION_MAJOR <= 14
  if (PT->getElementType() != ty)
    arg = BuilderZ.CreatePointerCast(
        arg, PointerType::get(ty, PT->getAddressSpace()), "pcld." + name);
#else
  auto PT2 = PointerType::get(ty, PT->getAddressSpace());
  if (!PT->isOpaqueOrPointeeTypeMatches(PT2))
    arg = BuilderZ.CreatePointerCast(
        arg, PointerType::get(ty, PT->getAddressSpace()), "pcld." + name);
#endif
#endif
  arg = BuilderZ.CreateLoad(ty, arg, "avld." + name);
  cacheValues.push_back(arg);
}

// julia_decl null means not julia decl, otherwise it is the integer type needed
// to cast to
llvm::Value *to_blas_callconv(IRBuilder<> &B, llvm::Value *V, bool byRef,
                              IntegerType *julia_decl,
                              IRBuilder<> &entryBuilder,
                              llvm::Twine const &name) {
  if (!byRef)
    return V;

  Value *allocV =
      entryBuilder.CreateAlloca(V->getType(), nullptr, "byref." + name);
  B.CreateStore(V, allocV);

  if (julia_decl)
    allocV = B.CreatePointerCast(allocV, Type::getInt8PtrTy(V->getContext()),
                                 "intcast." + name);

  return allocV;
}
llvm::Value *to_blas_fp_callconv(IRBuilder<> &B, llvm::Value *V, bool byRef,
                                 Type *fpTy, IRBuilder<> &entryBuilder,
                                 llvm::Twine const &name) {
  if (!byRef)
    return V;

  Value *allocV =
      entryBuilder.CreateAlloca(V->getType(), nullptr, "byref." + name);
  B.CreateStore(V, allocV);

  if (fpTy)
    allocV = B.CreatePointerCast(allocV, fpTy, "fpcast." + name);

  return allocV;
}

llvm::Value *select_vec_dims(IRBuilder<> &B, llvm::Value *trans,
                             llvm::Value *dim1, llvm::Value *dim2, bool byRef) {
  Value *width = B.CreateSelect(is_normal(B, trans, byRef), dim1, dim2);

  return width;
}

Value *is_uper(IRBuilder<> &B, Value *trans, bool byRef) {
  auto charTy = IntegerType::get(trans->getContext(), 8);
  if (byRef)
    trans = B.CreateLoad(charTy, trans, "loaded.trans");

  Value *trueVal = ConstantInt::getTrue(trans->getContext());

  Value *isUper =
      B.CreateOr(B.CreateICmpEQ(trans, ConstantInt::get(charTy, 'u')),
                 B.CreateICmpEQ(trans, ConstantInt::get(charTy, 'U')));
  return isUper;
}

llvm::Value *is_normal(IRBuilder<> &B, llvm::Value *trans, bool byRef) {
  auto charTy = IntegerType::get(trans->getContext(), 8);
  if (byRef)
    trans = B.CreateLoad(charTy, trans, "loaded.trans");

  Value *trueVal = ConstantInt::getTrue(trans->getContext());

  Value *isNormal =
      B.CreateOr(B.CreateICmpEQ(trans, ConstantInt::get(charTy, 'n')),
                 B.CreateICmpEQ(trans, ConstantInt::get(charTy, 'N')));
  return isNormal;
}

llvm::Value *transpose(IRBuilder<> &B, llvm::Value *V) {
  Value *out = B.CreateSelect(
      B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'T')),
      ConstantInt::get(V->getType(), 'N'),
      B.CreateSelect(
          B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 't')),
          ConstantInt::get(V->getType(), 'n'),
          B.CreateSelect(
              B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'N')),
              ConstantInt::get(V->getType(), 'T'),
              B.CreateSelect(
                  B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'n')),
                  ConstantInt::get(V->getType(), 't'),
                  ConstantInt::get(V->getType(), 0)))));
  return out;
}

// Implement the following logic to get the width of a matrix
// if (cache_A) {
//   ld_A = (arg_transa == 'N') ? arg_m : arg_k;
// } else {
//   ld_A = arg_lda;
// }
llvm::Value *get_cached_mat_width(llvm::IRBuilder<> &B, llvm::Value *trans,
                                  llvm::Value *arg_ld, llvm::Value *dim1,
                                  llvm::Value *dim2, bool cacheMat,
                                  bool byRef) {
  if (!cacheMat)
    return arg_ld;

  Value *width = B.CreateSelect(is_normal(B, trans, byRef), dim1, dim2);

  return width;
}

llvm::Value *transpose(llvm::IRBuilder<> &B, llvm::Value *V, bool byRef,
                       llvm::IntegerType *julia_decl,
                       llvm::IRBuilder<> &entryBuilder,
                       const llvm::Twine &name) {

  if (byRef) {
    auto charType = IntegerType::get(V->getContext(), 8);
    V = B.CreateLoad(charType, V, "ld." + name);
  }

  V = transpose(B, V);

  return to_blas_callconv(B, V, byRef, julia_decl, entryBuilder,
                          "transpose." + name);
}

llvm::Value *load_if_ref(llvm::IRBuilder<> &B, llvm::IntegerType *intType,
                         llvm::Value *V, bool byRef) {
  if (!byRef)
    return V;

  auto VP = B.CreatePointerCast(
      V, PointerType::get(intType,
                          cast<PointerType>(V->getType())->getAddressSpace()));
  return B.CreateLoad(intType, VP);
}

llvm::Value *get_blas_row(llvm::IRBuilder<> &B, llvm::Value *trans,
                          llvm::Value *row, llvm::Value *col, bool byRef) {

  if (byRef) {
    auto charType = IntegerType::get(trans->getContext(), 8);
    trans = B.CreateLoad(charType, trans, "ld.row.trans");
  }

  return B.CreateSelect(
      B.CreateOr(
          B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'N')),
          B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'n'))),
      row, col);
}

// return how many Special pointers are in T (count > 0),
// and if there is anything else in T (all == false)
CountTrackedPointers::CountTrackedPointers(Type *T) {
  if (isa<PointerType>(T)) {
    if (isSpecialPtr(T)) {
      count++;
      if (T->getPointerAddressSpace() != AddressSpace::Tracked)
        derived = true;
    }
  } else if (isa<StructType>(T) || isa<ArrayType>(T) || isa<VectorType>(T)) {
    for (Type *ElT : T->subtypes()) {
      auto sub = CountTrackedPointers(ElT);
      count += sub.count;
      all &= sub.all;
      derived |= sub.derived;
    }
    if (isa<ArrayType>(T))
      count *= cast<ArrayType>(T)->getNumElements();
    else if (isa<VectorType>(T)) {
#if LLVM_VERSION_MAJOR >= 12
      count *= cast<VectorType>(T)->getElementCount().getKnownMinValue();
#else
      count *= cast<VectorType>(T)->getNumElements();
#endif
    }
  }
  if (count == 0)
    all = false;
}
