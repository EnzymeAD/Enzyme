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
#include "GradientUtils.h"
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
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"

#if LLVM_VERSION_MAJOR >= 16
#include "llvm/TargetParser/Triple.h"
#else
#include "llvm/ADT/Triple.h"
#endif

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
LLVMValueRef (*EnzymeUndefinedValueForType)(LLVMModuleRef, LLVMTypeRef,
                                            uint8_t) = nullptr;

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
llvm::cl::opt<bool> EnzymeMemmoveWarning(
    "enzyme-memmove-warning", cl::init(true), cl::Hidden,
    cl::desc("Warn if using memmove implementation as a fallback for memmove"));
llvm::cl::opt<bool> EnzymeRuntimeError(
    "enzyme-runtime-error", cl::init(false), cl::Hidden,
    cl::desc("Emit Runtime errors instead of compile time ones"));

llvm::cl::opt<bool> EnzymeNonPower2Cache(
    "enzyme-non-power2-cache", cl::init(false), cl::Hidden,
    cl::desc("Disable caching of integers which are not a power of 2"));
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
  return getInt8PtrTy(C);
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
  AttributeList AL;
  if (newFunc->hasFnAttribute("enzymejl_world")) {
    AL = AL.addFnAttribute(newFunc->getContext(),
                           newFunc->getFnAttribute("enzymejl_world"));
  }
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT, AL).getCallee());

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
  auto popCnt = getIntrinsicDeclaration(&M, Intrinsic::ctpop, {types[1]});

  B.CreateCondBr(
      B.CreateAnd(B.CreateICmpULT(B.CreateCall(popCnt, {size}),
                                  ConstantInt::get(types[1], 3, false)),
                  hasOne),
      grow, ok);

  B.SetInsertPoint(grow);

  auto lz =
      B.CreateCall(getIntrinsicDeclaration(&M, Intrinsic::ctlz, {types[1]}),
                   {size, ConstantInt::getTrue(M.getContext())});
  Value *next =
      B.CreateShl(tsize, B.CreateSub(ConstantInt::get(types[1], 64, false), lz,
                                     "", true, true));

  Value *gVal;

  Value *prevSize =
      B.CreateSelect(B.CreateICmpEQ(size, ConstantInt::get(size->getType(), 1)),
                     ConstantInt::get(next->getType(), 0),
                     B.CreateLShr(next, ConstantInt::get(next->getType(), 1)));

  auto Arch = llvm::Triple(M.getTargetTriple()).getArch();
  bool forceMalloc = Arch == Triple::nvptx || Arch == Triple::nvptx64;

  if (!custom && !forceMalloc) {
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
    auto memsetF = getIntrinsicDeclaration(&M, Intrinsic::memcpy, tys);
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
    auto memsetF = getIntrinsicDeclaration(&M, Intrinsic::memset, tys);
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
    if (isa<UndefValue>(res))
      return res;
    if (isa<Constant>(res))
      return res;
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
#if LLVM_VERSION_MAJOR > 17
    res =
        Builder.CreateMalloc(Count->getType(), T, Align, Count, nullptr, Name);
#else
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
#endif

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
#if LLVM_VERSION_MAJOR < 17
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
        getIntrinsicDeclaration(&M, Intrinsic::memset, tys), args);
  }
  return res;
}

CallInst *CreateDealloc(llvm::IRBuilder<> &Builder, llvm::Value *ToFree) {
  CallInst *res = nullptr;

  if (CustomDeallocator) {
    res = dyn_cast_or_null<CallInst>(
        unwrap(CustomDeallocator(wrap(&Builder), wrap(ToFree))));
  } else {

    ToFree =
        Builder.CreatePointerCast(ToFree, getInt8PtrTy(ToFree->getContext()));
#if LLVM_VERSION_MAJOR > 17
    res = cast<CallInst>(Builder.CreateFree(ToFree));
#else
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
#endif
#if LLVM_VERSION_MAJOR >= 14
    res->addAttributeAtIndex(AttributeList::FirstArgIndex, Attribute::NonNull);
#else
    res->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
#endif
  }
  return res;
}

EnzymeWarning::EnzymeWarning(const llvm::Twine &RemarkName,
                             const llvm::DiagnosticLocation &Loc,
                             const llvm::Instruction *CodeRegion)
    : EnzymeWarning(RemarkName, Loc, CodeRegion->getParent()->getParent()) {}

EnzymeWarning::EnzymeWarning(const llvm::Twine &RemarkName,
                             const llvm::DiagnosticLocation &Loc,
                             const llvm::Function *CodeRegion)
    : DiagnosticInfoUnsupported(*CodeRegion, RemarkName, Loc, DS_Warning) {}

EnzymeFailure::EnzymeFailure(const llvm::Twine &RemarkName,
                             const llvm::DiagnosticLocation &Loc,
                             const llvm::Instruction *CodeRegion)
    : EnzymeFailure(RemarkName, Loc, CodeRegion->getParent()->getParent()) {}

EnzymeFailure::EnzymeFailure(const llvm::Twine &RemarkName,
                             const llvm::DiagnosticLocation &Loc,
                             const llvm::Function *CodeRegion)
    : DiagnosticInfoUnsupported(*CodeRegion, RemarkName, Loc) {}

/// Convert a floating type to a string
static inline std::string tofltstr(Type *T) {
  if (auto VT = dyn_cast<VectorType>(T)) {
#if LLVM_VERSION_MAJOR >= 12
    auto len = VT->getElementCount().getFixedValue();
#else
    auto len = VT->getNumElements();
#endif
    return "vec" + std::to_string(len) + tofltstr(VT->getElementType());
  }
  switch (T->getTypeID()) {
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::X86_FP80TyID:
    return "x87d";
  case Type::BFloatTyID:
    return "bf16";
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
                                       {getInt8PtrTy(M.getContext()),
                                        getInt8PtrTy(M.getContext()),
                                        getInt8PtrTy(M.getContext())},
                                       false);

  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());

  if (F->empty()) {
    F->setLinkage(Function::LinkageTypes::InternalLinkage);
    F->addFnAttr(Attribute::AlwaysInline);
    addFunctionNoCapture(F, 0);
    addFunctionNoCapture(F, 1);

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
                            {getInt8PtrTy(M.getContext())}, false);

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

  Value *args[] = {B.CreatePointerCast(primal, getInt8PtrTy(M.getContext())),
                   B.CreatePointerCast(shadow, getInt8PtrTy(M.getContext())),
                   getString(M, Message)};
  auto call = B.CreateCall(F, args);
  call->setDebugLoc(loc);
}

Type *BlasInfo::fpType(LLVMContext &ctx, bool to_scalar) const {
  if (floatType == "d" || floatType == "D") {
    return Type::getDoubleTy(ctx);
  } else if (floatType == "s" || floatType == "S") {
    return Type::getFloatTy(ctx);
  } else if (floatType == "c" || floatType == "C") {
    if (to_scalar)
      return Type::getFloatTy(ctx);
    return VectorType::get(Type::getFloatTy(ctx), 2, false);
  } else if (floatType == "z" || floatType == "Z") {
    if (to_scalar)
      return Type::getDoubleTy(ctx);
    return VectorType::get(Type::getDoubleTy(ctx), 2, false);
  } else {
    assert(false && "Unreachable");
    return nullptr;
  }
}

IntegerType *BlasInfo::intType(LLVMContext &ctx) const {
  if (is64)
    return IntegerType::get(ctx, 64);
  else
    return IntegerType::get(ctx, 32);
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
  addFunctionNoCapture(F, 0);
  addFunctionNoCapture(F, 1);

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

  auto elSize = (M.getDataLayout().getTypeSizeInBits(elementType) + 7) / 8;
  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(num->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(num->getType(), 0), entry);

    Value *dsti = B.CreateInBoundsGEP(elementType, dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(elementType, dsti, "dst.i.l");
    StoreInst *dsts = B.CreateStore(Constant::getNullValue(elementType), dsti);

    if (dstalign) {
      // If the element size is already aligned to current alignment, do nothing
      // e.g. elsize = double = 8, dstalign = 2
      if (elSize % dstalign == 0) {

      } else if (dstalign % elSize == 0) {
        // Otherwise if the dst alignment is a multiple of the element size,
        // use the element size as the new alignment. e.g. elsize = double = 8
        // and alignment = 16
        dstalign = elSize;
      } else {
        // else alignment only applies for first element, and we lose after all
        // other iterattions, assume nothing
        dstalign = 1;
      }
    }

    if (srcalign) {
      // If the element size is already aligned to current alignment, do nothing
      // e.g. elsize = double = 8, dstalign = 2
      if (elSize % srcalign == 0) {

      } else if (srcalign % elSize == 0) {
        // Otherwise if the dst alignment is a multiple of the element size,
        // use the element size as the new alignment. e.g. elsize = double = 8
        // and alignment = 16
        srcalign = elSize;
      } else {
        // else alignment only applies for first element, and we lose after all
        // other iterattions, assume nothing
        srcalign = 1;
      }
    }

    if (dstalign) {
      dstl->setAlignment(Align(dstalign));
      dsts->setAlignment(Align(dstalign));
    }

    Value *srci = B.CreateInBoundsGEP(elementType, src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(elementType, srci, "src.i.l");
    StoreInst *srcs = B.CreateStore(B.CreateFAdd(srcl, dstl), srci);
    if (srcalign) {
      srcl->setAlignment(Align(srcalign));
      srcs->setAlignment(Align(srcalign));
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

Value *lookup_with_layout(IRBuilder<> &B, Type *fpType, Value *layout,
                          Value *const base, Value *lda, Value *row,
                          Value *col) {
  Type *intType = row->getType();
  Value *is_row_maj =
      layout ? B.CreateICmpEQ(layout, ConstantInt::get(layout->getType(), 101))
             : B.getFalse();
  Value *offset = nullptr;
  if (col) {
    offset = B.CreateMul(
        row, CreateSelect(B, is_row_maj, lda, ConstantInt::get(intType, 1)));
    offset = B.CreateAdd(
        offset,
        B.CreateMul(col, CreateSelect(B, is_row_maj,
                                      ConstantInt::get(intType, 1), lda)));
  } else {
    offset = B.CreateMul(row, lda);
  }
  if (!base)
    return offset;

  Value *ptr = base;
  if (base->getType()->isIntegerTy())
    ptr = B.CreateIntToPtr(ptr, PointerType::getUnqual(fpType));

#if LLVM_VERSION_MAJOR < 17
#if LLVM_VERSION_MAJOR >= 15
  if (ptr->getContext().supportsTypedPointers()) {
#endif
    if (fpType != ptr->getType()->getPointerElementType()) {
      ptr = B.CreatePointerCast(
          ptr,
          PointerType::get(
              fpType, cast<PointerType>(ptr->getType())->getAddressSpace()));
    }
#if LLVM_VERSION_MAJOR >= 15
  }
#endif
#endif
  ptr = B.CreateGEP(fpType, ptr, offset);

  if (base->getType()->isIntegerTy()) {
    ptr = B.CreatePtrToInt(ptr, base->getType());
  } else if (ptr->getType() != base->getType()) {
    ptr = B.CreatePointerCast(ptr, base->getType());
  }
  return ptr;
}

void copy_lower_to_upper(llvm::IRBuilder<> &B, llvm::Type *fpType,
                         BlasInfo blas, bool byRef, llvm::Value *layout,
                         llvm::Value *islower, llvm::Value *A, llvm::Value *lda,
                         llvm::Value *N) {

  const bool cublasv2 =
      blas.prefix == "cublas" && StringRef(blas.suffix).contains("v2");

  const bool cublas = blas.prefix == "cublas";
  auto &M = *B.GetInsertBlock()->getParent()->getParent();

  llvm::Type *intType = N->getType();
  // add spmv diag update call if not already present
  auto fnc_name = "__enzyme_copy_lower_to_upper" + blas.floatType +
                  blas.prefix + blas.suffix;

  SmallVector<Type *, 1> tys = {islower->getType(), A->getType(),
                                lda->getType(), N->getType()};
  if (layout)
    tys.insert(tys.begin(), layout->getType());
  auto ltuFT = FunctionType::get(B.getVoidTy(), tys, false);

  auto F0 = M.getOrInsertFunction(fnc_name, ltuFT);

  SmallVector<Value *, 1> args = {islower, A, lda, N};
  if (layout)
    args.insert(args.begin(), layout);
  auto C = B.CreateCall(F0, args);
  auto F = getFunctionFromCall(C);
  assert(F);
  if (!F->empty()) {
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
  if (A->getType()->isPointerTy())
    addFunctionNoCapture(F, 1 + ((bool)layout));

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *loop = BasicBlock::Create(M.getContext(), "loop", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto arg = F->arg_begin();
  Argument *layoutarg = nullptr;
  if (layout) {
    layoutarg = arg;
    layoutarg->setName("layout");
    arg++;
  }
  auto islowerarg = arg;
  islowerarg->setName("islower");
  arg++;
  auto Aarg = arg;
  Aarg->setName("A");
  arg++;
  auto ldaarg = arg;
  ldaarg->setName("lda");
  arg++;
  auto Narg = arg;
  Narg->setName("N");

  IRBuilder<> EB(entry);

  auto one = ConstantInt::get(intType, 1);
  auto zero = ConstantInt::get(intType, 0);

  Value *N_minus_1 = EB.CreateSub(Narg, one);

  IRBuilder<> LB(loop);

  auto i = LB.CreatePHI(intType, 2);
  i->addIncoming(zero, entry);
  auto i_plus_one = LB.CreateAdd(i, one, "", true, true);
  i->addIncoming(i_plus_one, loop);

  Value *copyArgs[] = {
      to_blas_callconv(LB, LB.CreateSub(N_minus_1, i), byRef, cublas, nullptr,
                       EB),
      lookup_with_layout(LB, fpType, layoutarg, Aarg, ldaarg,
                         CreateSelect(LB, islowerarg, i_plus_one, i),
                         CreateSelect(LB, islowerarg, i, i_plus_one)),
      to_blas_callconv(
          LB,
          lookup_with_layout(LB, fpType, layoutarg, nullptr, ldaarg,
                             CreateSelect(LB, islowerarg, one, zero),
                             CreateSelect(LB, islowerarg, zero, one)),
          byRef, cublas, nullptr, EB),
      lookup_with_layout(LB, fpType, layoutarg, Aarg, ldaarg,
                         CreateSelect(LB, islowerarg, i, i_plus_one),
                         CreateSelect(LB, islowerarg, i_plus_one, i)),
      to_blas_callconv(
          LB,
          lookup_with_layout(LB, fpType, layoutarg, nullptr, ldaarg,
                             CreateSelect(LB, islowerarg, zero, one),
                             CreateSelect(LB, islowerarg, one, zero)),
          byRef, cublas, nullptr, EB)};

  Type *copyTys[] = {copyArgs[0]->getType(), copyArgs[1]->getType(),
                     copyArgs[2]->getType(), copyArgs[3]->getType(),
                     copyArgs[4]->getType()};

  FunctionType *FT = FunctionType::get(B.getVoidTy(), copyTys, false);

  auto copy_name = std::string(blas.prefix) + blas.floatType + "copy" +
                   (cublasv2 ? "" : blas.suffix);

  auto copyfn = M.getOrInsertFunction(copy_name, FT);
  if (Function *copyF = dyn_cast<Function>(copyfn.getCallee()))
    attributeKnownFunctions(*copyF);
  LB.CreateCall(copyfn, copyArgs);
  LB.CreateCondBr(LB.CreateICmpEQ(i_plus_one, N_minus_1), end, loop);

  EB.CreateCondBr(EB.CreateICmpSLE(N_minus_1, zero), end, loop);
  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }

  if (llvm::verifyFunction(*F, &llvm::errs())) {
    llvm::errs() << *F << "\n";
    report_fatal_error("helper function failed verification");
  }
}

void callMemcpyStridedBlas(llvm::IRBuilder<> &B, llvm::Module &M, BlasInfo blas,
                           llvm::ArrayRef<llvm::Value *> args,
                           llvm::Type *copy_retty,
                           llvm::ArrayRef<llvm::OperandBundleDef> bundles) {
  const bool cublasv2 =
      blas.prefix == "cublas" && StringRef(blas.suffix).contains("v2");
  auto copy_name = std::string(blas.prefix) + blas.floatType + "copy" +
                   (cublasv2 ? "" : blas.suffix);

  SmallVector<Type *, 1> tys;
  for (auto arg : args)
    tys.push_back(arg->getType());

  FunctionType *FT = FunctionType::get(copy_retty, tys, false);
  auto fn = M.getOrInsertFunction(copy_name, FT);
  Value *callVal = fn.getCallee();
  Function *called = nullptr;
  while (!called) {
    if (auto castinst = dyn_cast<ConstantExpr>(callVal))
      if (castinst->isCast()) {
        callVal = castinst->getOperand(0);
        continue;
      }
    if (auto fn = dyn_cast<Function>(callVal)) {
      called = fn;
      break;
    }
    if (auto alias = dyn_cast<GlobalAlias>(callVal)) {
      callVal = alias->getAliasee();
      continue;
    }
    break;
  }
  attributeKnownFunctions(*called);

  B.CreateCall(fn, args, bundles);
}

void callMemcpyStridedLapack(llvm::IRBuilder<> &B, llvm::Module &M,
                             BlasInfo blas, llvm::ArrayRef<llvm::Value *> args,
                             llvm::ArrayRef<llvm::OperandBundleDef> bundles) {
  auto copy_name =
      std::string(blas.prefix) + blas.floatType + "lacpy" + blas.suffix;

  SmallVector<Type *, 1> tys;
  for (auto arg : args)
    tys.push_back(arg->getType());

  auto FT = FunctionType::get(Type::getVoidTy(M.getContext()), tys, false);
  auto fn = M.getOrInsertFunction(copy_name, FT);
  if (auto F = GetFunctionFromValue(fn.getCallee()))
    attributeKnownFunctions(*F);

  B.CreateCall(fn, args, bundles);
}

void callSPMVDiagUpdate(IRBuilder<> &B, Module &M, BlasInfo blas,
                        IntegerType *IT, Type *BlasCT, Type *BlasFPT,
                        Type *BlasPT, Type *BlasIT, Type *fpTy,
                        ArrayRef<Value *> args,
                        ArrayRef<OperandBundleDef> bundles, bool byRef,
                        bool julia_decl) {
  // add spmv diag update call if not already present
  auto fnc_name = "__enzyme_spmv_diag" + blas.floatType + blas.suffix;

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
    addFunctionNoCapture(F, 3);
    addFunctionNoCapture(F, 5);
    addFunctionNoCapture(F, 7);
    F->addParamAttr(3, Attribute::NoAlias);
    F->addParamAttr(5, Attribute::NoAlias);
    F->addParamAttr(7, Attribute::NoAlias);
    F->addParamAttr(3, Attribute::ReadOnly);
    F->addParamAttr(5, Attribute::ReadOnly);
    if (byRef) {
      addFunctionNoCapture(F, 2);
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
    Value *is_l = is_lower(B1, blasuplo, byRef, /*cublas*/ false);
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
    B2.CreateCondBr(is_l, lower_code, uper_code);

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
                     bool byRef, bool cublas, bool julia_decl) {
  assert(fpTy->isFloatingPointTy());

  // add inner_prod call if not already present
  std::string prod_name = "__enzyme_inner_prod" + blas.floatType + blas.suffix;
  auto FInnerProdT =
      FunctionType::get(fpTy, {BlasIT, BlasIT, BlasPT, BlasIT, BlasPT}, false);
  Function *F =
      cast<Function>(M.getOrInsertFunction(prod_name, FInnerProdT).getCallee());

  if (!F->empty())
    return B.CreateCall(F, args, bundles);

  // add dot call if not already present
  std::string dot_name = blas.prefix + blas.floatType + "dot" + blas.suffix;
  auto FDotT =
      FunctionType::get(fpTy, {BlasIT, BlasPT, BlasIT, BlasPT, BlasIT}, false);
  auto FDot = M.getOrInsertFunction(dot_name, FDotT);
  if (auto F = GetFunctionFromValue(FDot.getCallee()))
    attributeKnownFunctions(*F);

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
    addFunctionNoCapture(F, 2);
    addFunctionNoCapture(F, 2);
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
    Value *blasOne = to_blas_callconv(B1, ConstantInt::get(IT, 1), byRef,
                                      cublas, nullptr, B1, "constant.one");

    if (blasOne->getType() != BlasIT)
      blasOne = B1.CreatePointerCast(blasOne, BlasIT, "intcast.constant.one");

    Value *m = load_if_ref(B1, IT, blasm, byRef);
    Value *n = load_if_ref(B1, IT, blasn, byRef);
    Value *size = B1.CreateNUWMul(m, n, "mat.size");
    Value *blasSize = to_blas_callconv(
        B1, size, byRef, cublas, julia_decl ? IT : nullptr, B1, "mat.size");

    if (blasSize->getType() != BlasIT)
      blasSize = B1.CreatePointerCast(blasSize, BlasIT, "intcast.mat.size");
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
    Value *fastSum =
        B3.CreateCall(FDot, {blasSize, blasA, blasOne, blasB, blasOne});
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
        B4.CreateCall(FDot, {blasm, AiDot, blasOne, BiDot, blasOne});

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
  addFunctionNoCapture(F, 0);
  F->addParamAttr(0, Attribute::NoAlias);
  addFunctionNoCapture(F, 1);
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
      dsts->setAlignment(Align(dstalign));
    }
    if (srcalign) {
      srcl->setAlignment(Align(srcalign));
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
  assert(elementType->isFPOrFPVectorTy());
#if LLVM_VERSION_MAJOR < 17
#if LLVM_VERSION_MAJOR >= 15
  if (Mod.getContext().supportsTypedPointers()) {
#endif
#if LLVM_VERSION_MAJOR >= 13
    if (!PT->isOpaquePointerTy())
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
  addFunctionNoCapture(F, 0);
  F->addParamAttr(0, Attribute::NoAlias);
  addFunctionNoCapture(F, 1);
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
      dsts->setAlignment(Align(dstalign));
    }
    if (srcalign) {
      srcl->setAlignment(Align(srcalign));
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
  if (EnzymeMemmoveWarning)
    llvm::errs()
        << "warning: didn't implement memmove, using memcpy as fallback "
           "which can result in errors\n";
  return getOrInsertDifferentialFloatMemcpy(M, T, dstalign, srcalign, dstaddr,
                                            srcaddr, bitwidth);
}

Function *getOrInsertCheckedFree(Module &M, CallInst *call, Type *Ty,
                                 unsigned width) {
  FunctionType *FreeTy = call->getFunctionType();
  Value *Free = call->getCalledOperand();
  AttributeList FreeAttributes = call->getAttributes();
  CallingConv::ID CallingConvention = call->getCallingConv();

  std::string name = "__enzyme_checked_free_" + std::to_string(width);

  auto callname = getFuncNameFromCall(call);
  if (callname != "free")
    name += "_" + callname.str();

  SmallVector<Type *, 3> types;
  types.push_back(Ty);
  for (unsigned i = 0; i < width; i++) {
    types.push_back(Ty);
  }
#if LLVM_VERSION_MAJOR >= 14
  for (size_t i = 1; i < call->arg_size(); i++)
#else
  for (size_t i = 1; i < call->getNumArgOperands(); i++)
#endif
  {
    types.push_back(call->getArgOperand(i)->getType());
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
  addFunctionNoCapture(F, 0);
  addFunctionNoCapture(F, 1);

  Value *isNotEqual = EntryBuilder.CreateICmpNE(primal, first_shadow);
  EntryBuilder.CreateCondBr(isNotEqual, free0, end);

  SmallVector<Value *, 1> args = {first_shadow};
#if LLVM_VERSION_MAJOR >= 14
  for (size_t i = 1; i < call->arg_size(); i++)
#else
  for (size_t i = 1; i < call->getNumArgOperands(); i++)
#endif
  {
    args.push_back(F->arg_begin() + width + i);
  }

  CallInst *CI = Free0Builder.CreateCall(FreeTy, Free, args);
  CI->setAttributes(FreeAttributes);
  CI->setCallingConv(CallingConvention);

  if (width > 1) {
    Value *checkResult = nullptr;
    BasicBlock *free1 = BasicBlock::Create(M.getContext(), "free1", F);
    IRBuilder<> Free1Builder(free1);

    for (unsigned i = 0; i < width; i++) {
      addFunctionNoCapture(F, i + 1);
      Argument *shadow = F->arg_begin() + i + 1;

      if (i < width - 1) {
        Argument *nextShadow = F->arg_begin() + i + 2;
        Value *isNotEqual = Free0Builder.CreateICmpNE(shadow, nextShadow);
        checkResult = checkResult
                          ? Free0Builder.CreateAnd(isNotEqual, checkResult)
                          : isNotEqual;

        args[0] = nextShadow;
        CallInst *CI = Free1Builder.CreateCall(FreeTy, Free, args);
        CI->setAttributes(FreeAttributes);
        CI->setCallingConv(CallingConvention);
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
                                                Type *reqType,
                                                StringRef caller) {
  llvm::SmallVector<llvm::Type *, 4> types(T.begin(), T.end());
  types.push_back(reqType);

  auto &&[prefix, _, postfix] = tripleSplitDollar(caller);

  std::string name = "__enzyme_differential_mpi_wait";
  if (prefix.size() != 0 || postfix.size() != 0) {
    name = (Twine(name) + "$" + prefix + "$" + postfix).str();
  }
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
    /*0 */getInt8PtrTy(call.getContext())
    /*1 */i64
    /*2 */getInt8PtrTy(call.getContext())
    /*3 */i64
    /*4 */i64
    /*5 */getInt8PtrTy(call.getContext())
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

  auto isendfn = M.getFunction(getRenamedPerCallingConv(caller, "MPI_Isend"));
  assert(isendfn);
  // TODO: what if Isend not defined, but Irecv is?
  FunctionType *FuT = isendfn->getFunctionType();

  auto irecvfn = cast<Function>(
      M.getOrInsertFunction(getRenamedPerCallingConv(caller, "MPI_Irecv"), FuT)
          .getCallee());
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
  addFunctionNoCapture(F, 0);
  F->addParamAttr(0, Attribute::ReadOnly);
  addFunctionNoCapture(F, 1);
  addFunctionNoCapture(F, 2);
  F->addParamAttr(2, Attribute::ReadOnly);
  addFunctionNoCapture(F, 3);
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

  llvm::Type *rtypes[] = {getInt8PtrTy(M.getContext()), intType, OpPtr};
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

bool overwritesToMemoryReadBy(const TypeResults *TR, llvm::AAResults &AA,
                              llvm::TargetLibraryInfo &TLI, ScalarEvolution &SE,
                              llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                              llvm::Instruction *maybeReader,
                              llvm::Instruction *maybeWriter,
                              llvm::Loop *scope) {
  using namespace llvm;
  if (!writesToMemoryReadBy(TR, AA, TLI, maybeReader, maybeWriter))
    return false;
  const SCEV *LoadBegin = SE.getCouldNotCompute();
  const SCEV *LoadEnd = SE.getCouldNotCompute();

  const SCEV *StoreBegin = SE.getCouldNotCompute();
  const SCEV *StoreEnd = SE.getCouldNotCompute();

  Value *loadPtr = nullptr;
  Value *storePtr = nullptr;
  if (auto LI = dyn_cast<LoadInst>(maybeReader)) {
    loadPtr = LI->getPointerOperand();
    LoadBegin = SE.getSCEV(LI->getPointerOperand());
    if (LoadBegin != SE.getCouldNotCompute() &&
        !LoadBegin->getType()->isIntegerTy()) {
      auto &DL = maybeWriter->getModule()->getDataLayout();
      auto width = cast<IntegerType>(DL.getIndexType(LoadBegin->getType()))
                       ->getBitWidth();
#if LLVM_VERSION_MAJOR >= 18
      auto TS = SE.getConstant(
          APInt(width, (int64_t)DL.getTypeStoreSize(LI->getType())));
#else
      auto TS = SE.getConstant(
          APInt(width, DL.getTypeStoreSize(LI->getType()).getFixedSize()));
#endif
      LoadEnd = SE.getAddExpr(LoadBegin, TS);
    }
  }
  if (auto SI = dyn_cast<StoreInst>(maybeWriter)) {
    storePtr = SI->getPointerOperand();
    StoreBegin = SE.getSCEV(SI->getPointerOperand());
    if (StoreBegin != SE.getCouldNotCompute() &&
        !StoreBegin->getType()->isIntegerTy()) {
      auto &DL = maybeWriter->getModule()->getDataLayout();
      auto width = cast<IntegerType>(DL.getIndexType(StoreBegin->getType()))
                       ->getBitWidth();
#if LLVM_VERSION_MAJOR >= 18
      auto TS =
          SE.getConstant(APInt(width, (int64_t)DL.getTypeStoreSize(
                                          SI->getValueOperand()->getType())));
#else
      auto TS = SE.getConstant(
          APInt(width, DL.getTypeStoreSize(SI->getValueOperand()->getType())
                           .getFixedSize()));
#endif
      StoreEnd = SE.getAddExpr(StoreBegin, TS);
    }
  }
  if (auto MS = dyn_cast<MemSetInst>(maybeWriter)) {
    storePtr = MS->getArgOperand(0);
    StoreBegin = SE.getSCEV(MS->getArgOperand(0));
    if (StoreBegin != SE.getCouldNotCompute() &&
        !StoreBegin->getType()->isIntegerTy()) {
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
    storePtr = MS->getArgOperand(0);
    StoreBegin = SE.getSCEV(MS->getArgOperand(0));
    if (StoreBegin != SE.getCouldNotCompute() &&
        !StoreBegin->getType()->isIntegerTy()) {
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
    loadPtr = MS->getArgOperand(1);
    LoadBegin = SE.getSCEV(MS->getArgOperand(1));
    if (LoadBegin != SE.getCouldNotCompute() &&
        !LoadBegin->getType()->isIntegerTy()) {
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

  if (loadPtr && storePtr)
    if (auto alias =
            arePointersGuaranteedNoAlias(TLI, AA, LI, loadPtr, storePtr, true))
      if (*alias)
        return false;

  if (!overwritesToMemoryReadByLoop(SE, LI, DT, maybeReader, LoadBegin, LoadEnd,
                                    maybeWriter, StoreBegin, StoreEnd, scope))
    return false;

  return true;
}

/// Return whether maybeReader can read from memory written to by maybeWriter
bool writesToMemoryReadBy(const TypeResults *TR, llvm::AAResults &AA,
                          llvm::TargetLibraryInfo &TLI,
                          llvm::Instruction *maybeReader,
                          llvm::Instruction *maybeWriter) {
  assert(maybeReader->getParent()->getParent() ==
         maybeWriter->getParent()->getParent());
  using namespace llvm;
  if (isa<StoreInst>(maybeReader))
    return false;
  if (isa<FenceInst>(maybeReader)) {
    return false;
  }
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

    if (funcName == "jl_genericmemory_copy_slice" ||
        funcName == "ijl_genericmemory_copy_slice")
      return false;

    if (funcName == "jl_new_array" || funcName == "ijl_new_array")
      return false;

    if (funcName == "julia.safepoint")
      return false;

    if (funcName == "jl_idtable_rehash" || funcName == "ijl_idtable_rehash")
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
        auto R = parseTBAA(
            *maybeReader,
            maybeReader->getParent()->getParent()->getParent()->getDataLayout(),
            nullptr)[{-1}];
        // Could still conflict with the mpi_request unless a non pointer
        // type.
        if (R != BaseType::Unknown && R != BaseType::Anything &&
            R != BaseType::Pointer)
          return false;
      }
    }
    // Isend only writes to inaccessible mem and request.
    if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
      auto R = parseTBAA(
          *maybeReader,
          maybeReader->getParent()->getParent()->getParent()->getDataLayout(),
          nullptr)[{-1}];
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
        auto R = parseTBAA(
            *maybeReader,
            maybeReader->getParent()->getParent()->getParent()->getDataLayout(),
            nullptr)[{-1}];
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

    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand())) {
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

    if (funcName == "jl_genericmemory_copy_slice" ||
        funcName == "ijl_genericmemory_copy_slice")
      return false;

    if (funcName == "jl_idtable_rehash" || funcName == "ijl_idtable_rehash")
      return false;

    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand())) {
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
    if (TR) {
      auto TT = TR->query(li)[{-1}];
      if (TT != BaseType::Unknown && TT != BaseType::Anything) {
        if (auto si = dyn_cast<StoreInst>(maybeWriter)) {
          auto TT2 = TR->query(si->getValueOperand())[{-1}];
          if (TT2 != BaseType::Unknown && TT2 != BaseType::Anything) {
            if (TT != TT2)
              return false;
          }
          auto &dl = li->getParent()->getParent()->getParent()->getDataLayout();
          auto len =
              (dl.getTypeSizeInBits(si->getValueOperand()->getType()) + 7) / 8;
          TT2 = TR->query(si->getPointerOperand()).Lookup(len, dl)[{-1}];
          if (TT2 != BaseType::Unknown && TT2 != BaseType::Anything) {
            if (TT != TT2)
              return false;
          }
        }
      }
    }
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

// Find the base pointer of ptr and the offset in bytes from the start of
// the returned base pointer to this value.
AllocaInst *getBaseAndOffset(Value *ptr, size_t &offset) {
  offset = 0;
  while (true) {
    if (auto CI = dyn_cast<CastInst>(ptr)) {
      ptr = CI->getOperand(0);
      continue;
    }
    if (auto CI = dyn_cast<GetElementPtrInst>(ptr)) {
      auto &DL = CI->getParent()->getParent()->getParent()->getDataLayout();
#if LLVM_VERSION_MAJOR >= 20
      SmallMapVector<Value *, APInt, 4> VariableOffsets;
#else
      MapVector<Value *, APInt> VariableOffsets;
#endif
      auto width = sizeof(size_t) * 8;
      APInt Offset(width, 0);
      bool success = collectOffset(cast<GEPOperator>(CI), DL, width,
                                   VariableOffsets, Offset);
      if (!success || VariableOffsets.size() != 0 || Offset.isNegative()) {
        return nullptr;
      }
      offset += Offset.getZExtValue();
      ptr = CI->getOperand(0);
      continue;
    }
    if (isa<AllocaInst>(ptr)) {
      break;
    }
    if (auto LI = dyn_cast<LoadInst>(ptr)) {
      if (auto S = simplifyLoad(LI)) {
        ptr = S;
        continue;
      }
    }
    return nullptr;
  }
  return cast<AllocaInst>(ptr);
}

// Find all user instructions of AI, returning tuples of <instruction, value,
// byte offet from AI> Unlike a simple get users, this will recurse through any
// constant gep offsets and casts
SmallVector<std::tuple<Instruction *, Value *, size_t>, 1>
findAllUsersOf(Value *AI) {
  SmallVector<std::pair<Value *, size_t>, 1> todo;
  todo.emplace_back(AI, 0);

  SmallVector<std::tuple<Instruction *, Value *, size_t>, 1> users;
  while (todo.size()) {
    auto pair = todo.pop_back_val();
    Value *ptr = pair.first;
    size_t suboff = pair.second;

    for (auto U : ptr->users()) {
      if (auto CI = dyn_cast<CastInst>(U)) {
        todo.emplace_back(CI, suboff);
        continue;
      }
      if (auto CI = dyn_cast<GetElementPtrInst>(U)) {
        auto &DL = CI->getParent()->getParent()->getParent()->getDataLayout();
#if LLVM_VERSION_MAJOR >= 20
        SmallMapVector<Value *, APInt, 4> VariableOffsets;
#else
        MapVector<Value *, APInt> VariableOffsets;
#endif
        auto width = sizeof(size_t) * 8;
        APInt Offset(width, 0);
        bool success = collectOffset(cast<GEPOperator>(CI), DL, width,
                                     VariableOffsets, Offset);

        if (!success || VariableOffsets.size() != 0 || Offset.isNegative()) {
          users.emplace_back(cast<Instruction>(U), ptr, suboff);
          continue;
        }
        todo.emplace_back(CI, suboff + Offset.getZExtValue());
        continue;
      }
      users.emplace_back(cast<Instruction>(U), ptr, suboff);
      continue;
    }
  }
  return users;
}

// Given a pointer, find all values of size `valSz` which could be loaded from
// that pointer when indexed at offset. If it is impossible to guarantee that
// the set contains all such values, set legal to false
SmallVector<std::pair<Value *, size_t>, 1>
getAllLoadedValuesFrom(AllocaInst *ptr0, size_t offset, size_t valSz,
                       bool &legal) {
  SmallVector<std::pair<Value *, size_t>, 1> options;

  auto todo = findAllUsersOf(ptr0);
  std::set<std::tuple<Instruction *, Value *, size_t>> seen;

  while (todo.size()) {
    auto pair = todo.pop_back_val();
    if (seen.count(pair))
      continue;
    seen.insert(pair);
    Instruction *U = std::get<0>(pair);
    Value *ptr = std::get<1>(pair);
    size_t suboff = std::get<2>(pair);

    // Read only users do not set the memory inside of ptr
    if (isa<LoadInst>(U)) {
      continue;
    }
    if (auto MTI = dyn_cast<MemTransferInst>(U))
      if (MTI->getOperand(0) != ptr) {
        continue;
      }
    if (auto I = dyn_cast<Instruction>(U)) {
      if (!I->mayWriteToMemory() && I->getType()->isVoidTy())
        continue;
    }

    if (auto SI = dyn_cast<StoreInst>(U)) {
      auto &DL = SI->getParent()->getParent()->getParent()->getDataLayout();

      // We are storing into the ptr
      if (SI->getPointerOperand() == ptr) {
        auto storeSz =
            (DL.getTypeStoreSizeInBits(SI->getValueOperand()->getType()) + 7) /
            8;
        // If store is before the load would start
        if (storeSz + suboff <= offset)
          continue;
        // if store starts after load would start
        if (offset + valSz <= suboff)
          continue;

        if (valSz <= storeSz) {
          assert(offset >= suboff);
          options.emplace_back(SI->getValueOperand(), offset - suboff);
          continue;
        }
      }

      // We capture our pointer of interest, if it is stored into an alloca,
      // all loads of said alloca would potentially store into.
      if (SI->getValueOperand() == ptr) {
        if (suboff == 0) {
          size_t mid_offset = 0;
          if (auto AI2 =
                  getBaseAndOffset(SI->getPointerOperand(), mid_offset)) {
            bool sublegal = true;
            auto ptrSz = (DL.getTypeStoreSizeInBits(ptr->getType()) + 7) / 8;
            auto subPtrs =
                getAllLoadedValuesFrom(AI2, mid_offset, ptrSz, sublegal);
            if (!sublegal) {
              legal = false;
              return options;
            }
            for (auto &&[subPtr, subOff] : subPtrs) {
              if (subOff != 0)
                return options;
              for (const auto &pair3 : findAllUsersOf(subPtr)) {
                todo.emplace_back(std::move(pair3));
              }
            }
            continue;
          }
        }
      }
    }

    if (auto II = dyn_cast<IntrinsicInst>(U)) {
      if (II->getCalledFunction()->getName() == "llvm.enzyme.lifetime_start" ||
          II->getCalledFunction()->getName() == "llvm.enzyme.lifetime_end")
        continue;
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end)
        continue;
    }

    // If we copy into the ptr at a location that includes the offset, consider
    // all sub uses
    if (auto MTI = dyn_cast<MemTransferInst>(U)) {
      if (auto CI = dyn_cast<ConstantInt>(MTI->getLength())) {
        if (MTI->getOperand(0) == ptr) {
          auto storeSz = CI->getValue();

          // If store is before the load would start
          if ((storeSz + suboff).ule(offset))
            continue;

          // if store starts after load would start
          if (offset + valSz <= suboff)
            continue;

          if (suboff == 0 && CI->getValue().uge(offset + valSz)) {
            size_t midoffset = 0;
            auto AI2 = getBaseAndOffset(MTI->getOperand(1), midoffset);
            if (!AI2) {
              legal = false;
              return options;
            }
            if (midoffset != 0) {
              legal = false;
              return options;
            }
            for (const auto &pair3 : findAllUsersOf(AI2)) {
              todo.emplace_back(std::move(pair3));
            }
            continue;
          }
        }
      }
    }

    legal = false;
    return options;
  }

  return options;
}

// Perform mem2reg/sroa to identify the innermost value being represented.
Value *simplifyLoad(Value *V, size_t valSz, size_t preOffset) {
  if (auto LI = dyn_cast<LoadInst>(V)) {
    if (valSz == 0) {
      auto &DL = LI->getParent()->getParent()->getParent()->getDataLayout();
      valSz = (DL.getTypeSizeInBits(LI->getType()) + 7) / 8;
    }

    Value *ptr = LI->getPointerOperand();
    size_t offset = 0;

    if (auto ptr2 = simplifyLoad(ptr)) {
      ptr = ptr2;
    }
    auto AI = getBaseAndOffset(ptr, offset);
    if (!AI) {
      return nullptr;
    }
    offset += preOffset;

    bool legal = true;
    auto opts = getAllLoadedValuesFrom(AI, offset, valSz, legal);

    if (!legal) {
      return nullptr;
    }
    std::set<Value *> res;
    for (auto &&[opt, startOff] : opts) {
      Value *v2 = simplifyLoad(opt, valSz, startOff);
      if (v2)
        res.insert(v2);
      else
        res.insert(opt);
    }
    if (res.size() != 1) {
      return nullptr;
    }
    Value *retval = *res.begin();
    return retval;
  }
  if (auto EVI = dyn_cast<ExtractValueInst>(V)) {
    IRBuilder<> B(EVI);
    auto em =
        GradientUtils::extractMeta(B, EVI->getAggregateOperand(),
                                   EVI->getIndices(), "", /*fallback*/ false);
    if (em != nullptr) {
      if (auto SL2 = simplifyLoad(em, valSz))
        em = SL2;
      return em;
    }
    if (auto LI = dyn_cast<LoadInst>(EVI->getAggregateOperand())) {
      auto offset = preOffset;

      auto &DL = LI->getParent()->getParent()->getParent()->getDataLayout();
      SmallVector<Value *, 4> vec;
      vec.push_back(ConstantInt::get(Type::getInt64Ty(EVI->getContext()), 0));
      for (auto ind : EVI->getIndices()) {
        vec.push_back(
            ConstantInt::get(Type::getInt32Ty(EVI->getContext()), ind));
      }
      auto ud = UndefValue::get(
          PointerType::getUnqual(EVI->getOperand(0)->getType()));
      auto g2 =
          GetElementPtrInst::Create(EVI->getOperand(0)->getType(), ud, vec);
      APInt ai(DL.getIndexSizeInBits(g2->getPointerAddressSpace()), 0);
      g2->accumulateConstantOffset(DL, ai);
      // Using destructor rather than eraseFromParent
      //   as g2 has no parent
      delete g2;

      offset += (size_t)ai.getLimitedValue();

      if (valSz == 0) {
        auto &DL = EVI->getParent()->getParent()->getParent()->getDataLayout();
        valSz = (DL.getTypeSizeInBits(EVI->getType()) + 7) / 8;
      }
      return simplifyLoad(LI, valSz, offset);
    }
  }
  return nullptr;
}

Value *GetFunctionValFromValue(Value *fn) {
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
          val = GetFunctionValFromValue(val);
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
          while (isa<LoadInst>(val)) {
            auto v2 = simplifyLoad(val);
            if (v2) {
              val = v2;
              continue;
            }
            break;
          }
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
    if (auto S = simplifyLoad(fn)) {
      fn = S;
      continue;
    }
    break;
  }

  return fn;
}

Function *GetFunctionFromValue(Value *fn) {
  return dyn_cast<Function>(GetFunctionValFromValue(fn));
}

#if LLVM_VERSION_MAJOR >= 16
std::optional<BlasInfo> extractBLAS(llvm::StringRef in)
#else
llvm::Optional<BlasInfo> extractBLAS(llvm::StringRef in)
#endif
{
  const char *extractable[] = {
      "dot",   "scal",  "axpy",  "gemv",  "gemm",  "spmv", "syrk", "nrm2",
      "trmm",  "trmv",  "symm",  "potrf", "potrs", "copy", "spmv", "syr2k",
      "potrs", "getrf", "getrs", "trtrs", "getri", "symv",
  };
  const char *floatType[] = {"s", "d", "c", "z"};
  const char *prefixes[] = {"" /*Fortran*/, "cblas_"};
  const char *suffixes[] = {"", "_", "64_", "_64_"};
  for (auto t : floatType) {
    for (auto f : extractable) {
      for (auto p : prefixes) {
        for (auto s : suffixes) {
          if (in == (Twine(p) + t + f + s).str()) {
            bool is64 = llvm::StringRef(s).contains("64");
            return BlasInfo{
                t, p, s, f, is64,
            };
          }
        }
      }
    }
  }
  // c interface to cublas
  const char *cuCFloatType[] = {"S", "D", "C", "Z"};
  const char *cuFFloatType[] = {"s", "d", "c", "z"};
  const char *cuCPrefixes[] = {"cublas"};
  const char *cuSuffixes[] = {"", "_v2", "_64", "_v2_64"};
  for (auto t : llvm::enumerate(cuCFloatType)) {
    for (auto f : extractable) {
      for (auto p : cuCPrefixes) {
        for (auto s : cuSuffixes) {
          if (in == (Twine(p) + t.value() + f + s).str()) {
            bool is64 = llvm::StringRef(s).contains("64");
            return BlasInfo{
                t.value(), p, s, f, is64,
            };
          }
        }
      }
    }
  }
  // Fortran interface to cublas
  const char *cuFPrefixes[] = {"cublas_"};
  for (auto t : cuFFloatType) {
    for (auto f : extractable) {
      for (auto p : cuFPrefixes) {
        if (in == (Twine(p) + t + f).str()) {
          return BlasInfo{
              t, p, "", f, false,
          };
        }
      }
    }
  }
  return {};
}

llvm::Constant *getUndefinedValueForType(llvm::Module &M, llvm::Type *T,
                                         bool forceZero) {
  if (EnzymeUndefinedValueForType)
    return cast<Constant>(
        unwrap(EnzymeUndefinedValueForType(wrap(&M), wrap(T), forceZero)));
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
  if (!cache_arg)
    return;
  if (!arg->getType()->isPointerTy()) {
    assert(arg->getType() == ty);
    cacheValues.push_back(arg);
    return;
  }
#if LLVM_VERSION_MAJOR < 17
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
                              bool cublas, IntegerType *julia_decl,
                              IRBuilder<> &entryBuilder,
                              llvm::Twine const &name) {
  if (!byRef)
    return V;

  Value *allocV =
      entryBuilder.CreateAlloca(V->getType(), nullptr, "byref." + name);
  B.CreateStore(V, allocV);

  if (julia_decl)
    allocV = B.CreatePointerCast(allocV, getInt8PtrTy(V->getContext()),
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

Value *is_lower(IRBuilder<> &B, Value *uplo, bool byRef, bool cublas) {
  if (cublas) {
    Value *isNormal = nullptr;
    isNormal = B.CreateICmpEQ(
        uplo, ConstantInt::get(uplo->getType(),
                               /*cublasFillMode_t::CUBLAS_FILL_MODE_LOWER*/ 0));
    return isNormal;
  }
  if (auto CI = dyn_cast<ConstantInt>(uplo)) {
    if (CI->getValue() == 'L' || CI->getValue() == 'l')
      return ConstantInt::getTrue(B.getContext());
    if (CI->getValue() == 'U' || CI->getValue() == 'u')
      return ConstantInt::getFalse(B.getContext());
  }
  if (byRef) {
    // can't inspect opaque ptr, so assume 8 (Julia)
    IntegerType *charTy = IntegerType::get(uplo->getContext(), 8);
    uplo = B.CreateLoad(charTy, uplo, "loaded.trans");

    auto isL = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'L'));
    auto isl = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'l'));
    // fortran blas
    return B.CreateOr(isl, isL);
  } else {
    // we can inspect scalars
    auto capi = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 122));
    // TODO we really should just return capi, but for sake of consistency,
    // we will accept either here.
    auto isL = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'L'));
    auto isl = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'l'));
    return B.CreateOr(capi, B.CreateOr(isl, isL));
  }
}

Value *is_nonunit(IRBuilder<> &B, Value *uplo, bool byRef, bool cublas) {
  if (cublas) {
    Value *isNormal = nullptr;
    isNormal =
        B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(),
                                              /*CUBLAS_DIAG_NON_UNIT*/ 0));
    return isNormal;
  }
  if (auto CI = dyn_cast<ConstantInt>(uplo)) {
    if (CI->getValue() == 'N' || CI->getValue() == 'n')
      return ConstantInt::getTrue(B.getContext());
    if (CI->getValue() == 'U' || CI->getValue() == 'u')
      return ConstantInt::getFalse(B.getContext());
  }
  if (byRef) {
    // can't inspect opaque ptr, so assume 8 (Julia)
    IntegerType *charTy = IntegerType::get(uplo->getContext(), 8);
    uplo = B.CreateLoad(charTy, uplo, "loaded.nonunit");

    auto isL = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'N'));
    auto isl = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'n'));
    // fortran blas
    return B.CreateOr(isl, isL);
  } else {
    // we can inspect scalars
    auto capi = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 131));
    // TODO we really should just return capi, but for sake of consistency,
    // we will accept either here.
    auto isL = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'N'));
    auto isl = B.CreateICmpEQ(uplo, ConstantInt::get(uplo->getType(), 'n'));
    return B.CreateOr(capi, B.CreateOr(isl, isL));
  }
}

llvm::Value *is_normal(IRBuilder<> &B, llvm::Value *trans, bool byRef,
                       bool cublas) {
  if (cublas) {
    Value *isNormal = nullptr;
    isNormal = B.CreateICmpEQ(
        trans, ConstantInt::get(trans->getType(),
                                /*cublasOperation_t::CUBLAS_OP_N*/ 0));
    return isNormal;
  }
  // Explicitly support 'N' always, since we use in the rule infra
  if (auto CI = dyn_cast<ConstantInt>(trans)) {
    if (CI->getValue() == 'N' || CI->getValue() == 'n')
      return ConstantInt::getTrue(
          B.getContext()); //(Type::getInt1Ty(B.getContext()), true);
  }
  if (byRef) {
    // can't inspect opaque ptr, so assume 8 (Julia)
    IntegerType *charTy = IntegerType::get(trans->getContext(), 8);
    trans = B.CreateLoad(charTy, trans, "loaded.trans");

    auto isN = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'N'));
    auto isn = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'n'));
    // fortran blas
    return B.CreateOr(isn, isN);
  } else {
    // TODO we really should just return capi, but for sake of consistency,
    // we will accept either here.
    // we can inspect scalars
    auto capi = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 111));
    auto isN = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'N'));
    auto isn = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'n'));
    // fortran blas
    return B.CreateOr(capi, B.CreateOr(isn, isN));
  }
}

llvm::Value *is_left(IRBuilder<> &B, llvm::Value *side, bool byRef,
                     bool cublas) {
  if (cublas) {
    Value *isNormal = nullptr;
    isNormal = B.CreateICmpEQ(
        side, ConstantInt::get(side->getType(),
                               /*cublasSideMode_t::CUBLAS_SIDE_LEFT*/ 0));
    return isNormal;
  }
  // Explicitly support 'L'/'R' always, since we use in the rule infra
  if (auto CI = dyn_cast<ConstantInt>(side)) {
    if (CI->getValue() == 'L' || CI->getValue() == 'l')
      return ConstantInt::getTrue(B.getContext());
    if (CI->getValue() == 'R' || CI->getValue() == 'r')
      return ConstantInt::getFalse(B.getContext());
  }
  if (byRef) {
    // can't inspect opaque ptr, so assume 8 (Julia)
    IntegerType *charTy = IntegerType::get(side->getContext(), 8);
    side = B.CreateLoad(charTy, side, "loaded.side");

    auto isL = B.CreateICmpEQ(side, ConstantInt::get(side->getType(), 'L'));
    auto isl = B.CreateICmpEQ(side, ConstantInt::get(side->getType(), 'l'));
    // fortran blas
    return B.CreateOr(isl, isL);
  } else {
    // TODO we really should just return capi, but for sake of consistency,
    // we will accept either here.
    // we can inspect scalars
    auto capi = B.CreateICmpEQ(side, ConstantInt::get(side->getType(), 141));
    auto isL = B.CreateICmpEQ(side, ConstantInt::get(side->getType(), 'L'));
    auto isl = B.CreateICmpEQ(side, ConstantInt::get(side->getType(), 'l'));
    // fortran blas
    return B.CreateOr(capi, B.CreateOr(isl, isL));
  }
}

// Ok. Here we are.
// netlib declares trans args as something out of
// N,n,T,t,C,c, represented as 8 bit chars.
// However, if we ask openBlas c ABI,
// it is one of the following 32 bit integers values:
// enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
llvm::Value *transpose(std::string floatType, IRBuilder<> &B, llvm::Value *V,
                       bool cublas) {
  llvm::Type *T = V->getType();
  if (cublas) {
    auto isT1 = B.CreateICmpEQ(V, ConstantInt::get(T, 1));
    auto isT0 = B.CreateICmpEQ(V, ConstantInt::get(T, 0));
    return B.CreateSelect(isT1, ConstantInt::get(V->getType(), 0),
                          B.CreateSelect(isT0,
                                         ConstantInt::get(V->getType(), 1),
                                         ConstantInt::get(V->getType(), 42)));
  } else if (T->isIntegerTy(8)) {
    if (floatType == "z" || floatType == "c") {
      auto isn = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'n'));
      auto sel1 = B.CreateSelect(isn, ConstantInt::get(V->getType(), 'c'),
                                 ConstantInt::get(V->getType(), 0));

      auto isN = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'N'));
      auto sel2 =
          B.CreateSelect(isN, ConstantInt::get(V->getType(), 'C'), sel1);

      auto ist = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'c'));
      auto sel3 =
          B.CreateSelect(ist, ConstantInt::get(V->getType(), 'n'), sel2);

      auto isT = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'C'));
      return B.CreateSelect(isT, ConstantInt::get(V->getType(), 'N'), sel3);
    } else {
      // the base case here of 'C' or 'c' becomes simply 'N'
      auto isn = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'n'));
      auto sel1 = B.CreateSelect(isn, ConstantInt::get(V->getType(), 't'),
                                 ConstantInt::get(V->getType(), 'N'));

      auto isN = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'N'));
      auto sel2 =
          B.CreateSelect(isN, ConstantInt::get(V->getType(), 'T'), sel1);

      auto ist = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 't'));
      auto sel3 =
          B.CreateSelect(ist, ConstantInt::get(V->getType(), 'n'), sel2);

      auto isT = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 'T'));
      return B.CreateSelect(isT, ConstantInt::get(V->getType(), 'N'), sel3);
    }

  } else if (T->isIntegerTy(32)) {
    auto is111 = B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 111));
    auto sel1 = B.CreateSelect(
        B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 112)),
        ConstantInt::get(V->getType(), 111), ConstantInt::get(V->getType(), 0));
    return B.CreateSelect(is111, ConstantInt::get(V->getType(), 112), sel1);
  } else {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "cannot handle unknown trans blas value\n" << V;
    if (CustomErrorHandler) {
      CustomErrorHandler(ss.str().c_str(), nullptr, ErrorType::NoDerivative,
                         nullptr, nullptr, nullptr);
    } else {
      EmitFailure("unknown trans blas value", B.getCurrentDebugLocation(),
                  B.GetInsertBlock()->getParent(), ss.str());
    }
    return V;
  }
}

// Implement the following logic to get the width of a matrix
// if (cache_A) {
//   ld_A = (arg_transa == 'N') ? arg_k : arg_m;
// } else {
//   ld_A = arg_lda;
// }
llvm::Value *get_cached_mat_width(llvm::IRBuilder<> &B,
                                  llvm::ArrayRef<llvm::Value *> trans,
                                  llvm::Value *arg_ld, llvm::Value *dim1,
                                  llvm::Value *dim2, bool cacheMat, bool byRef,
                                  bool cublas) {
  if (!cacheMat)
    return arg_ld;

  assert(trans.size() == 1);

  llvm::Value *width =
      CreateSelect(B, is_normal(B, trans[0], byRef, cublas), dim2, dim1);

  return width;
}

llvm::Value *transpose(std::string floatType, llvm::IRBuilder<> &B,
                       llvm::Value *V, bool byRef, bool cublas,
                       llvm::IntegerType *julia_decl,
                       llvm::IRBuilder<> &entryBuilder,
                       const llvm::Twine &name) {

  if (!byRef) {
    // Explicitly support 'N' always, since we use in the rule infra
    if (auto CI = dyn_cast<ConstantInt>(V)) {
      if (floatType == "c" || floatType == "z") {
        if (CI->getValue() == 'N')
          return ConstantInt::get(CI->getType(), 'C');
        if (CI->getValue() == 'c')
          return ConstantInt::get(CI->getType(), 'c');
      } else {
        if (CI->getValue() == 'N')
          return ConstantInt::get(CI->getType(), 'T');
        if (CI->getValue() == 'n')
          return ConstantInt::get(CI->getType(), 't');
      }
    }

    // cblas
    if (!cublas)
      return B.CreateSelect(
          B.CreateICmpEQ(V, ConstantInt::get(V->getType(), 111)),
          ConstantInt::get(V->getType(), 112),
          ConstantInt::get(V->getType(), 111));
  }

  if (byRef) {
    auto charType = IntegerType::get(V->getContext(), 8);
    V = B.CreateLoad(charType, V, "ld." + name);
  }

  V = transpose(floatType, B, V, cublas);

  return to_blas_callconv(B, V, byRef, cublas, julia_decl, entryBuilder,
                          "transpose." + name);
}

llvm::Value *load_if_ref(llvm::IRBuilder<> &B, llvm::Type *intType,
                         llvm::Value *V, bool byRef) {
  if (!byRef)
    return V;

  if (V->getType()->isIntegerTy())
    V = B.CreateIntToPtr(V, PointerType::getUnqual(intType));
  else
    V = B.CreatePointerCast(
        V, PointerType::get(
               intType, cast<PointerType>(V->getType())->getAddressSpace()));
  return B.CreateLoad(intType, V);
}

SmallVector<llvm::Value *, 1> get_blas_row(llvm::IRBuilder<> &B,
                                           ArrayRef<llvm::Value *> transA,
                                           bool byRef, bool cublas) {
  assert(transA.size() == 1);
  auto trans = transA[0];
  if (byRef) {
    auto charType = IntegerType::get(trans->getContext(), 8);
    trans = B.CreateLoad(charType, trans, "ld.row.trans");
  }

  Value *cond = nullptr;
  if (!cublas) {

    if (!byRef) {
      cond = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 111));
    } else {
      auto isn = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'n'));
      auto isN = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 'N'));
      cond = B.CreateOr(isN, isn);
    }
  } else {
    // CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2
    // TODO: verify
    cond = B.CreateICmpEQ(trans, ConstantInt::get(trans->getType(), 0));
  }
  return {cond};
}
SmallVector<llvm::Value *, 1> get_blas_row(llvm::IRBuilder<> &B,
                                           ArrayRef<llvm::Value *> transA,
                                           ArrayRef<llvm::Value *> row,
                                           ArrayRef<llvm::Value *> col,
                                           bool byRef, bool cublas) {
  auto conds = get_blas_row(B, transA, byRef, cublas);
  assert(row.size() == col.size());
  SmallVector<Value *, 1> toreturn;
  for (size_t i = 0; i < row.size(); i++) {
    auto lhs = row[i];
    auto rhs = col[i];
    if (lhs->getType() != rhs->getType())
      rhs = B.CreatePointerCast(rhs, lhs->getType());
    toreturn.push_back(B.CreateSelect(conds[0], lhs, rhs));
  }
  return toreturn;
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

#if LLVM_VERSION_MAJOR >= 20
bool collectOffset(GEPOperator *gep, const DataLayout &DL, unsigned BitWidth,
                   SmallMapVector<Value *, APInt, 4> &VariableOffsets,
                   APInt &ConstantOffset)
#else
bool collectOffset(GEPOperator *gep, const DataLayout &DL, unsigned BitWidth,
                   MapVector<Value *, APInt> &VariableOffsets,
                   APInt &ConstantOffset)
#endif
{
#if LLVM_VERSION_MAJOR >= 13
  return gep->collectOffset(DL, BitWidth, VariableOffsets, ConstantOffset);
#else
  assert(BitWidth == DL.getIndexSizeInBits(gep->getPointerAddressSpace()) &&
         "The offset bit width does not match DL specification.");

  auto CollectConstantOffset = [&](APInt Index, uint64_t Size) {
    Index = Index.sextOrTrunc(BitWidth);
    APInt IndexedSize = APInt(BitWidth, Size);
    ConstantOffset += Index * IndexedSize;
  };

  for (gep_type_iterator GTI = gep_type_begin(gep), GTE = gep_type_end(gep);
       GTI != GTE; ++GTI) {
    // Scalable vectors are multiplied by a runtime constant.
    bool ScalableType = isa<ScalableVectorType>(GTI.getIndexedType());

    Value *V = GTI.getOperand();
    StructType *STy = GTI.getStructTypeOrNull();
    // Handle ConstantInt if possible.
    if (auto ConstOffset = dyn_cast<ConstantInt>(V)) {
      if (ConstOffset->isZero())
        continue;
      // If the type is scalable and the constant is not zero (vscale * n * 0 =
      // 0) bailout.
      // TODO: If the runtime value is accessible at any point before DWARF
      // emission, then we could potentially keep a forward reference to it
      // in the debug value to be filled in later.
      if (ScalableType)
        return false;
      // Handle a struct index, which adds its field offset to the pointer.
      if (STy) {
        unsigned ElementIdx = ConstOffset->getZExtValue();
        const StructLayout *SL = DL.getStructLayout(STy);
        // Element offset is in bytes.
        CollectConstantOffset(APInt(BitWidth, SL->getElementOffset(ElementIdx)),
                              1);
        continue;
      }
      CollectConstantOffset(ConstOffset->getValue(),
                            DL.getTypeAllocSize(GTI.getIndexedType()));
      continue;
    }

    if (STy || ScalableType)
      return false;
    APInt IndexedSize =
        APInt(BitWidth, DL.getTypeAllocSize(GTI.getIndexedType()));
    // Insert an initial offset of 0 for V iff none exists already, then
    // increment the offset by IndexedSize.
    if (IndexedSize != 0) {
      VariableOffsets.insert({V, APInt(BitWidth, 0)});
      VariableOffsets[V] += IndexedSize;
    }
  }
  return true;
#endif
}

llvm::CallInst *createIntrinsicCall(llvm::IRBuilderBase &B,
                                    llvm::Intrinsic::ID ID, llvm::Type *RetTy,
                                    llvm::ArrayRef<llvm::Value *> Args,
                                    llvm::Instruction *FMFSource,
                                    const llvm::Twine &Name) {
#if LLVM_VERSION_MAJOR >= 16
  llvm::CallInst *nres = B.CreateIntrinsic(RetTy, ID, Args, FMFSource, Name);
#else
  SmallVector<Intrinsic::IITDescriptor, 1> Table;
  Intrinsic::getIntrinsicInfoTableEntries(ID, Table);
  ArrayRef<Intrinsic::IITDescriptor> TableRef(Table);

  SmallVector<Type *, 2> ArgTys;
  ArgTys.reserve(Args.size());
  for (auto &I : Args)
    ArgTys.push_back(I->getType());
  FunctionType *FTy = FunctionType::get(RetTy, ArgTys, false);
  SmallVector<Type *, 2> OverloadTys;
  Intrinsic::MatchIntrinsicTypesResult Res =
      matchIntrinsicSignature(FTy, TableRef, OverloadTys);
  (void)Res;
  assert(Res == Intrinsic::MatchIntrinsicTypes_Match && TableRef.empty() &&
         "Wrong types for intrinsic!");
  Function *Fn = Intrinsic::getDeclaration(B.GetInsertPoint()->getModule(), ID,
                                           OverloadTys);
  CallInst *nres = B.CreateCall(Fn, Args, {}, Name);
  if (FMFSource)
    nres->copyFastMathFlags(FMFSource);
#endif
  return nres;
}

/* Bithack to compute 1 ulp as follows:
double ulp(double res) {
  double nres = res;
  (*(uint64_t*)&nres) = 0x1 ^ *(uint64_t*)&nres;
  return abs(nres - res);
}
*/
llvm::Value *get1ULP(llvm::IRBuilder<> &builder, llvm::Value *res) {
  auto ty = res->getType();
  unsigned tsize = builder.GetInsertBlock()
                       ->getParent()
                       ->getParent()
                       ->getDataLayout()
                       .getTypeSizeInBits(ty);

  auto ity = IntegerType::get(ty->getContext(), tsize);

  auto as_int = builder.CreateBitCast(res, ity);
  auto masked = builder.CreateXor(as_int, ConstantInt::get(ity, 1));
  auto neighbor = builder.CreateBitCast(masked, ty);

  auto diff = builder.CreateFSub(res, neighbor);

  auto absres = builder.CreateIntrinsic(Intrinsic::fabs,
                                        ArrayRef<Type *>(diff->getType()),
                                        ArrayRef<Value *>(diff));

  return absres;
}

llvm::Value *EmitNoDerivativeError(const std::string &message,
                                   llvm::Instruction &inst,
                                   GradientUtils *gutils,
                                   llvm::IRBuilder<> &Builder2,
                                   llvm::Value *condition) {
  if (CustomErrorHandler) {
    return unwrap(CustomErrorHandler(message.c_str(), wrap(&inst),
                                     ErrorType::NoDerivative, gutils,
                                     wrap(condition), wrap(&Builder2)));
  } else if (EnzymeRuntimeError) {
    auto &M = *inst.getParent()->getParent()->getParent();
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(M.getContext()),
                                         {getInt8PtrTy(M.getContext())}, false);
    auto msg = getString(M, message);
    auto PutsF = M.getOrInsertFunction("puts", FT);
    Builder2.CreateCall(PutsF, msg);

    FunctionType *FT2 =
        FunctionType::get(Type::getVoidTy(M.getContext()),
                          {Type::getInt32Ty(M.getContext())}, false);

    auto ExitF = M.getOrInsertFunction("exit", FT2);
    Builder2.CreateCall(ExitF,
                        ConstantInt::get(Type::getInt32Ty(M.getContext()), 1));
    return nullptr;
  } else {
    if (StringRef(message).contains("cannot handle above cast")) {
      gutils->TR.dump();
    }
    EmitFailure("NoDerivative", inst.getDebugLoc(), &inst, message);
    return nullptr;
  }
}

bool EmitNoDerivativeError(const std::string &message, Value *todiff,
                           RequestContext &context) {
  Value *toshow = todiff;
  if (context.req) {
    toshow = context.req;
  }
  if (CustomErrorHandler) {
    CustomErrorHandler(message.c_str(), wrap(toshow), ErrorType::NoDerivative,
                       nullptr, wrap(todiff), wrap(context.ip));
    return true;
  } else if (context.ip && EnzymeRuntimeError) {
    auto &M = *context.ip->GetInsertBlock()->getParent()->getParent();
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(M.getContext()),
                                         {getInt8PtrTy(M.getContext())}, false);
    auto msg = getString(M, message);
    auto PutsF = M.getOrInsertFunction("puts", FT);
    context.ip->CreateCall(PutsF, msg);

    FunctionType *FT2 =
        FunctionType::get(Type::getVoidTy(M.getContext()),
                          {Type::getInt32Ty(M.getContext())}, false);

    auto ExitF = M.getOrInsertFunction("exit", FT2);
    context.ip->CreateCall(
        ExitF, ConstantInt::get(Type::getInt32Ty(M.getContext()), 1));
    return true;
  } else if (context.req) {
    EmitFailure("NoDerivative", context.req->getDebugLoc(), context.req,
                message);
    return true;
  } else if (auto arg = dyn_cast<Instruction>(todiff)) {
    auto loc = arg->getDebugLoc();
    EmitFailure("NoDerivative", loc, arg, message);
    return true;
  }
  return false;
}

void EmitNoTypeError(const std::string &message, llvm::Instruction &inst,
                     GradientUtils *gutils, llvm::IRBuilder<> &Builder2) {
  if (CustomErrorHandler) {
    CustomErrorHandler(message.c_str(), wrap(&inst), ErrorType::NoType,
                       gutils->TR.analyzer, nullptr, wrap(&Builder2));
  } else if (EnzymeRuntimeError) {
    auto &M = *inst.getParent()->getParent()->getParent();
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(M.getContext()),
                                         {getInt8PtrTy(M.getContext())}, false);
    auto msg = getString(M, message);
    auto PutsF = M.getOrInsertFunction("puts", FT);
    Builder2.CreateCall(PutsF, msg);

    FunctionType *FT2 =
        FunctionType::get(Type::getVoidTy(M.getContext()),
                          {Type::getInt32Ty(M.getContext())}, false);

    auto ExitF = M.getOrInsertFunction("exit", FT2);
    Builder2.CreateCall(ExitF,
                        ConstantInt::get(Type::getInt32Ty(M.getContext()), 1));
  } else {
    std::string str;
    raw_string_ostream ss(str);
    ss << message << "\n";
    gutils->TR.dump(ss);
    EmitFailure("CannotDeduceType", inst.getDebugLoc(), &inst, ss.str());
  }
}

std::vector<std::tuple<llvm::Type *, size_t, size_t>>
parseTrueType(const llvm::MDNode *md, DerivativeMode Mode, bool const_src) {
  std::vector<std::pair<ConcreteType, size_t>> parsed;
  for (size_t i = 0; i < md->getNumOperands(); i += 2) {
    ConcreteType base(
        llvm::cast<llvm::MDString>(md->getOperand(i))->getString(),
        md->getContext());
    auto size = llvm::cast<llvm::ConstantInt>(
                    llvm::cast<llvm::ConstantAsMetadata>(md->getOperand(i + 1))
                        ->getValue())
                    ->getSExtValue();
    parsed.emplace_back(base, size);
  }

  std::vector<std::tuple<llvm::Type *, size_t, size_t>> toIterate;
  size_t idx = 0;
  while (idx < parsed.size()) {

    auto dt = parsed[idx].first;
    size_t start = parsed[idx].second;
    size_t end = 0x0fffffff;
    for (idx = idx + 1; idx < parsed.size(); ++idx) {
      bool Legal = true;
      auto tmp = dt;
      auto next = parsed[idx].first;
      tmp.checkedOrIn(next, /*PointerIntSame*/ true, Legal);
      // Prevent fusion of {Anything, Float} since anything is an int rule
      // but float requires zeroing.
      if ((dt == BaseType::Anything &&
           (next != BaseType::Anything && next.isKnown())) ||
          (next == BaseType::Anything &&
           (dt != BaseType::Anything && dt.isKnown())))
        Legal = false;
      if (!Legal) {
        if (Mode == DerivativeMode::ForwardMode ||
            Mode == DerivativeMode::ForwardModeError) {
          // if both are floats (of any type), forward mode is the same.
          //   + [potentially zero if const, otherwise copy]
          // if both are int/pointer (of any type), also the same
          //   + copy
          // if known non-constant, also the same
          //   + copy
          if ((parsed[idx].first.isFloat() == nullptr) ==
              (parsed[idx - 1].first.isFloat() == nullptr)) {
            Legal = true;
          }
          if (const_src) {
            Legal = true;
          }
        }
        if (!Legal) {
          end = parsed[idx].second;
          break;
        }
      } else
        dt = tmp;
    }
    assert(dt.isKnown());
    toIterate.emplace_back(dt.isFloat(), start, end - start);
  }
  return toIterate;
}

void dumpModule(llvm::Module *mod) { llvm::errs() << *mod << "\n"; }

void dumpValue(llvm::Value *val) { llvm::errs() << *val << "\n"; }

void dumpBlock(llvm::BasicBlock *blk) { llvm::errs() << *blk << "\n"; }

void dumpType(llvm::Type *ty) { llvm::errs() << *ty << "\n"; }

void dumpTypeResults(TypeResults &TR) { TR.dump(); }

bool isNVLoad(const llvm::Value *V) {
  auto II = dyn_cast<IntrinsicInst>(V);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_p:
  case Intrinsic::nvvm_ldu_global_f:
#if LLVM_VERSION_MAJOR < 20
  case Intrinsic::nvvm_ldg_global_i:
  case Intrinsic::nvvm_ldg_global_p:
  case Intrinsic::nvvm_ldg_global_f:
#endif
    return true;
  default:
    return false;
  }
  return false;
}

bool notCapturedBefore(llvm::Value *V, Instruction *inst,
                       size_t checkLoadCaptures) {
  Instruction *VI = dyn_cast<Instruction>(V);
  if (!VI)
    VI = &*inst->getParent()->getParent()->getEntryBlock().begin();
  else
    VI = VI->getNextNode();
  SmallPtrSet<BasicBlock *, 1> regionBetween;
  if (inst) {
    SmallVector<BasicBlock *, 1> todo;
    todo.push_back(VI->getParent());
    while (todo.size()) {
      auto cur = todo.pop_back_val();
      if (regionBetween.count(cur))
        continue;
      regionBetween.insert(cur);
      if (cur == inst->getParent())
        continue;
      for (auto BB : successors(cur))
        todo.push_back(BB);
    }
  }
  SmallVector<std::tuple<Instruction *, size_t, Value *>, 1> todo;
  for (auto U : V->users()) {
    todo.emplace_back(cast<Instruction>(U), checkLoadCaptures, V);
  }
  std::set<std::tuple<Value *, size_t, Value *>> seen;
  while (todo.size()) {
    auto pair = todo.pop_back_val();
    if (seen.count(pair))
      continue;
    auto UI = std::get<0>(pair);
    auto level = std::get<1>(pair);
    auto prev = std::get<2>(pair);
    if (inst) {
      if (!regionBetween.count(UI->getParent()))
        continue;
      if (UI->getParent() == VI->getParent()) {
        if (UI->comesBefore(VI))
          continue;
      }
      if (UI->getParent() == inst->getParent())
        if (inst->comesBefore(UI))
          continue;
    }

    if (isPointerArithmeticInst(UI, /*includephi*/ true,
                                /*includebin*/ true)) {
      for (auto U2 : UI->users()) {
        auto UI2 = cast<Instruction>(U2);
        todo.emplace_back(UI2, level, UI);
      }
      continue;
    }

    if (isa<MemSetInst>(UI))
      continue;

    if (isa<MemTransferInst>(UI)) {
      if (level == 0)
        continue;
      if (UI->getOperand(1) != prev)
        continue;
    }

    if (auto CI = dyn_cast<CallBase>(UI)) {
#if LLVM_VERSION_MAJOR >= 14
      for (size_t i = 0, size = CI->arg_size(); i < size; i++)
#else
      for (size_t i = 0, size = CI->getNumArgOperands(); i < size; i++)
#endif
      {
        if (prev == CI->getArgOperand(i)) {
          if (isNoCapture(CI, i) && level == 0)
            continue;
          return false;
        }
      }
      return true;
    }

    if (isa<CmpInst>(UI)) {
      continue;
    }
    if (isa<LoadInst>(UI)) {
      if (level) {
        for (auto U2 : UI->users()) {
          auto UI2 = cast<Instruction>(U2);
          todo.emplace_back(UI2, level - 1, UI);
        }
      }
      continue;
    }
    // storing into it.
    if (auto SI = dyn_cast<StoreInst>(UI)) {
      if (SI->getValueOperand() != prev) {
        continue;
      }
    }
    return false;
  }
  return true;
}

bool notCaptured(llvm::Value *V) { return notCapturedBefore(V, nullptr, 0); }

// Return true if guaranteed not to alias
// Return false if guaranteed to alias [with possible offset depending on flag].
// Return {} if no information is given.
#if LLVM_VERSION_MAJOR >= 16
std::optional<bool>
#else
llvm::Optional<bool>
#endif
arePointersGuaranteedNoAlias(TargetLibraryInfo &TLI, llvm::AAResults &AA,
                             llvm::LoopInfo &LI, llvm::Value *op0,
                             llvm::Value *op1, bool offsetAllowed) {
  auto lhs = getBaseObject(op0, offsetAllowed);
  auto rhs = getBaseObject(op1, offsetAllowed);

  if (lhs == rhs) {
    return false;
  }
  if (auto i1 = dyn_cast<Instruction>(op1))
    if (isa<ConstantPointerNull>(op0) &&
        hasMetadata(i1, LLVMContext::MD_nonnull)) {
      return true;
    }
  if (auto i0 = dyn_cast<Instruction>(op0))
    if (isa<ConstantPointerNull>(op1) &&
        hasMetadata(i0, LLVMContext::MD_nonnull)) {
      return true;
    }

  if (!lhs->getType()->isPointerTy() && !rhs->getType()->isPointerTy())
    return {};

  bool noalias_lhs = isNoAlias(lhs);
  bool noalias_rhs = isNoAlias(rhs);

  bool noalias[2] = {noalias_lhs, noalias_rhs};

  for (int i = 0; i < 2; i++) {
    Value *start = (i == 0) ? lhs : rhs;
    Value *end = (i == 0) ? rhs : lhs;
    if (noalias[i]) {
      if (noalias[1 - i]) {
        return true;
      }
      if (isa<Argument>(end)) {
        return true;
      }
      if (auto endi = dyn_cast<Instruction>(end)) {
        if (notCapturedBefore(start, endi, 0)) {
          return true;
        }
      }
    }
    if (auto ld = dyn_cast<LoadInst>(start)) {
      auto base = getBaseObject(ld->getOperand(0), /*offsetAllowed*/ false);
      if (isAllocationCall(base, TLI)) {
        if (isa<Argument>(end))
          return true;
        if (auto endi = dyn_cast<Instruction>(end))
          if (isNoAlias(end) || (notCapturedBefore(start, endi, 1))) {
            Instruction *starti = dyn_cast<Instruction>(start);
            if (!starti) {
              if (!isa<Argument>(start))
                continue;
              starti =
                  &cast<Argument>(start)->getParent()->getEntryBlock().front();
            }

            bool overwritten = false;
            allInstructionsBetween(
                LI, starti, endi, [&](Instruction *I) -> bool {
                  if (!I->mayWriteToMemory())
                    return /*earlyBreak*/ false;

                  if (writesToMemoryReadBy(nullptr, AA, TLI,
                                           /*maybeReader*/ ld,
                                           /*maybeWriter*/ I)) {
                    overwritten = true;
                    return /*earlyBreak*/ true;
                  }
                  return /*earlyBreak*/ false;
                });

            if (!overwritten) {
              return true;
            }
          }
      }
    }
  }

  return {};
}

bool isTargetNVPTX(llvm::Module &M) {
#if LLVM_VERSION_MAJOR > 20
  return M.getTargetTriple().getArch() == Triple::ArchType::nvptx ||
         M.getTargetTriple().getArch() == Triple::ArchType::nvptx64;
#else
  return M.getTargetTriple().find("nvptx") != std::string::npos;
#endif
}

static Value *constantInBoundsGEPHelper(llvm::IRBuilder<> &B, llvm::Type *type,
                                        llvm::Value *value,
                                        ArrayRef<unsigned> path) {
  SmallVector<Value *, 2> vals;
  vals.push_back(ConstantInt::get(B.getInt64Ty(), 0));
  for (auto v : path) {
    vals.push_back(ConstantInt::get(B.getInt32Ty(), v));
  }
  return B.CreateInBoundsGEP(type, value, vals);
}

llvm::Value *moveSRetToFromRoots(llvm::IRBuilder<> &B, llvm::Type *jltype,
                                 llvm::Value *sret, llvm::Type *root_ty,
                                 llvm::Value *rootRet, size_t rootOffset,
                                 SRetRootMovement direction) {
  std::deque<std::pair<llvm::Type *, std::vector<unsigned>>> todo;
  SmallVector<Value *> extracted;
  Value *val = sret;
  auto rootOffset0 = rootOffset;
  while (!todo.empty()) {
    auto cur = std::move(todo[0]);
    todo.pop_front();
    auto path = std::move(cur.second);
    auto ty = cur.first;

    if (auto PT = dyn_cast<PointerType>(ty)) {
      if (!isSpecialPtr(PT))
        continue;

      Value *loc = nullptr;
      switch (direction) {
      case SRetRootMovement::SRetPointerToRootPointer:
      case SRetRootMovement::SRetValueToRootPointer:
      case SRetRootMovement::RootPointerToSRetPointer:
      case SRetRootMovement::RootPointerToSRetValue:
        loc = constantInBoundsGEPHelper(B, root_ty, rootRet, rootOffset);
      default:
        llvm_unreachable("Unhandled");
      }

      switch (direction) {
      case SRetRootMovement::SRetPointerToRootPointer: {
        Value *outloc = constantInBoundsGEPHelper(B, jltype, sret, path);
        outloc = B.CreateLoad(ty, outloc);
        B.CreateStore(outloc, loc);
        break;
      }
      case SRetRootMovement::SRetValueToRootPointer: {
        Value *outloc = GradientUtils::extractMeta(B, sret, path);
        B.CreateStore(outloc, loc);
        break;
      }
      case SRetRootMovement::RootPointerToSRetValue: {
        loc = B.CreateLoad(ty, loc);
        val = B.CreateInsertValue(val, loc, path);
        break;
      }
      case SRetRootMovement::NullifySRetValue: {
        loc = getUndefinedValueForType(
            *B.GetInsertBlock()->getParent()->getParent(), ty, false);
        val = B.CreateInsertValue(val, loc, path);
        break;
      }
      case SRetRootMovement::RootPointerToSRetPointer: {
        Value *outloc = constantInBoundsGEPHelper(B, jltype, sret, path);
        loc = B.CreateLoad(ty, loc);
        extracted.push_back(loc);
        B.CreateStore(loc, outloc);
      }
      default:
        llvm_unreachable("Unhandled");
      }

      rootOffset += 1;
      continue;
    }

    if (auto AT = dyn_cast<ArrayType>(ty)) {
      for (size_t i = 0; i < AT->getNumElements(); i++) {
        std::vector<unsigned> path2(path);
        path2.push_back(i);
        todo.emplace_back(AT->getElementType(), path2);
      }
      continue;
    }

    if (auto VT = dyn_cast<VectorType>(ty)) {
      for (size_t i = 0; i < VT->getElementCount().getKnownMinValue(); i++) {
        std::vector<unsigned> path2(path);
        path2.push_back(i);
        todo.emplace_back(VT->getElementType(), path2);
      }
      continue;
    }

    if (auto ST = dyn_cast<StructType>(ty)) {
      for (size_t i = 0; i < ST->getNumElements(); i++) {
        std::vector<unsigned> path2(path);
        path2.push_back(i);
        todo.emplace_back(ST->getTypeAtIndex(i), path2);
      }
      continue;
    }
  }

  if (direction == SRetRootMovement::RootPointerToSRetPointer) {
    auto obj = getBaseObject(sret);
    auto PT = cast<PointerType>(obj->getType());
    assert(PT->getAddressSpace() == 0 || PT->getAddressSpace() == 10);
    if (PT->getAddressSpace() == 10 && extracted.size()) {
      extracted.insert(extracted.begin(), obj);
      auto JLT = PointerType::get(StructType::get(PT->getContext(), {}), 10);
      auto FT = FunctionType::get(JLT, {}, true);
      auto wb =
          B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
              "julia.write_barrier", FT);
      assert(obj->getType() == JLT);
      B.CreateCall(wb, extracted);
    }
  }

  CountTrackedPointers tracked(jltype);
  assert(rootOffset - rootOffset0 == tracked.count);

  return val;
}

void copyNonJLValueInto(llvm::IRBuilder<> &B, llvm::Type *curType,
                        llvm::Type *dstType, llvm::Value *dst,
                        llvm::ArrayRef<unsigned> dstPrefix0,
                        llvm::Type *srcType, llvm::Value *src,
                        llvm::ArrayRef<unsigned> srcPrefix0, bool shouldZero) {
  std::deque<
      std::tuple<llvm::Type *, std::vector<unsigned>, std::vector<unsigned>>>
      todo = {{curType,
               std::vector<unsigned>(dstPrefix0.begin(), dstPrefix0.end()),
               std::vector<unsigned>(srcPrefix0.begin(), srcPrefix0.end())}};

  auto &M = *B.GetInsertBlock()->getParent()->getParent();

  size_t numRootsSeen = 0;

  while (!todo.empty()) {
    auto cur = std::move(todo[0]);
    auto &&[ty, dstPrefix, srcPrefix] = cur;
    todo.pop_front();

    if (auto PT = dyn_cast<PointerType>(ty)) {
      if (PT->getAddressSpace() == 10) {
        numRootsSeen++;
        if (shouldZero) {
          Value *out = dst;
          if (dstPrefix.size() > 0)
            out = constantInBoundsGEPHelper(B, dstType, out, dstPrefix);
          B.CreateStore(getUndefinedValueForType(M, ty), out);
        }
      }
      // We don't actually need pointers either here
      continue;
    }

    if (auto AT = dyn_cast<ArrayType>(ty)) {
      for (size_t i = 0; i < AT->getNumElements(); i++) {
        std::vector<unsigned> nextDst(dstPrefix);
        std::vector<unsigned> nextSrc(srcPrefix);
        nextDst.push_back(i);
        nextSrc.push_back(i);
        todo.emplace_back(AT->getElementType(), std::move(nextDst),
                          std::move(nextSrc));
      }
      continue;
    }

    if (auto ST = dyn_cast<StructType>(curType)) {
      for (size_t i = 0; i < ST->getNumElements(); i++) {
        std::vector<unsigned> nextDst(dstPrefix);
        std::vector<unsigned> nextSrc(srcPrefix);
        nextDst.push_back(i);
        nextSrc.push_back(i);
        todo.emplace_back(ST->getElementType(i), std::move(nextDst),
                          std::move(nextSrc));
      }
      continue;
    }

    Value *out = dst;
    if (dstPrefix.size() > 0)
      out = constantInBoundsGEPHelper(B, dstType, out, dstPrefix);

    Value *in = src;
    if (srcPrefix.size() > 0)
      in = constantInBoundsGEPHelper(B, srcType, in, srcPrefix);

    auto ld = B.CreateLoad(ty, in);
    B.CreateStore(ld, out);
  }

  CountTrackedPointers tracked(curType);
  assert(numRootsSeen == tracked.count);
  (void)tracked;
  (void)numRootsSeen;
}
