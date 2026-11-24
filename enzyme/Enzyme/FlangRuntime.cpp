//===- FlangRuntime.cpp - LLVM flang runtime functions known to Enzyme ----===//
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
// Helper functions, emitted into the module being differentiated, for the
// reverse-mode derivatives of LLVM flang runtime functions. They walk the
// elements of arrays described by Fortran descriptors (CFI_cdesc_t, see
// flang/ISO_Fortran_binding.h).
//
//===----------------------------------------------------------------------===//

#include "FlangRuntime.h"
#include "Utils.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

/// The layout of a CFI_cdesc_t: base_addr, elem_len, version (int), rank,
/// type, attribute and extra (one byte each), then a lower bound, an extent
/// and a byte stride (sm) per dimension.
struct DescriptorLayout {
  LLVMContext &C;
  PointerType *I8PtrTy;
  IntegerType *IndexTy;
  uint64_t PointerBytes;
  uint64_t IndexBytes;

  explicit DescriptorLayout(Module &M)
      : C(M.getContext()), I8PtrTy(getInt8PtrTy(M.getContext())),
        IndexTy(M.getDataLayout().getIntPtrType(M.getContext())),
        PointerBytes(M.getDataLayout().getPointerSize()),
        IndexBytes(M.getDataLayout().getTypeStoreSize(
            M.getDataLayout().getIntPtrType(M.getContext()))) {}

  uint64_t rankOffset() const { return 2 * PointerBytes + 4; }
  uint64_t typeOffset() const { return 2 * PointerBytes + 5; }
  uint64_t dimOffset() const { return 2 * PointerBytes + 8; }

  Value *address(IRBuilder<> &B, Value *Descriptor, Value *Offset) const {
    return B.CreateInBoundsGEP(
        B.getInt8Ty(), B.CreatePointerCast(Descriptor, I8PtrTy), Offset);
  }
  Value *load(IRBuilder<> &B, Type *T, Value *Descriptor, Value *Offset) const {
    return B.CreateLoad(
        T, B.CreatePointerCast(address(B, Descriptor, Offset), getUnqual(T)));
  }
  Value *load(IRBuilder<> &B, Type *T, Value *Descriptor,
              uint64_t Offset) const {
    return load(B, T, Descriptor, ConstantInt::get(IndexTy, Offset));
  }

  Value *base(IRBuilder<> &B, Value *Descriptor) const {
    return load(B, I8PtrTy, Descriptor, uint64_t(0));
  }
  Value *elemLen(IRBuilder<> &B, Value *Descriptor) const {
    return load(B, IndexTy, Descriptor, PointerBytes);
  }
  Value *rank(IRBuilder<> &B, Value *Descriptor) const {
    return B.CreateZExt(load(B, B.getInt8Ty(), Descriptor, rankOffset()),
                        IndexTy);
  }
  /// The CFI type code, which is signed (CFI_type_other is -1).
  Value *type(IRBuilder<> &B, Value *Descriptor) const {
    return B.CreateSExt(load(B, B.getInt8Ty(), Descriptor, typeOffset()),
                        B.getInt32Ty());
  }
  /// Field \p Field (0: lower bound, 1: extent, 2: sm) of dimension \p Dim.
  Value *dim(IRBuilder<> &B, Value *Descriptor, Value *Dim,
             unsigned Field) const {
    Value *ByteOffset = B.CreateAdd(
        ConstantInt::get(IndexTy, dimOffset() + Field * IndexBytes),
        B.CreateMul(Dim, ConstantInt::get(IndexTy, 3 * IndexBytes)));
    return load(B, IndexTy, Descriptor, ByteOffset);
  }
  /// The size of a descriptor of rank \p Rank, without addendum.
  Value *size(IRBuilder<> &B, Value *Rank) const {
    return B.CreateAdd(
        ConstantInt::get(IndexTy, dimOffset()),
        B.CreateMul(Rank, ConstantInt::get(IndexTy, 3 * IndexBytes)));
  }
};

/// Element type classes of the reverse-mode helpers.
enum FlangTypeClass : int {
  /// No derivative: integers, logicals, characters, unsigned.
  Inactive = 0,
  /// REAL(4) and COMPLEX(4).
  Float = 1,
  /// REAL(8) and COMPLEX(8).
  Double = 2,
  /// Derived types and the other real and complex kinds.
  Unsupported = 3,
};

/// CFI type codes (flang/ISO_Fortran_binding.h) the helpers dispatch on.
enum CFIType : int {
  CFITypeFloat = 27,
  CFITypeDouble = 28,
  CFITypeFloatComplex = 34,
  CFITypeDoubleComplex = 35,
  CFITypeStruct = 42,
};

} // namespace

/// The name of a float or double type in helper names.
static StringRef fpName(Type *T) { return T->isDoubleTy() ? "f64" : "f32"; }

/// Returns the internal helper \p Name of type \p FT, and whether it still
/// needs a body.
static Function *getOrInsertHelper(Module &M, StringRef Name, FunctionType *FT,
                                   bool &NeedsBody) {
  auto *F = cast<Function>(M.getOrInsertFunction(Name, FT).getCallee());
  NeedsBody = F->empty();
  if (NeedsBody) {
    F->setLinkage(Function::LinkageTypes::InternalLinkage);
    F->addFnAttr(Attribute::NoUnwind);
  }
  return F;
}

/// Emits `for (i = 0; i < N; ++i) Body(B, i)` at the insertion point of
/// \p B and leaves \p B after the loop. \p Body may add blocks.
static void emitLoop(IRBuilder<> &B, Value *N, const Twine &Name,
                     function_ref<void(IRBuilder<> &, Value *)> Body) {
  Function *F = B.GetInsertBlock()->getParent();
  LLVMContext &C = F->getContext();
  BasicBlock *Pre = B.GetInsertBlock();
  BasicBlock *Loop = BasicBlock::Create(C, Name, F);
  BasicBlock *Exit = BasicBlock::Create(C, Name + ".end", F);
  Value *Zero = ConstantInt::get(N->getType(), 0);
  B.CreateCondBr(B.CreateICmpSGT(N, Zero), Loop, Exit);

  B.SetInsertPoint(Loop);
  PHINode *I = B.CreatePHI(N->getType(), 2, Name + ".i");
  I->addIncoming(Zero, Pre);
  Body(B, I);
  Value *Next = B.CreateAdd(I, ConstantInt::get(N->getType(), 1), "",
                            /*HasNUW*/ true, /*HasNSW*/ true);
  I->addIncoming(Next, B.GetInsertBlock());
  B.CreateCondBr(B.CreateICmpSLT(Next, N), Loop, Exit);
  B.SetInsertPoint(Exit);
}

/// An alloca in the entry block of the function of \p B, set to \p Init.
static AllocaInst *createSlot(IRBuilder<> &B, Value *Init) {
  Function *F = B.GetInsertBlock()->getParent();
  IRBuilder<> EB(&F->getEntryBlock(), F->getEntryBlock().begin());
  AllocaInst *A = EB.CreateAlloca(Init->getType());
  B.CreateStore(Init, A);
  return A;
}

static FunctionCallee getMalloc(Module &M) {
  DescriptorLayout L(M);
  return M.getOrInsertFunction("malloc", L.I8PtrTy, L.IndexTy);
}

static FunctionCallee getFree(Module &M) {
  DescriptorLayout L(M);
  return M.getOrInsertFunction("free", Type::getVoidTy(M.getContext()),
                               L.I8PtrTy);
}

/// i64 __enzyme_flang_element_count(i8 *desc): the number of elements.
static Function *getOrInsertElementCount(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F = getOrInsertHelper(
      M, "__enzyme_flang_element_count",
      FunctionType::get(L.IndexTy, {L.I8PtrTy}, false), NeedsBody);
  if (!NeedsBody)
    return F;
  F->addFnAttr(Attribute::AlwaysInline);
  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  Value *Descriptor = F->getArg(0);
  AllocaInst *Count = createSlot(B, ConstantInt::get(L.IndexTy, 1));
  emitLoop(B, L.rank(B, Descriptor), "dim", [&](IRBuilder<> &B, Value *K) {
    B.CreateStore(
        B.CreateMul(B.CreateLoad(L.IndexTy, Count), L.dim(B, Descriptor, K, 1)),
        Count);
  });
  B.CreateRet(B.CreateLoad(L.IndexTy, Count));
  return F;
}

/// i64 __enzyme_flang_element_offset(i8 *desc, i64 i): the byte offset from
/// base_addr of element \p i, counting in array element order. A scalar
/// (rank 0) has every element at offset 0.
static Function *getOrInsertElementOffset(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F = getOrInsertHelper(
      M, "__enzyme_flang_element_offset",
      FunctionType::get(L.IndexTy, {L.I8PtrTy, L.IndexTy}, false), NeedsBody);
  if (!NeedsBody)
    return F;
  F->addFnAttr(Attribute::AlwaysInline);
  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  Value *Descriptor = F->getArg(0);
  AllocaInst *Rest = createSlot(B, F->getArg(1));
  AllocaInst *Offset = createSlot(B, ConstantInt::get(L.IndexTy, 0));
  emitLoop(B, L.rank(B, Descriptor), "dim", [&](IRBuilder<> &B, Value *K) {
    Value *Extent = L.dim(B, Descriptor, K, 1);
    Value *Remaining = B.CreateLoad(L.IndexTy, Rest);
    B.CreateStore(B.CreateUDiv(Remaining, Extent), Rest);
    Value *Index = B.CreateURem(Remaining, Extent);
    B.CreateStore(B.CreateAdd(B.CreateLoad(L.IndexTy, Offset),
                              B.CreateMul(Index, L.dim(B, Descriptor, K, 2))),
                  Offset);
  });
  B.CreateRet(B.CreateLoad(L.IndexTy, Offset));
  return F;
}

/// The address of element \p I of the array described by \p Descriptor.
static Value *elementAddress(IRBuilder<> &B, const DescriptorLayout &L,
                             Value *Descriptor, Value *I) {
  Module &M = *B.GetInsertBlock()->getModule();
  Value *ByteOffset =
      B.CreateCall(getOrInsertElementOffset(M), {Descriptor, I});
  return B.CreateGEP(B.getInt8Ty(), L.base(B, Descriptor), ByteOffset);
}

/// i32 __enzyme_flang_type_class(i32 type): the FlangTypeClass of a CFI type
/// code.
static Function *getOrInsertTypeClass(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Type *I32 = Type::getInt32Ty(L.C);
  Function *F =
      getOrInsertHelper(M, "__enzyme_flang_type_class",
                        FunctionType::get(I32, {I32}, false), NeedsBody);
  if (!NeedsBody)
    return F;
  F->addFnAttr(Attribute::AlwaysInline);
  BasicBlock *Entry = BasicBlock::Create(L.C, "entry", F);
  auto Ret = [&](FlangTypeClass Class) {
    BasicBlock *BB = BasicBlock::Create(L.C, "class", F);
    ReturnInst::Create(L.C, ConstantInt::get(I32, Class), BB);
    return BB;
  };
  BasicBlock *Unsupported = Ret(FlangTypeClass::Unsupported);
  BasicBlock *Inactive = Ret(FlangTypeClass::Inactive);
  BasicBlock *Float = Ret(FlangTypeClass::Float);
  BasicBlock *Double = Ret(FlangTypeClass::Double);
  IRBuilder<> B(Entry);
  SwitchInst *S = B.CreateSwitch(F->getArg(0), Unsupported);
  auto Add = [&](int Code, BasicBlock *BB) {
    S->addCase(ConstantInt::get(cast<IntegerType>(I32), Code), BB);
  };
  // CFI_type_signed_char .. CFI_type_ptrdiff_t
  for (int Code = 1; Code <= 24; ++Code)
    Add(Code, Inactive);
  // CFI_type_Bool, CFI_type_char, CFI_type_cptr
  for (int Code = 39; Code <= 41; ++Code)
    Add(Code, Inactive);
  // CFI_type_char16_t .. CFI_type_uint128_t
  for (int Code = 43; Code <= 49; ++Code)
    Add(Code, Inactive);
  Add(CFITypeFloat, Float);
  Add(CFITypeFloatComplex, Float);
  Add(CFITypeDouble, Double);
  Add(CFITypeDoubleComplex, Double);
  return F;
}

/// Calls _FortranAReportFatalUserError(\p Message, \p File, \p Line), which
/// does not return.
static void emitFatalError(IRBuilder<> &B, const DescriptorLayout &L,
                           StringRef Message, Value *File, Value *Line) {
  Module &M = *B.GetInsertBlock()->getModule();
  FunctionCallee Fatal = M.getOrInsertFunction(
      "_FortranAReportFatalUserError", Type::getVoidTy(L.C), L.I8PtrTy,
      L.I8PtrTy, Type::getInt32Ty(L.C));
  Value *MessageString =
      B.CreatePointerCast(B.CreateGlobalString(Message), L.I8PtrTy);
  B.CreateCall(Fatal, {MessageString, File, Line});
  B.CreateUnreachable();
}

Function *getOrInsertFlangDescriptorSave(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F = getOrInsertHelper(
      M, "__enzyme_flang_descriptor_save",
      FunctionType::get(L.I8PtrTy, {L.I8PtrTy}, false), NeedsBody);
  if (!NeedsBody)
    return F;
  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  Value *Descriptor = F->getArg(0);
  Value *Size = L.size(B, L.rank(B, Descriptor));
  Value *Copy = B.CreateCall(getMalloc(M), {Size});
  B.CreateMemCpy(Copy, MaybeAlign(), Descriptor, MaybeAlign(), Size);
  B.CreateRet(Copy);
  return F;
}

Function *getOrInsertFlangShadowSave(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F = getOrInsertHelper(
      M, "__enzyme_flang_shadow_save",
      FunctionType::get(L.I8PtrTy, {L.I8PtrTy, L.I8PtrTy}, false), NeedsBody);
  if (!NeedsBody)
    return F;
  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  Value *Copy = B.CreateCall(getOrInsertFlangDescriptorSave(M), {F->getArg(1)});
  B.CreateStore(L.base(B, F->getArg(0)),
                B.CreatePointerCast(Copy, getUnqual(L.I8PtrTy)));
  B.CreateRet(Copy);
  return F;
}

Function *getOrInsertFlangShadowAssignRealloc(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F =
      getOrInsertHelper(M, "__enzyme_flang_shadow_realloc",
                        FunctionType::get(L.I8PtrTy,
                                          {L.I8PtrTy, L.I8PtrTy, L.I8PtrTy,
                                           L.I8PtrTy, Type::getInt32Ty(L.C)},
                                          false),
                        NeedsBody);
  if (!NeedsBody)
    return F;
  Value *Shadow = F->getArg(0);
  Value *Primal = F->getArg(1);
  Value *Before = F->getArg(2);
  Value *File = F->getArg(3);
  Value *Line = F->getArg(4);

  // The assignment (re)allocated `to` if its data moved or its shape
  // changed.
  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  AllocaInst *Changed =
      createSlot(B, B.CreateICmpNE(L.base(B, Primal), L.base(B, Before)));
  emitLoop(B, L.rank(B, Primal), "dim", [&](IRBuilder<> &B, Value *K) {
    Value *ExtentDiffers =
        B.CreateICmpNE(L.dim(B, Primal, K, 1), L.dim(B, Before, K, 1));
    B.CreateStore(
        B.CreateOr(B.CreateLoad(B.getInt1Ty(), Changed), ExtentDiffers),
        Changed);
  });
  Value *Needed = B.CreateOr(B.CreateLoad(B.getInt1Ty(), Changed),
                             B.CreateIsNull(L.base(B, Shadow)));
  BasicBlock *Realloc = BasicBlock::Create(L.C, "realloc", F);
  BasicBlock *Keep = BasicBlock::Create(L.C, "keep", F);
  B.CreateCondBr(Needed, Realloc, Keep);

  B.SetInsertPoint(Keep);
  B.CreateRet(ConstantPointerNull::get(L.I8PtrTy));

  // The components of derived types hold descriptors, which zeroed memory
  // does not: their derivatives are not supported.
  B.SetInsertPoint(Realloc);
  BasicBlock *Error = BasicBlock::Create(L.C, "error", F);
  BasicBlock *Alloc = BasicBlock::Create(L.C, "alloc", F);
  B.CreateCondBr(B.CreateICmpEQ(L.type(B, Primal), B.getInt32(CFITypeStruct)),
                 Error, Alloc);
  B.SetInsertPoint(Error);
  emitFatalError(B, L,
                 "Enzyme: the reverse pass of an assignment that allocates a "
                 "derived type is not supported",
                 File, Line);

  // Give the shadow new, zeroed memory with the shape of `to`. The old
  // shadow data stays allocated: the reverse pass of earlier code may still
  // need its adjoint.
  B.SetInsertPoint(Alloc);
  Value *Old = B.CreateCall(getOrInsertFlangDescriptorSave(M), {Shadow});
  Value *Bytes = B.CreateMul(B.CreateCall(getOrInsertElementCount(M), {Primal}),
                             L.elemLen(B, Primal));
  // malloc(0) may return null, which would read as unallocated.
  Value *Data = B.CreateCall(
      getMalloc(M), {B.CreateAdd(Bytes, ConstantInt::get(L.IndexTy, 1))});
  B.CreateMemSet(Data, B.getInt8(0), Bytes, MaybeAlign());
  B.CreateMemCpy(Shadow, MaybeAlign(), Primal, MaybeAlign(),
                 L.size(B, L.rank(B, Primal)));
  B.CreateStore(Data, B.CreatePointerCast(Shadow, getUnqual(L.I8PtrTy)));
  B.CreateRet(Old);
  return F;
}

/// void __enzyme_flang_assign_reverse_<To>_<From>(i8 *dto, i8 *dfrom,
///     i64 n, i64 ncomp_to, i64 ncomp_from)
/// moves the adjoint of the \p n elements of dto (\p ncomp_to components of
/// type \p ToTy each) into those of dfrom (\p ncomp_from of type \p FromTy),
/// if dfrom is not null, and zeroes it. The adjoint goes through a
/// temporary, since dto and dfrom may overlap.
static Function *getOrInsertAssignReverseKernel(Module &M, Type *ToTy,
                                                Type *FromTy) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F = getOrInsertHelper(
      M,
      ("__enzyme_flang_assign_reverse_" + fpName(ToTy) + "_" + fpName(FromTy))
          .str(),
      FunctionType::get(Type::getVoidTy(L.C),
                        {L.I8PtrTy, L.I8PtrTy, L.IndexTy, L.IndexTy, L.IndexTy},
                        false),
      NeedsBody);
  if (!NeedsBody)
    return F;
  Value *ShadowTo = F->getArg(0);
  Value *ShadowFrom = F->getArg(1);
  Value *N = F->getArg(2);
  Value *ComponentsTo = F->getArg(3);
  Value *ComponentsFrom = F->getArg(4);

  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  Value *Total = B.CreateMul(N, ComponentsTo);
  uint64_t ToTyBytes = M.getDataLayout().getTypeStoreSize(ToTy);
  Value *Temporary = B.CreatePointerCast(
      B.CreateCall(
          getMalloc(M),
          {B.CreateMul(Total, ConstantInt::get(L.IndexTy, ToTyBytes))}),
      getUnqual(ToTy));

  emitLoop(B, N, "zero", [&](IRBuilder<> &B, Value *I) {
    Value *Element =
        B.CreatePointerCast(elementAddress(B, L, ShadowTo, I), getUnqual(ToTy));
    emitLoop(B, ComponentsTo, "zero.comp", [&](IRBuilder<> &B, Value *J) {
      Value *Address = B.CreateInBoundsGEP(ToTy, Element, J);
      Value *Saved = B.CreateInBoundsGEP(
          ToTy, Temporary, B.CreateAdd(B.CreateMul(I, ComponentsTo), J));
      B.CreateStore(B.CreateLoad(ToTy, Address), Saved);
      B.CreateStore(Constant::getNullValue(ToTy), Address);
    });
  });

  BasicBlock *Accumulate = BasicBlock::Create(L.C, "accumulate", F);
  BasicBlock *Done = BasicBlock::Create(L.C, "done", F);
  B.CreateCondBr(B.CreateIsNull(ShadowFrom), Done, Accumulate);
  B.SetInsertPoint(Accumulate);
  // REAL = COMPLEX takes the real part; COMPLEX = REAL has a zero imaginary
  // part.
  Value *Components =
      B.CreateSelect(B.CreateICmpULT(ComponentsTo, ComponentsFrom),
                     ComponentsTo, ComponentsFrom);
  emitLoop(B, N, "acc", [&](IRBuilder<> &B, Value *I) {
    Value *Element = B.CreatePointerCast(elementAddress(B, L, ShadowFrom, I),
                                         getUnqual(FromTy));
    emitLoop(B, Components, "acc.comp", [&](IRBuilder<> &B, Value *J) {
      Value *Address = B.CreateInBoundsGEP(FromTy, Element, J);
      Value *Saved = B.CreateInBoundsGEP(
          ToTy, Temporary, B.CreateAdd(B.CreateMul(I, ComponentsTo), J));
      Value *Adjoint = B.CreateFPCast(B.CreateLoad(ToTy, Saved), FromTy);
      B.CreateStore(B.CreateFAdd(B.CreateLoad(FromTy, Address), Adjoint),
                    Address);
    });
  });
  B.CreateBr(Done);

  B.SetInsertPoint(Done);
  B.CreateCall(getFree(M), {B.CreatePointerCast(Temporary, L.I8PtrTy)});
  B.CreateRetVoid();
  return F;
}

Function *getOrInsertFlangAssignReverse(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Type *I32 = Type::getInt32Ty(L.C);
  Function *F = getOrInsertHelper(
      M, "__enzyme_flang_assign_reverse",
      FunctionType::get(
          Type::getVoidTy(L.C),
          {L.I8PtrTy, L.I8PtrTy, L.I8PtrTy, L.I8PtrTy, L.I8PtrTy, I32}, false),
      NeedsBody);
  if (!NeedsBody)
    return F;
  Value *Shadow = F->getArg(0);
  Value *ShadowTo = F->getArg(1);
  Value *ShadowFrom = F->getArg(2);
  Value *Old = F->getArg(3);
  Value *File = F->getArg(4);
  Value *Line = F->getArg(5);
  Function *Class = getOrInsertTypeClass(M);

  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  Value *ToClass = B.CreateCall(Class, {L.type(B, ShadowTo)});
  // An inactive or missing dfrom receives nothing.
  AllocaInst *FromClass =
      createSlot(B, ConstantInt::get(I32, FlangTypeClass::Inactive));
  AllocaInst *FromLen = createSlot(B, ConstantInt::get(L.IndexTy, 0));
  BasicBlock *HasFrom = BasicBlock::Create(L.C, "from", F);
  BasicBlock *Check = BasicBlock::Create(L.C, "check", F);
  B.CreateCondBr(B.CreateIsNull(ShadowFrom), Check, HasFrom);
  B.SetInsertPoint(HasFrom);
  B.CreateStore(B.CreateCall(Class, {L.type(B, ShadowFrom)}), FromClass);
  B.CreateStore(L.elemLen(B, ShadowFrom), FromLen);
  B.CreateBr(Check);

  B.SetInsertPoint(Check);
  Value *FromClassValue = B.CreateLoad(I32, FromClass);
  Value *Bad = B.CreateOr(
      B.CreateICmpEQ(ToClass, B.getInt32(FlangTypeClass::Unsupported)),
      B.CreateICmpEQ(FromClassValue, B.getInt32(FlangTypeClass::Unsupported)));
  BasicBlock *Error = BasicBlock::Create(L.C, "error", F);
  BasicBlock *Dispatch = BasicBlock::Create(L.C, "dispatch", F);
  BasicBlock *Restore = BasicBlock::Create(L.C, "restore", F);
  B.CreateCondBr(Bad, Error, Dispatch);

  B.SetInsertPoint(Error);
  emitFatalError(B, L,
                 "Enzyme: the reverse pass of a Fortran assignment supports "
                 "REAL and COMPLEX of kinds 4 and 8 only",
                 File, Line);

  // Call the kernel for the element types of dto and dfrom.
  B.SetInsertPoint(Dispatch);
  Value *FromInactive =
      B.CreateICmpEQ(FromClassValue, B.getInt32(FlangTypeClass::Inactive));
  Value *FromKey = B.CreateSelect(FromInactive, ToClass, FromClassValue);
  Value *FromPtr = B.CreateSelect(
      FromInactive, ConstantPointerNull::get(L.I8PtrTy), ShadowFrom);
  Value *N = B.CreateCall(getOrInsertElementCount(M), {ShadowTo});
  auto CompBytes = [&](Value *C) {
    return B.CreateSelect(B.CreateICmpEQ(C, B.getInt32(FlangTypeClass::Double)),
                          ConstantInt::get(L.IndexTy, 8),
                          ConstantInt::get(L.IndexTy, 4));
  };
  Value *ComponentsTo =
      B.CreateUDiv(L.elemLen(B, ShadowTo), CompBytes(ToClass));
  Value *ComponentsFrom =
      B.CreateUDiv(B.CreateLoad(L.IndexTy, FromLen), CompBytes(FromKey));
  Value *Key = B.CreateAdd(B.CreateMul(ToClass, B.getInt32(4)), FromKey);
  // dto inactive: nothing to move.
  SwitchInst *S = B.CreateSwitch(Key, Restore);
  Type *FPTys[] = {Type::getFloatTy(L.C), Type::getDoubleTy(L.C)};
  for (int ToIndex = 0; ToIndex < 2; ++ToIndex)
    for (int FromIndex = 0; FromIndex < 2; ++FromIndex) {
      BasicBlock *BB = BasicBlock::Create(L.C, "kernel", F);
      S->addCase(B.getInt32((ToIndex + 1) * 4 + (FromIndex + 1)), BB);
      IRBuilder<> KB(BB);
      KB.CreateCall(
          getOrInsertAssignReverseKernel(M, FPTys[ToIndex], FPTys[FromIndex]),
          {ShadowTo, FromPtr, N, ComponentsTo, ComponentsFrom});
      KB.CreateBr(Restore);
    }

  // If the augmented forward pass gave the shadow of `to` new memory, free
  // it and restore the shadow descriptor from before the assignment.
  B.SetInsertPoint(Restore);
  BasicBlock *HasOld = BasicBlock::Create(L.C, "old", F);
  BasicBlock *Done = BasicBlock::Create(L.C, "done", F);
  B.CreateCondBr(B.CreateIsNull(Old), Done, HasOld);
  B.SetInsertPoint(HasOld);
  B.CreateCall(getFree(M), {L.base(B, ShadowTo)});
  B.CreateMemCpy(Shadow, MaybeAlign(), Old, MaybeAlign(),
                 L.size(B, L.rank(B, Old)));
  B.CreateCall(getFree(M), {Old});
  B.CreateBr(Done);

  B.SetInsertPoint(Done);
  B.CreateCall(getFree(M), {ShadowTo});
  B.CreateCall(getFree(M), {ShadowFrom});
  B.CreateRetVoid();
  return F;
}

Function *getOrInsertFlangMaskSave(Module &M) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F = getOrInsertHelper(
      M, "__enzyme_flang_mask_save",
      FunctionType::get(L.I8PtrTy, {L.I8PtrTy, L.I8PtrTy}, false), NeedsBody);
  if (!NeedsBody)
    return F;
  Value *Mask = F->getArg(0);
  Value *Array = F->getArg(1);

  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  BasicBlock *Save = BasicBlock::Create(L.C, "save", F);
  BasicBlock *None = BasicBlock::Create(L.C, "none", F);
  B.CreateCondBr(B.CreateIsNull(Mask), None, Save);
  B.SetInsertPoint(None);
  B.CreateRet(ConstantPointerNull::get(L.I8PtrTy));

  B.SetInsertPoint(Save);
  Value *N = B.CreateCall(getOrInsertElementCount(M), {Array});
  Value *Bits = B.CreateCall(getMalloc(M),
                             {B.CreateAdd(N, ConstantInt::get(L.IndexTy, 1))});
  Value *ElementBytes = L.elemLen(B, Mask);
  AllocaInst *AnyByte = createSlot(B, B.getInt8(0));
  emitLoop(B, N, "elem", [&](IRBuilder<> &B, Value *I) {
    // A LOGICAL is true if any of its bytes is not zero.
    Value *Element = elementAddress(B, L, Mask, I);
    B.CreateStore(B.getInt8(0), AnyByte);
    emitLoop(B, ElementBytes, "byte", [&](IRBuilder<> &B, Value *J) {
      Value *Byte = B.CreateLoad(
          B.getInt8Ty(), B.CreateInBoundsGEP(B.getInt8Ty(), Element, J));
      B.CreateStore(B.CreateOr(B.CreateLoad(B.getInt8Ty(), AnyByte), Byte),
                    AnyByte);
    });
    Value *True =
        B.CreateICmpNE(B.CreateLoad(B.getInt8Ty(), AnyByte), B.getInt8(0));
    B.CreateStore(B.CreateZExt(True, B.getInt8Ty()),
                  B.CreateInBoundsGEP(B.getInt8Ty(), Bits, I));
  });
  B.CreateRet(Bits);
  return F;
}

Function *getOrInsertFlangSumReverse(Module &M, Type *FPTy) {
  DescriptorLayout L(M);
  bool NeedsBody;
  Function *F =
      getOrInsertHelper(M, ("__enzyme_flang_sum_reverse_" + fpName(FPTy)).str(),
                        FunctionType::get(Type::getVoidTy(L.C),
                                          {L.I8PtrTy, L.I8PtrTy, FPTy}, false),
                        NeedsBody);
  if (!NeedsBody)
    return F;
  Value *ShadowArray = F->getArg(0);
  Value *Bits = F->getArg(1);
  Value *ReturnAdjoint = F->getArg(2);

  IRBuilder<> B(BasicBlock::Create(L.C, "entry", F));
  Value *N = B.CreateCall(getOrInsertElementCount(M), {ShadowArray});
  Value *NoMask = B.CreateIsNull(Bits);
  emitLoop(B, N, "elem", [&](IRBuilder<> &B, Value *I) {
    BasicBlock *Test = BasicBlock::Create(L.C, "mask", F);
    BasicBlock *Add = BasicBlock::Create(L.C, "add", F);
    BasicBlock *Next = BasicBlock::Create(L.C, "next", F);
    B.CreateCondBr(NoMask, Add, Test);
    B.SetInsertPoint(Test);
    Value *Bit = B.CreateLoad(B.getInt8Ty(),
                              B.CreateInBoundsGEP(B.getInt8Ty(), Bits, I));
    B.CreateCondBr(B.CreateICmpNE(Bit, B.getInt8(0)), Add, Next);
    B.SetInsertPoint(Add);
    Value *Address = B.CreatePointerCast(elementAddress(B, L, ShadowArray, I),
                                         getUnqual(FPTy));
    B.CreateStore(B.CreateFAdd(B.CreateLoad(FPTy, Address), ReturnAdjoint),
                  Address);
    B.CreateBr(Next);
    B.SetInsertPoint(Next);
  });
  B.CreateCall(getFree(M), {ShadowArray});
  B.CreateRetVoid();
  return F;
}
