//===- LibraryFuncs.h - Utilities for handling library functions ---------===//
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
// Automatically Synthesize Fast Gradients}, author = {Moses, William S. and
// Churavy, Valentin}, booktitle = {Advances in Neural Information Processing
// Systems 33}, year = {2020}, note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file defines miscelaious utilities for handling library functions.
//
//===----------------------------------------------------------------------===//

#ifndef LIBRARYFUNCS_H_
#define LIBRARYFUNCS_H_

#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>

#include "Utils.h"

class GradientUtils;
extern llvm::StringMap<std::function<llvm::Value *(
    llvm::IRBuilder<> &, llvm::CallInst *, llvm::ArrayRef<llvm::Value *>,
    GradientUtils *)>>
    shadowHandlers;
extern llvm::StringMap<
    std::function<llvm::CallInst *(llvm::IRBuilder<> &, llvm::Value *)>>
    shadowErasers;

/// Return whether a given function is a known C/C++ memory allocation function
/// For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
static inline bool isAllocationFunction(const llvm::StringRef name,
                                        const llvm::TargetLibraryInfo &TLI) {
  if (name == "enzyme_allocator")
    return true;
  if (name == "calloc" || name == "malloc")
    return true;
  if (name == "swift_allocObject")
    return true;
  if (name == "__rust_alloc" || name == "__rust_alloc_zeroed")
    return true;
  if (name == "julia.gc_alloc_obj" || name == "jl_gc_alloc_typed" ||
      name == "ijl_gc_alloc_typed")
    return true;
  if (shadowHandlers.find(name) != shadowHandlers.end())
    return true;

  using namespace llvm;
  llvm::LibFunc libfunc;
  if (!TLI.getLibFunc(name, libfunc))
    return false;

  switch (libfunc) {
  case LibFunc_malloc: // malloc(unsigned int);
  case LibFunc_valloc: // valloc(unsigned int);

  case LibFunc_Znwj:                // new(unsigned int);
  case LibFunc_ZnwjRKSt9nothrow_t:  // new(unsigned int, nothrow);
  case LibFunc_ZnwjSt11align_val_t: // new(unsigned int, align_val_t)
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t: // new(unsigned int,
                                                  // align_val_t, nothrow)

  case LibFunc_Znwm:                // new(unsigned long);
  case LibFunc_ZnwmRKSt9nothrow_t:  // new(unsigned long, nothrow);
  case LibFunc_ZnwmSt11align_val_t: // new(unsigned long, align_val_t)
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t: // new(unsigned long,
                                                  // align_val_t, nothrow)

  case LibFunc_Znaj:                // new[](unsigned int);
  case LibFunc_ZnajRKSt9nothrow_t:  // new[](unsigned int, nothrow);
  case LibFunc_ZnajSt11align_val_t: // new[](unsigned int, align_val_t)
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t: // new[](unsigned int,
                                                  // align_val_t, nothrow)

  case LibFunc_Znam:                // new[](unsigned long);
  case LibFunc_ZnamRKSt9nothrow_t:  // new[](unsigned long, nothrow);
  case LibFunc_ZnamSt11align_val_t: // new[](unsigned long, align_val_t)
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t: // new[](unsigned long,
                                                  // align_val_t, nothrow)

  case LibFunc_msvc_new_int:               // new(unsigned int);
  case LibFunc_msvc_new_int_nothrow:       // new(unsigned int, nothrow);
  case LibFunc_msvc_new_longlong:          // new(unsigned long long);
  case LibFunc_msvc_new_longlong_nothrow:  // new(unsigned long long, nothrow);
  case LibFunc_msvc_new_array_int:         // new[](unsigned int);
  case LibFunc_msvc_new_array_int_nothrow: // new[](unsigned int, nothrow);
  case LibFunc_msvc_new_array_longlong:    // new[](unsigned long long);
  case LibFunc_msvc_new_array_longlong_nothrow: // new[](unsigned long long,
                                                // nothrow);

    // TODO strdup, strndup

    // TODO call, realloc, reallocf

    // TODO (perhaps) posix_memalign
    return true;
  default:
    return false;
  }
}

/// Return whether a given function is a known C/C++ memory deallocation
/// function For updating below one should read MemoryBuiltins.cpp,
/// TargetLibraryInfo.cpp
static inline bool isDeallocationFunction(const llvm::StringRef name,
                                          const llvm::TargetLibraryInfo &TLI) {
  using namespace llvm;
  llvm::LibFunc libfunc;
  if (!TLI.getLibFunc(name, libfunc)) {
    if (name == "free")
      return true;
    if (name == "__rust_dealloc")
      return true;
    if (name == "swift_release")
      return true;
    return false;
  }

  switch (libfunc) {
  // void free(void*);
  case LibFunc_free:

  // void operator delete[](void*);
  case LibFunc_ZdaPv:
  // void operator delete(void*);
  case LibFunc_ZdlPv:
  // void operator delete[](void*);
  case LibFunc_msvc_delete_array_ptr32:
  // void operator delete[](void*);
  case LibFunc_msvc_delete_array_ptr64:
  // void operator delete(void*);
  case LibFunc_msvc_delete_ptr32:
  // void operator delete(void*);
  case LibFunc_msvc_delete_ptr64:

  // void operator delete[](void*, nothrow);
  case LibFunc_ZdaPvRKSt9nothrow_t:
  // void operator delete[](void*, unsigned int);
  case LibFunc_ZdaPvj:
  // void operator delete[](void*, unsigned long);
  case LibFunc_ZdaPvm:
  // void operator delete(void*, nothrow);
  case LibFunc_ZdlPvRKSt9nothrow_t:
  // void operator delete(void*, unsigned int);
  case LibFunc_ZdlPvj:
  // void operator delete(void*, unsigned long);
  case LibFunc_ZdlPvm:
  // void operator delete(void*, align_val_t)
  case LibFunc_ZdlPvSt11align_val_t:
  // void operator delete[](void*, align_val_t)
  case LibFunc_ZdaPvSt11align_val_t:
  // void operator delete[](void*, unsigned int);
  case LibFunc_msvc_delete_array_ptr32_int:
  // void operator delete[](void*, nothrow);
  case LibFunc_msvc_delete_array_ptr32_nothrow:
  // void operator delete[](void*, unsigned long long);
  case LibFunc_msvc_delete_array_ptr64_longlong:
  // void operator delete[](void*, nothrow);
  case LibFunc_msvc_delete_array_ptr64_nothrow:
  // void operator delete(void*, unsigned int);
  case LibFunc_msvc_delete_ptr32_int:
  // void operator delete(void*, nothrow);
  case LibFunc_msvc_delete_ptr32_nothrow:
  // void operator delete(void*, unsigned long long);
  case LibFunc_msvc_delete_ptr64_longlong:
  // void operator delete(void*, nothrow);
  case LibFunc_msvc_delete_ptr64_nothrow:
  // void operator delete(void*, align_val_t, nothrow)
  case LibFunc_ZdlPvSt11align_val_tRKSt9nothrow_t:
  // void operator delete[](void*, align_val_t, nothrow)
  case LibFunc_ZdaPvSt11align_val_tRKSt9nothrow_t:
    return true;
  default:
    return false;
  }
}

static inline void zeroKnownAllocation(llvm::IRBuilder<> &bb,
                                       llvm::Value *toZero,
                                       llvm::ArrayRef<llvm::Value *> argValues,
                                       const llvm::StringRef funcName,
                                       const llvm::TargetLibraryInfo &TLI,
                                       llvm::CallInst *orig) {
  using namespace llvm;
  assert(isAllocationFunction(funcName, TLI));

  // Don't re-zero an already-zero buffer
  if (funcName == "calloc" || funcName == "__rust_alloc_zeroed")
    return;

  Value *allocSize = argValues[0];
  if (funcName == "julia.gc_alloc_obj" || funcName == "jl_gc_alloc_typed" ||
      funcName == "ijl_gc_alloc_typed") {
    allocSize = argValues[1];
  }
  if (funcName == "enzyme_allocator") {
    auto index = getAllocationIndexFromCall(orig);
#if LLVM_VERSION_MAJOR >= 16
    allocSize = argValues[index.value()];
#else
    allocSize = argValues[index.getValue()];
#endif
  }
  Value *dst_arg = toZero;

  if (dst_arg->getType()->isIntegerTy())
    dst_arg =
        bb.CreateIntToPtr(dst_arg, Type::getInt8PtrTy(toZero->getContext()));
  else
    dst_arg = bb.CreateBitCast(
        dst_arg,
        Type::getInt8PtrTy(toZero->getContext(),
                           toZero->getType()->getPointerAddressSpace()));

  auto val_arg = ConstantInt::get(Type::getInt8Ty(toZero->getContext()), 0);
  auto len_arg =
      bb.CreateZExtOrTrunc(allocSize, Type::getInt64Ty(toZero->getContext()));
  auto volatile_arg = ConstantInt::getFalse(toZero->getContext());

  Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
  Type *tys[] = {dst_arg->getType(), len_arg->getType()};

  auto memset = cast<CallInst>(bb.CreateCall(
      Intrinsic::getDeclaration(bb.GetInsertBlock()->getParent()->getParent(),
                                Intrinsic::memset, tys),
      nargs));
  memset->addParamAttr(0, Attribute::NonNull);
  if (auto CI = dyn_cast<ConstantInt>(allocSize)) {
    auto derefBytes = CI->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 14
    memset->addDereferenceableParamAttr(0, derefBytes);
    memset->setAttributes(
        memset->getAttributes().addDereferenceableOrNullParamAttr(
            memset->getContext(), 0, derefBytes));
#else
    memset->addDereferenceableAttr(llvm::AttributeList::FirstArgIndex,
                                   derefBytes);
    memset->addDereferenceableOrNullAttr(llvm::AttributeList::FirstArgIndex,
                                         derefBytes);
#endif
  }
}

/// Perform the corresponding deallocation of tofree, given it was allocated by
/// allocationfn
// For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
llvm::CallInst *freeKnownAllocation(llvm::IRBuilder<> &builder,
                                    llvm::Value *tofree,
                                    const llvm::StringRef allocationfn,
                                    const llvm::DebugLoc &debuglocation,
                                    const llvm::TargetLibraryInfo &TLI,
                                    llvm::CallInst *orig,
                                    GradientUtils *gutils);

static inline bool isAllocationCall(const llvm::Value *TmpOrig,
                                    llvm::TargetLibraryInfo &TLI) {
  if (auto *CI = llvm::dyn_cast<llvm::CallInst>(TmpOrig)) {
    return isAllocationFunction(getFuncNameFromCall(CI), TLI);
  }
  if (auto *CI = llvm::dyn_cast<llvm::InvokeInst>(TmpOrig)) {
    return isAllocationFunction(getFuncNameFromCall(CI), TLI);
  }
  return false;
}

static inline bool isDeallocationCall(const llvm::Value *TmpOrig,
                                      llvm::TargetLibraryInfo &TLI) {
  if (auto *CI = llvm::dyn_cast<llvm::CallInst>(TmpOrig)) {
    return isDeallocationFunction(getFuncNameFromCall(CI), TLI);
  }
  if (auto *CI = llvm::dyn_cast<llvm::InvokeInst>(TmpOrig)) {
    return isDeallocationFunction(getFuncNameFromCall(CI), TLI);
  }
  return false;
}

#endif
