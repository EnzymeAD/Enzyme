//===- FPFormat.cpp - the formats a precision change uses -----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPFormat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

namespace poseidon {

// The LLVMType column as one function type (getInt8Ty returns an IntegerType);
// null for an expansion.
template <auto Get> static Type *storage(LLVMContext &C) {
  if constexpr (Get == nullptr)
    return nullptr;
  else
    return Get(C);
}

static constexpr FPFormatInfo kFormats[] = {
#define FP_FORMAT(Kind, Name, Dtype, CostType, CType, Words, LLVMType,         \
                  Semantics, Components, Matrix)                               \
  {FPKind::Kind,      Name,       Dtype, CostType, CType, Words,               \
   storage<LLVMType>, Components, Matrix},
#include "Model/FPFormat.def"
};

const FPFormatInfo &formatOf(FPKind K) {
  assert(K != FPKind::Invalid && "Invalid has no row");
  return kFormats[unsigned(K) - 1];
}

// A format no graph value has (TF32, the integer formats, an expansion) has no
// C type, so a float is FP32 and an i8 is no format.
static bool isGraphFormat(const FPFormatInfo &F) { return !F.CType.empty(); }

const FPFormatInfo *formatOf(Type *T) {
  const FPFormatInfo *F = find_if(kFormats, [&](const FPFormatInfo &F) {
    return isGraphFormat(F) && F.LLVMType(T->getContext()) == T;
  });
  return F == std::end(kFormats) ? nullptr : F;
}

const FPFormatInfo *formatOfDtype(StringRef Dtype) {
  const FPFormatInfo *F = find_if(kFormats, [&](const FPFormatInfo &F) {
    return isGraphFormat(F) && F.Dtype == Dtype;
  });
  return F == std::end(kFormats) ? nullptr : F;
}

FPKind fpKindFromType(Type *T) {
  const FPFormatInfo *F = formatOf(T);
  return F && F->Matrix ? F->Kind : FPKind::Invalid;
}

Type *llvmTypeForFPKind(LLVMContext &ctx, FPKind k) {
  const FPFormatInfo &F = formatOf(k);
  assert(!F.Components && "an expansion has no LLVM type");
  return F.LLVMType(ctx);
}

} // namespace poseidon
