//===- FPFormat.cpp - the formats a precision change uses -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPFormat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace poseidon {

static constexpr FPFormatInfo kFormats[] = {
    {PrecisionChangeType::BF16, FPKind::BF16, "bf16", "bf16", "__bf16", "BF16",
     Type::getBFloatTy},
    {PrecisionChangeType::FP16, FPKind::F16, "f16", "half", "_Float16", "FP16",
     Type::getHalfTy},
    {PrecisionChangeType::FP32, FPKind::F32, "f32", "float", "float", "FP32",
     Type::getFloatTy},
    {PrecisionChangeType::FP64, FPKind::F64, "f64", "double", "double", "FP64",
     Type::getDoubleTy},
    {PrecisionChangeType::FP80, FPKind::Invalid, "f80", "fp80", "long double",
     "FP80", Type::getX86_FP80Ty},
    {PrecisionChangeType::FP128, FPKind::Invalid, "f128", "fp128", "__float128",
     "FP128", Type::getFP128Ty},
    {PrecisionChangeType::Expansion2, FPKind::Invalid, "", "", "", "Expansion2",
     nullptr},
    {PrecisionChangeType::Expansion3, FPKind::Invalid, "", "", "", "Expansion3",
     nullptr},
    {PrecisionChangeType::Expansion4, FPKind::Invalid, "", "", "", "Expansion4",
     nullptr},
};

const FPFormatInfo &formatOf(PrecisionChangeType P) {
  const FPFormatInfo *F = find_if(
      kFormats, [&](const FPFormatInfo &F) { return F.Precision == P; });
  assert(F != std::end(kFormats) && "every precision has a row");
  return *F;
}

const FPFormatInfo *formatOf(Type *T) {
  const FPFormatInfo *F = find_if(kFormats, [&](const FPFormatInfo &F) {
    return F.LLVMType && F.LLVMType(T->getContext()) == T;
  });
  return F == std::end(kFormats) ? nullptr : F;
}

const FPFormatInfo *formatOfDtype(StringRef Dtype) {
  const FPFormatInfo *F = find_if(kFormats, [&](const FPFormatInfo &F) {
    return !F.Dtype.empty() && F.Dtype == Dtype;
  });
  return F == std::end(kFormats) ? nullptr : F;
}

FPKind fpKindFromType(Type *T) {
  const FPFormatInfo *F = formatOf(T);
  return F ? F->Kind : FPKind::Invalid;
}

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

Type *getLLVMFPType(PrecisionChangeType type, LLVMContext &context) {
  const FPFormatInfo &F = formatOf(type);
  if (!F.LLVMType)
    llvm_unreachable("Unsupported FP precision");
  return F.LLVMType(context);
}

PrecisionChangeType getPrecisionChangeType(Type *type) {
  if (const FPFormatInfo *F = formatOf(type))
    return F->Precision;
  llvm_unreachable("Unsupported FP precision");
}

StringRef getPrecisionChangeTypeString(PrecisionChangeType type) {
  return formatOf(type).Name;
}

} // namespace poseidon
