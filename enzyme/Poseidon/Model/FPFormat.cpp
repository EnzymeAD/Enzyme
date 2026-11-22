//===- FPFormat.cpp - the formats a precision change uses -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPFormat.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>

using namespace llvm;

namespace poseidon {

FPKind fpKindFromType(Type *T) {
  if (T->isHalfTy())
    return FPKind::F16;
  if (T->isBFloatTy())
    return FPKind::BF16;
  if (T->isFloatTy())
    return FPKind::F32;
  if (T->isDoubleTy())
    return FPKind::F64;
  return FPKind::Invalid;
}

Type *getLLVMFPType(PrecisionChangeType type, LLVMContext &context) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return Type::getBFloatTy(context);
  case PrecisionChangeType::FP16:
    return Type::getHalfTy(context);
  case PrecisionChangeType::FP32:
    return Type::getFloatTy(context);
  case PrecisionChangeType::FP64:
    return Type::getDoubleTy(context);
  case PrecisionChangeType::FP80:
    return Type::getX86_FP80Ty(context);
  case PrecisionChangeType::FP128:
    return Type::getFP128Ty(context);
  default:
    llvm_unreachable("Unsupported FP precision");
  }
}

PrecisionChangeType getPrecisionChangeType(Type *type) {
  if (type->isBFloatTy())
    return PrecisionChangeType::BF16;
  if (type->isHalfTy())
    return PrecisionChangeType::FP16;
  if (type->isFloatTy())
    return PrecisionChangeType::FP32;
  if (type->isDoubleTy())
    return PrecisionChangeType::FP64;
  if (type->isX86_FP80Ty())
    return PrecisionChangeType::FP80;
  if (type->isFP128Ty())
    return PrecisionChangeType::FP128;
  llvm_unreachable("Unsupported FP precision");
}

StringRef getPrecisionChangeTypeString(PrecisionChangeType type) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return "BF16";
  case PrecisionChangeType::FP16:
    return "FP16";
  case PrecisionChangeType::FP32:
    return "FP32";
  case PrecisionChangeType::FP64:
    return "FP64";
  case PrecisionChangeType::FP80:
    return "FP80";
  case PrecisionChangeType::FP128:
    return "FP128";
  case PrecisionChangeType::Expansion2:
    return "Expansion2";
  case PrecisionChangeType::Expansion3:
    return "Expansion3";
  case PrecisionChangeType::Expansion4:
    return "Expansion4";
  default:
    return "Unknown PT type";
  }
}

} // namespace poseidon
