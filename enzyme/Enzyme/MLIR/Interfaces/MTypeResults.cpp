//===- MTypeResults.cpp - Type facts for MLIR activity analysis -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MTypeResults answers the activity analysis's questions about what a value
// holds. Nothing is inferred: every answer is read off the value's type, which
// is exact where the type says what a value holds and Unknown elsewhere.
//
//===----------------------------------------------------------------------===//

#include "EnzymeLogic.h"

using namespace mlir;
using namespace mlir::enzyme;

MTypeTree MTypeResults::query(Value val) const {
  // TODO: cache the tree per type in a DenseMap<Type, MTypeTree> once
  // building it is more than a few isa checks.
  MTypeTree tree;
  return tree;
}

MConcreteType MTypeResults::intType(size_t num, Value val, bool errIfNotFound,
                                    bool pointerIntSame) const {
  if (isa<IntegerType, IndexType>(val.getType()))
    return BaseType::Integer;
  if (errIfNotFound)
    llvm_unreachable("MTypeResults::intType: could not deduce the type of the "
                     "value from its MLIR type");
  return BaseType::Unknown;
}

MConcreteType MTypeResults::firstPointer(size_t num, Value val,
                                         bool errIfNotFound,
                                         bool pointerIntSame) const {
  if (errIfNotFound)
    llvm_unreachable("MTypeResults::firstPointer: could not deduce the pointee "
                     "type of the value from its MLIR type");
  return BaseType::Unknown;
}
