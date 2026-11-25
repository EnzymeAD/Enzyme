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

#include "Dialect/Ops.h"

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

/// Builds the node of the entries in [begin, end), which share their first
/// `depth` offsets: its type is that of the entry of exactly `depth` offsets,
/// if any, and its subtrees group the longer entries by their next offset.
static TypeTreeAttr getTypeTreeNode(
    MLIRContext *context, MTypeTree::ConcreteTypeMapType::const_iterator begin,
    MTypeTree::ConcreteTypeMapType::const_iterator end, size_t depth) {
  TypeTreeKind kind = TypeTreeKind::Unknown;
  Type floatType;
  // The mapping orders an entry before its extensions.
  if (begin != end && begin->first.size() == depth) {
    const MConcreteType &type = begin->second;
    switch (type.SubTypeEnum) {
    case BaseType::Integer:
      kind = TypeTreeKind::Integer;
      break;
    case BaseType::Float:
      kind = TypeTreeKind::Float;
      floatType = type.isFloat();
      break;
    case BaseType::Pointer:
      kind = TypeTreeKind::Pointer;
      break;
    case BaseType::Anything:
      kind = TypeTreeKind::Anything;
      break;
    case BaseType::Unknown:
      break;
    }
    ++begin;
  }
  SmallVector<int64_t> offsets;
  SmallVector<TypeTreeAttr> subtrees;
  while (begin != end) {
    int offset = begin->first[depth];
    auto groupEnd = std::find_if(begin, end, [&](const auto &entry) {
      return entry.first[depth] != offset;
    });
    if (TypeTreeAttr subtree =
            getTypeTreeNode(context, begin, groupEnd, depth + 1)) {
      offsets.push_back(offset);
      subtrees.push_back(subtree);
    }
    begin = groupEnd;
  }
  if (kind == TypeTreeKind::Unknown && subtrees.empty())
    return {};
  return TypeTreeAttr::get(context, kind, floatType, offsets, subtrees);
}

TypeTreeAttr mlir::enzyme::getTypeTreeAttr(MLIRContext *context,
                                           const MTypeTree &tree) {
  const MTypeTree::ConcreteTypeMapType &mapping = tree.getMapping();
  return getTypeTreeNode(context, mapping.begin(), mapping.end(),
                         /*depth=*/0);
}
