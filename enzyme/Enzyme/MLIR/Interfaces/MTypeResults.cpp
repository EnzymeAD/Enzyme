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

#include "llvm/Support/raw_ostream.h"

#ifdef ENZYME_CLANG_HAS_CIR
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#endif

using namespace mlir;
using namespace mlir::enzyme;

std::string MConcreteType::str() const {
  switch (kind) {
  case MBaseType::Integer:
    return "Integer";
  case MBaseType::Pointer:
    return "Pointer";
  case MBaseType::Anything:
    return "Anything";
  case MBaseType::Unknown:
    return "Unknown";
  case MBaseType::Float: {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "Float@" << floatType;
    return os.str();
  }
  }
  return "Unknown";
}

void MTypeTree::insert(const std::vector<int> &seq, MConcreteType ct) {
  mapping.insert_or_assign(seq, ct);
}

MConcreteType MTypeTree::operator[](const std::vector<int> &seq) const {
  auto found = mapping.find(seq);
  if (found != mapping.end())
    return found->second;
  if (seq.empty())
    return MBaseType::Unknown;

  auto extend = [](const std::vector<int> &prefix, int idx) {
    llvm::SmallVector<std::vector<int>, 2> out;
    out.push_back(prefix);
    out.back().push_back(-1);
    if (idx != -1) {
      out.push_back(prefix);
      out.back().push_back(idx);
    }
    return out;
  };

  std::vector<std::vector<int>> prefixes = {{}};
  for (size_t i = 0; i + 1 < seq.size(); ++i) {
    std::vector<std::vector<int>> next;
    for (const auto &prefix : prefixes)
      for (auto &cand : extend(prefix, seq[i]))
        if (mapping.count(cand))
          next.push_back(std::move(cand));
    prefixes = std::move(next);
  }

  for (const auto &prefix : prefixes)
    for (auto &cand : extend(prefix, seq.back()))
      if (auto f = mapping.find(cand); f != mapping.end())
        return f->second;
  return MBaseType::Unknown;
}

MTypeTree MTypeResults::query(Value val) const {
  // TODO: cache the tree per type in a DenseMap<Type, MTypeTree> once
  // building it is more than a few isa checks.
  MTypeTree tree;
  return tree;
}

MConcreteType MTypeResults::intType(size_t num, Value val, bool errIfNotFound,
                                    bool pointerIntSame) const {
  if (isa<IntegerType, IndexType>(val.getType()))
    return MBaseType::Integer;
#ifdef ENZYME_CLANG_HAS_CIR
  if (isa<cir::IntType, cir::BoolType>(val.getType()))
    return MBaseType::Integer;
#endif
  if (errIfNotFound)
    llvm_unreachable("something happened");
  return MBaseType::Unknown;
}

MConcreteType MTypeResults::firstPointer(size_t num, Value val,
                                         bool errIfNotFound,
                                         bool pointerIntSame) const {
  if (errIfNotFound)
    llvm_unreachable("something happened");
  return MBaseType::Unknown;
}
