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
// is exact for dialects whose pointers name their pointee, such as ClangIR,
// and Unknown for the rest.
//
//===----------------------------------------------------------------------===//

#include "EnzymeLogic.h"

#include "llvm/Support/raw_ostream.h"

#ifdef ENZYME_CLANG_HAS_CIR
#include "clang/CIR/Dialect/IR/CIRDialect.h"
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

#ifdef ENZYME_CLANG_HAS_CIR
static MConcreteType cirConcreteType(Type t) {
  if (isa<cir::IntType, cir::BoolType>(t))
    return MBaseType::Integer;
  if (isa<cir::PointerType>(t))
    return MBaseType::Pointer;
  if (isa<cir::FPTypeInterface>(t))
    return MConcreteType::getFloat(t);
  if (auto v = dyn_cast<cir::VectorType>(t))
    return cirConcreteType(v.getElementType());
  if (auto c = dyn_cast<cir::ComplexType>(t))
    return cirConcreteType(c.getElementType());
  if (auto a = dyn_cast<cir::ArrayType>(t))
    return cirConcreteType(a.getElementType());
  if (auto r = dyn_cast<cir::RecordType>(t)) {
    MConcreteType acc;
    for (Type m : r.getMembers()) {
      MConcreteType ct = cirConcreteType(m);
      if (!ct.isKnown() || (acc.isKnown() && acc != ct))
        return MBaseType::Unknown;
      acc = ct;
    }
    return acc;
  }
  return MBaseType::Unknown;
}

// Handle opaque pointer type.
static bool cirIsOpaquePointee(Type pointee) {
  if (isa<cir::VoidType>(pointee))
    return true;
  // As per C spec, a pointer points to char can be point to anytype. Treat it
  // as a opaque ptr.
  if (auto it = dyn_cast<cir::IntType>(pointee))
    return it.getWidth() == 8;
  return false;
}

static void cirFillTree(MTypeTree &tree, Type t, std::vector<int> prefix,
                        unsigned depth) {
  MConcreteType ct = cirConcreteType(t);
  if (!ct.isKnown())
    return;
  tree.insert(prefix, ct);
  auto ptr = dyn_cast<cir::PointerType>(t);
  if (!ptr || depth == 0 || cirIsOpaquePointee(ptr.getPointee()))
    return;
  prefix.push_back(-1);
  cirFillTree(tree, ptr.getPointee(), prefix, depth - 1);
}
#endif

MTypeTree MTypeResults::query(Value val) const {
  // TODO: cache the tree per type in a DenseMap<Type, MTypeTree> once
  // building it is more than a few isa checks.
  MTypeTree tree;
#ifdef ENZYME_CLANG_HAS_CIR
  cirFillTree(tree, val.getType(), {-1}, /*depth=*/3);
#endif
  return tree;
}

MConcreteType MTypeResults::intType(size_t num, Value val, bool errIfNotFound,
                                    bool pointerIntSame) const {
  if (isa<IntegerType, IndexType>(val.getType()))
    return MBaseType::Integer;
#ifdef ENZYME_CLANG_HAS_CIR
  {
    MConcreteType ct = cirConcreteType(val.getType());
    if (ct.isKnown())
      return ct;
  }
#endif
  if (errIfNotFound)
    llvm_unreachable("something happened");
  return MBaseType::Unknown;
}

// A pointer produced by a cast may be punning, so it answers Unknown; so do
// byte and void pointers.
MConcreteType MTypeResults::firstPointer(size_t num, Value val,
                                         bool errIfNotFound,
                                         bool pointerIntSame) const {
#ifdef ENZYME_CLANG_HAS_CIR
  if (auto ptr = dyn_cast<cir::PointerType>(val.getType())) {
    if (!val.getDefiningOp<cir::CastOp>() &&
        !cirIsOpaquePointee(ptr.getPointee())) {
      MConcreteType ct = cirConcreteType(ptr.getPointee());
      if (ct.isKnown())
        return ct;
    }
  }
#endif
  if (errIfNotFound)
    llvm_unreachable("something happened");
  return MBaseType::Unknown;
}
