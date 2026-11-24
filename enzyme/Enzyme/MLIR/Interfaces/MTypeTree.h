//===- MTypeTree.h - Enzyme's type trees of MLIR values ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enzyme's TypeTree over MLIR floating-point types, and its conversion to the
// #enzyme.type_tree attribute.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_MTYPETREE_H
#define ENZYME_MLIR_INTERFACES_MTYPETREE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/CommandLine.h"

#include "../../TypeAnalysis/TypeTree.h"

template <> struct ConcreteTypeTraits<mlir::Type> {
  static bool isFloatingPoint(mlir::Type T) { return static_cast<bool>(T); }

  static void print(llvm::raw_ostream &OS, mlir::Type T) { OS << T; }

  static std::string floatName(mlir::Type T) {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS << T;
    return S;
  }

  static bool less(mlir::Type A, mlir::Type B) {
    return A.getAsOpaquePointer() < B.getAsOpaquePointer();
  }
};

template <> struct TypeTreeTraits<mlir::Type> {
  using Origin = mlir::Operation *;

  template <typename T>
  static llvm::cl::opt<T> *registeredOption(llvm::StringRef Name) {
    auto &Opts = llvm::cl::getRegisteredOptions();
    auto It = Opts.find(Name);
    return It == Opts.end() ? nullptr
                            : static_cast<llvm::cl::opt<T> *>(It->second);
  }

  static unsigned maxDepth() {
    static auto *Opt = registeredOption<unsigned>("enzyme-max-type-depth");
    return Opt ? *Opt : 6;
  }

  static int maxOffset() {
    static auto *Opt = registeredOption<int>("enzyme-max-type-offset");
    return Opt ? *Opt : 500;
  }

  template <typename TreeT>
  static void depthLimitExceeded(const TreeT &Tree, Origin Orig,
                                 const llvm::Twine &Detail) {
    std::string Msg = ("not handling more than " + llvm::Twine(maxDepth()) +
                       " pointer lookups deep dt:" + Tree.str() + Detail)
                          .str();
    if (Orig)
      Orig->emitWarning(Msg);
    else
      llvm::errs() << Msg << "\n";
  }
};

namespace mlir {
class MLIRContext;

namespace enzyme {
class TypeTreeAttr;

using MConcreteType = GenericConcreteType<mlir::Type>;

using MTypeTree = GenericTypeTree<mlir::Type>;

/// The #enzyme.type_tree attribute holding `tree`, or null if `tree` is empty.
TypeTreeAttr getTypeTreeAttr(MLIRContext *context, const MTypeTree &tree);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_MTYPETREE_H
