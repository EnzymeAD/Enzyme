//===- EnzymeToLLVMIRTranslation.cpp - Enzyme dialect to LLVM IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translates the enzyme dialect's #enzyme.type_tree attributes to what LLVM
// Enzyme's type analysis reads:
//   - enzyme.type on a global: `!enzyme_type` metadata, the layout of its
//     memory; on another op: `!enzyme_type` on each instruction it became;
//     on a function argument: its `"enzyme_type"` attribute.
//   - enzyme.arg_types on a call: the `"enzyme_type"` of each argument.
//   - enzyme.ret_types on a function: the `"enzyme_type"` of its result.
//
//===----------------------------------------------------------------------===//

#include "Target/EnzymeToLLVMIRTranslation.h"

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "TypeAnalysis/ConcreteType.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

using namespace mlir;
using namespace mlir::enzyme;

/// Print the concrete type of `node` as ConcreteType::str does, e.g.
/// "Float@double".
static void printConcreteType(raw_ostream &os, TypeTreeAttr node,
                              LLVM::ModuleTranslation &moduleTranslation) {
  if (node.getKind() == BaseType::Float)
    os << ConcreteType(moduleTranslation.convertType(node.getFloatType()))
              .str();
  else
    os << ConcreteType(node.getKind()).str();
}

/// Convert `tree` to `!enzyme_type` metadata, in the encoding of
/// TypeTree::toMD: !{!"<type>", i32 <offset>, <subtree>, ...}.
static llvm::MDNode *typeTreeToMD(TypeTreeAttr tree,
                                  LLVM::ModuleTranslation &moduleTranslation) {
  llvm::LLVMContext &ctx = moduleTranslation.getLLVMContext();
  SmallString<16> type;
  llvm::raw_svector_ostream os(type);
  printConcreteType(os, tree, moduleTranslation);
  SmallVector<llvm::Metadata *> ops{llvm::MDString::get(ctx, type)};
  for (auto [offset, subtree] :
       llvm::zip(tree.getOffsets(), tree.getSubtrees())) {
    ops.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::IntegerType::get(ctx, 32), offset, /*IsSigned=*/true)));
    ops.push_back(typeTreeToMD(subtree, moduleTranslation));
  }
  return llvm::MDNode::get(ctx, ops);
}

/// Print `[<path>]:<type>` for each node of `tree` with a known type, where
/// `path` holds the offsets leading to `tree`.
static void printTypeTreeEntries(raw_ostream &os, TypeTreeAttr tree,
                                 SmallVectorImpl<int64_t> &path,
                                 llvm::ListSeparator &separator,
                                 LLVM::ModuleTranslation &moduleTranslation) {
  if (tree.getKind() != BaseType::Unknown) {
    os << separator << '[';
    llvm::interleave(path, os, ",");
    os << "]:";
    printConcreteType(os, tree, moduleTranslation);
  }
  for (auto [offset, subtree] :
       llvm::zip(tree.getOffsets(), tree.getSubtrees())) {
    path.push_back(offset);
    printTypeTreeEntries(os, subtree, path, separator, moduleTranslation);
    path.pop_back();
  }
}

/// Convert `tree` to an "enzyme_type" attribute, in the syntax of
/// TypeTree::str: `{[-1]:Pointer, [-1,0]:Float@double}`.
static llvm::Attribute
typeTreeToAttr(TypeTreeAttr tree, LLVM::ModuleTranslation &moduleTranslation) {
  SmallString<64> str;
  llvm::raw_svector_ostream os(str);
  SmallVector<int64_t> path;
  llvm::ListSeparator separator;
  os << '{';
  printTypeTreeEntries(os, tree, path, separator, moduleTranslation);
  os << '}';
  return llvm::Attribute::get(moduleTranslation.getLLVMContext(), "enzyme_type",
                              str);
}

namespace {
/// Translates the type tree attributes, which EnzymeDialect verifies.
struct EnzymeDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    StringRef name = attribute.getName();
    if (name == EnzymeDialect::getArgTypeTreesAttrName()) {
      auto trees = cast<ArrayAttr>(attribute.getValue());
      for (llvm::Instruction *inst : instructions) {
        auto *call = dyn_cast<llvm::CallBase>(inst);
        if (!call)
          continue;
        assert(trees.size() <= call->arg_size() &&
               "more type trees than arguments");
        for (auto [i, attr] : llvm::enumerate(trees))
          if (auto tree = dyn_cast<TypeTreeAttr>(attr))
            call->addParamAttr(i, typeTreeToAttr(tree, moduleTranslation));
      }
      return success();
    }
    if (name == EnzymeDialect::getRetTypeTreesAttrName()) {
      auto trees = cast<ArrayAttr>(attribute.getValue());
      llvm::Function *fn = moduleTranslation.lookupFunction(
          cast<LLVM::LLVMFuncOp>(op).getName());
      assert(fn && "expected the function to be translated");
      // An LLVM function has at most one result.
      if (auto tree =
              trees.empty() ? TypeTreeAttr() : dyn_cast<TypeTreeAttr>(trees[0]))
        fn->addRetAttr(typeTreeToAttr(tree, moduleTranslation));
      return success();
    }
    if (name != EnzymeDialect::getTypeTreeAttrName())
      return success();
    auto tree = cast<TypeTreeAttr>(attribute.getValue());
    if (isa<LLVM::GlobalOp>(op)) {
      cast<llvm::GlobalVariable>(moduleTranslation.lookupGlobal(op))
          ->setMetadata("enzyme_type", typeTreeToMD(tree, moduleTranslation));
      return success();
    }
    for (llvm::Instruction *inst : instructions)
      inst->setMetadata("enzyme_type", typeTreeToMD(tree, moduleTranslation));
    return success();
  }

  LogicalResult
  convertParameterAttr(LLVM::LLVMFuncOp function, int argIdx,
                       NamedAttribute attribute,
                       LLVM::ModuleTranslation &moduleTranslation) const final {
    if (attribute.getName() != EnzymeDialect::getTypeTreeAttrName())
      return success();
    llvm::Function *fn = moduleTranslation.lookupFunction(function.getName());
    assert(fn && "expected the function to be translated");
    fn->addParamAttr(argIdx,
                     typeTreeToAttr(cast<TypeTreeAttr>(attribute.getValue()),
                                    moduleTranslation));
    return success();
  }
};
} // namespace

void mlir::enzyme::registerEnzymeDialectTranslation(DialectRegistry &registry) {
  registry.insert<EnzymeDialect>();
  registry.addExtension(+[](MLIRContext *, EnzymeDialect *dialect) {
    if (!dialect->getRegisteredInterface<LLVMTranslationDialectInterface>())
      dialect->addInterfaces<EnzymeDialectLLVMIRTranslationInterface>();
  });
}
