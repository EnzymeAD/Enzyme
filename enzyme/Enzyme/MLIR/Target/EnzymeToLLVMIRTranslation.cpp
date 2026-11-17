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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

#include "../../TypeAnalysis/ConcreteType.h"

#include <map>

using namespace mlir;
using namespace mlir::enzyme;

/// The name LLVM Enzyme gives the concrete type of `entry`
/// (ConcreteType::str, e.g. "Float@double").
static std::string
concreteTypeName(TypeTreeEntryAttr entry,
                 LLVM::ModuleTranslation &moduleTranslation) {
  if (entry.getKind() != BaseType::Float)
    return ConcreteType(entry.getKind()).str();
  return ConcreteType(moduleTranslation.convertType(entry.getFloatType()))
      .str();
}

/// The type tree as TypeTree::str prints it, which TypeTree::parse reads from
/// "enzyme_type" attributes: `{[-1]:Pointer, [-1,0]:Float@double}`.
static std::string
typeTreeToString(TypeTreeAttr tree,
                 LLVM::ModuleTranslation &moduleTranslation) {
  std::string out = "{";
  llvm::raw_string_ostream os(out);
  llvm::interleave(
      tree.getEntries(), os,
      [&](TypeTreeEntryAttr entry) {
        os << '[';
        llvm::interleave(entry.getPath(), os, ",");
        os << "]:" << concreteTypeName(entry, moduleTranslation);
      },
      ", ");
  os << '}';
  return out;
}

/// The entries of `entries` (all with paths of at least `depth` offsets and
/// the same first `depth` ones) in the encoding of TypeTree::toMD:
/// !{!"<type at depth>", i32 <offset>, <subtree>, ...}, by ascending offset.
static llvm::MDNode *typeTreeToMD(ArrayRef<TypeTreeEntryAttr> entries,
                                  size_t depth,
                                  LLVM::ModuleTranslation &moduleTranslation) {
  llvm::LLVMContext &ctx = moduleTranslation.getLLVMContext();
  std::string base = ConcreteType(BaseType::Unknown).str();
  std::map<int64_t, SmallVector<TypeTreeEntryAttr>> children;
  for (TypeTreeEntryAttr entry : entries) {
    if (entry.getPath().size() == depth)
      base = concreteTypeName(entry, moduleTranslation);
    else
      children[entry.getPath()[depth]].push_back(entry);
  }
  SmallVector<llvm::Metadata *> ops{llvm::MDString::get(ctx, base)};
  for (const auto &[offset, sub] : children) {
    ops.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::IntegerType::get(ctx, 32), offset, /*IsSigned=*/true)));
    ops.push_back(typeTreeToMD(sub, depth + 1, moduleTranslation));
  }
  return llvm::MDNode::get(ctx, ops);
}

static llvm::MDNode *typeTreeToMD(TypeTreeAttr tree,
                                  LLVM::ModuleTranslation &moduleTranslation) {
  return typeTreeToMD(tree.getEntries(), 0, moduleTranslation);
}

static llvm::Attribute
typeTreeToAttr(TypeTreeAttr tree, LLVM::ModuleTranslation &moduleTranslation) {
  return llvm::Attribute::get(moduleTranslation.getLLVMContext(), "enzyme_type",
                              typeTreeToString(tree, moduleTranslation));
}

namespace {
struct EnzymeDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    StringRef name = attribute.getName();
    if (name == EnzymeDialect::getArgTypeTreesAttrName()) {
      auto trees = dyn_cast<ArrayAttr>(attribute.getValue());
      if (!trees)
        return op->emitError() << name << " must be an array";
      for (llvm::Instruction *inst : instructions)
        if (auto *call = dyn_cast<llvm::CallBase>(inst))
          for (auto [i, attr] : llvm::enumerate(trees))
            if (auto tree = dyn_cast<TypeTreeAttr>(attr);
                tree && i < call->arg_size())
              call->addParamAttr(i, typeTreeToAttr(tree, moduleTranslation));
      return success();
    }
    if (name == EnzymeDialect::getRetTypeTreesAttrName()) {
      auto trees = dyn_cast<ArrayAttr>(attribute.getValue());
      auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!trees || !fn)
        return op->emitError() << name << " must be an array on a function";
      // An LLVM function has at most one result.
      if (trees.size() > 1)
        return op->emitError() << name << " has more than one result";
      llvm::Function *f = moduleTranslation.lookupFunction(fn.getName());
      if (auto tree =
              trees.empty() ? TypeTreeAttr() : dyn_cast<TypeTreeAttr>(trees[0]);
          tree && f && !f->getReturnType()->isVoidTy())
        f->addRetAttr(typeTreeToAttr(tree, moduleTranslation));
      return success();
    }
    if (name != EnzymeDialect::getTypeTreeAttrName())
      return success();
    auto tree = dyn_cast<TypeTreeAttr>(attribute.getValue());
    if (!tree)
      return op->emitError() << name << " must be a type tree";
    if (isa<LLVM::GlobalOp>(op)) {
      if (auto *gv = dyn_cast_or_null<llvm::GlobalVariable>(
              moduleTranslation.lookupGlobal(op)))
        gv->setMetadata("enzyme_type", typeTreeToMD(tree, moduleTranslation));
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
    auto tree = dyn_cast<TypeTreeAttr>(attribute.getValue());
    if (!tree)
      return function.emitError()
             << attribute.getName() << " must be a type tree";
    if (llvm::Function *fn =
            moduleTranslation.lookupFunction(function.getName()))
      fn->addParamAttr(argIdx, typeTreeToAttr(tree, moduleTranslation));
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
