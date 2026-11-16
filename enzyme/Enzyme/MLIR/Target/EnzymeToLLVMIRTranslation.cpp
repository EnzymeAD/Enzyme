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
//   - enzyme.ret_type on a function: the `"enzyme_type"` of its result.
//   - enzyme.local_types on a function: `!enzyme_type` on the alloca of each
//     local variable, by name, if its size matches.
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

/// Mark each alloca of `function` that is a local variable listed in `locals`
/// (name -> [type tree, size in bytes or -1 if not constant]), if its size is
/// that of the variable.
static void annotateLocals(Operation *function, DictionaryAttr locals,
                           LLVM::ModuleTranslation &moduleTranslation) {
  function->walk([&](LLVM::AllocaOp alloca) {
    auto name = alloca->getAttrOfType<StringAttr>("bindc_name");
    auto entry = name ? locals.getAs<ArrayAttr>(name) : ArrayAttr();
    if (!entry || entry.size() != 2)
      return;
    auto tree = dyn_cast<TypeTreeAttr>(entry[0]);
    auto bytes = dyn_cast<IntegerAttr>(entry[1]);
    auto *inst = dyn_cast_or_null<llvm::AllocaInst>(
        moduleTranslation.lookupValue(alloca.getResult()));
    if (!tree || !bytes || !inst)
      return;
    std::optional<llvm::TypeSize> size =
        inst->getAllocationSize(inst->getDataLayout());
    int64_t expected = bytes.getInt();
    if (expected >= 0 ? (!size || size->isScalable() ||
                         (int64_t)size->getFixedValue() != expected)
                      : size.has_value())
      return;
    inst->setMetadata("enzyme_type", typeTreeToMD(tree, moduleTranslation));
  });
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
    if (name == EnzymeDialect::getRetTypeTreeAttrName()) {
      auto tree = dyn_cast<TypeTreeAttr>(attribute.getValue());
      auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!tree || !fn)
        return op->emitError() << name << " must be a type tree on a function";
      if (llvm::Function *f = moduleTranslation.lookupFunction(fn.getName()))
        if (!f->getReturnType()->isVoidTy())
          f->addRetAttr(typeTreeToAttr(tree, moduleTranslation));
      return success();
    }
    if (name == EnzymeDialect::getLocalTypeTreesAttrName()) {
      // Amended after the body of the function is translated, so its values
      // are mapped.
      auto locals = dyn_cast<DictionaryAttr>(attribute.getValue());
      if (!locals)
        return op->emitError() << name << " must be a dictionary";
      annotateLocals(op, locals, moduleTranslation);
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
