//===- EnzymeDialect.cpp - Enzyme dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect.h"
#include "Ops.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/EnzymeEnums.cpp.inc"
#include "Dialect/EnzymeOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/EnzymeOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "Dialect/EnzymeOpsTypes.cpp.inc"
// #include "Dialect/EnzymeTypes.cpp.inc"

using namespace mlir;
using namespace mlir::enzyme;

//===----------------------------------------------------------------------===//
// Enzyme dialect.
//===----------------------------------------------------------------------===//

namespace {
struct EnzymeDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation * /*call*/, Operation * /*callable*/,
                       bool /*wouldBeCloned*/) const final {
    return true;
  }

  bool isLegalToInline(Region * /*dest*/, Region * /*src*/,
                       bool /*wouldBeCloned*/,
                       IRMapping & /*valueMapping*/) const final {
    return true;
  }

  bool isLegalToInline(Operation * /*op*/, Region * /*dest*/,
                       bool /*wouldBeCloned*/,
                       IRMapping & /*valueMapping*/) const final {
    return true;
  }
};
} // namespace

void EnzymeDialect::initialize() {
  addInterfaces<EnzymeDialectInlinerInterface>();
  addOperations<
#define GET_OP_LIST
#include "Dialect/EnzymeOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/EnzymeAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/EnzymeOpsTypes.cpp.inc"
      >();
}

void EnzymeDialect::registerForwardCallHandler(StringRef key,
                                               ForwardCallHandler handler) {
  forwardCallHandlers[key] = std::move(handler);
}

const EnzymeDialect::ForwardCallHandler *
EnzymeDialect::lookupForwardCallHandler(StringRef key) const {
  auto found = forwardCallHandlers.find(key);
  return found == forwardCallHandlers.end() ? nullptr : &found->second;
}

LogicalResult EnzymeDialect::verifyOperationAttribute(Operation *op,
                                                      NamedAttribute attr) {
  if (attr.getName() != getForwardCallHandlersAttrName())
    return success();
  if (!op->hasTrait<OpTrait::SymbolTable>())
    return op->emitOpError() << "'" << attr.getName().getValue()
                             << "' requires a symbol table operation";
  auto entries = dyn_cast<ArrayAttr>(attr.getValue());
  if (!entries)
    return op->emitOpError() << "'" << attr.getName().getValue()
                             << "' must be an array attribute";
  for (Attribute entry : entries) {
    auto pair = dyn_cast<ArrayAttr>(entry);
    if (!pair || pair.size() != 2 || !isa<FlatSymbolRefAttr>(pair[0]) ||
        !isa<StringAttr>(pair[1]))
      return op->emitOpError()
             << "'" << attr.getName().getValue()
             << "' entry must be a [@function, \"key\"] pair, got " << entry;
  }
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/EnzymeAttributes.cpp.inc"
