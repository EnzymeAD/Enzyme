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
#include "llvm/ADT/StringSwitch.h"
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

#define GET_ATTRDEF_CLASSES
#include "Dialect/EnzymeAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type trees
//===----------------------------------------------------------------------===//

LogicalResult
enzyme::TypeTreeEntryAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> path, BaseType kind,
                                  Type floatType) {
  if (kind == BaseType::Unknown)
    return emitError() << "a type tree entry must have a known type";
  if ((kind == BaseType::Float) != isa_and_nonnull<FloatType>(floatType))
    return emitError() << "a type tree entry has a floating-point type "
                          "exactly if it is a float";
  for (int64_t off : path)
    if (off < -1)
      return emitError() << "type tree offset " << off << " is below -1";
  return success();
}

LogicalResult
enzyme::TypeTreeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<TypeTreeEntryAttr> entries) {
  if (entries.empty())
    return success();
  for (auto [prev, next] : llvm::zip(entries, entries.drop_front()))
    if (!std::lexicographical_compare(
            prev.getPath().begin(), prev.getPath().end(),
            next.getPath().begin(), next.getPath().end()))
      return emitError() << "type tree paths must be unique and ordered";
  return success();
}

/// `[<offset>, ...]: <type>`, where <type> is integer, pointer, anything, or
/// a floating-point type.
static ParseResult parseTypeTreeEntry(AsmParser &parser,
                                      enzyme::TypeTreeEntryAttr &entry) {
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<int64_t> path;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square,
          [&]() { return parser.parseInteger(path.emplace_back()); }) ||
      parser.parseColon())
    return failure();
  BaseType kind = BaseType::Float;
  Type floatType;
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(
          &keyword, {"integer", "pointer", "anything"})))
    kind = llvm::StringSwitch<BaseType>(keyword)
               .Case("integer", BaseType::Integer)
               .Case("pointer", BaseType::Pointer)
               .Default(BaseType::Anything);
  else if (parser.parseType(floatType))
    return failure();
  entry = enzyme::TypeTreeEntryAttr::getChecked(
      [&]() { return parser.emitError(loc); }, parser.getContext(), path, kind,
      floatType);
  return success(static_cast<bool>(entry));
}

static void printTypeTreeEntry(AsmPrinter &printer,
                               enzyme::TypeTreeEntryAttr entry) {
  printer << "[";
  llvm::interleaveComma(entry.getPath(), printer);
  printer << "]: ";
  switch (entry.getKind()) {
  case BaseType::Integer:
    printer << "integer";
    break;
  case BaseType::Pointer:
    printer << "pointer";
    break;
  case BaseType::Anything:
    printer << "anything";
    break;
  case BaseType::Float:
    printer << entry.getFloatType();
    break;
  case BaseType::Unknown:
    llvm_unreachable("verified to be known");
  }
}

Attribute enzyme::TypeTreeEntryAttr::parse(AsmParser &parser, Type) {
  TypeTreeEntryAttr entry;
  if (parser.parseLess() || parseTypeTreeEntry(parser, entry) ||
      parser.parseGreater())
    return {};
  return entry;
}

void enzyme::TypeTreeEntryAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printTypeTreeEntry(printer, *this);
  printer << ">";
}

Attribute enzyme::TypeTreeAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  SmallVector<TypeTreeEntryAttr> entries;
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&]() {
        return parseTypeTreeEntry(parser, entries.emplace_back());
      }))
    return {};
  return TypeTreeAttr::getChecked([&]() { return parser.emitError(loc); },
                                  parser.getContext(), entries);
}

void enzyme::TypeTreeAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getEntries(), printer, [&](TypeTreeEntryAttr entry) {
    printTypeTreeEntry(printer, entry);
  });
  printer << ">";
}
