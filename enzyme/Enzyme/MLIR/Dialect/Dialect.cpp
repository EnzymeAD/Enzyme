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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

/// Verify that `attr` is an array with one type tree or `unit` for each of the
/// `expected` arguments or results (`what`) of `op`.
static LogicalResult verifyTypeTreeArray(Operation *op, NamedAttribute attr,
                                         size_t expected, StringRef what) {
  auto trees = dyn_cast<ArrayAttr>(attr.getValue());
  if (!trees ||
      !llvm::all_of(trees, llvm::IsaPred<enzyme::TypeTreeAttr, UnitAttr>))
    return op->emitError() << attr.getName().getValue()
                           << " must be an array of type trees or unit";
  if (trees.size() != expected)
    return op->emitError() << attr.getName().getValue() << " has "
                           << trees.size() << " entries for " << expected << ' '
                           << what;
  return success();
}

LogicalResult EnzymeDialect::verifyOperationAttribute(Operation *op,
                                                      NamedAttribute attr) {
  StringRef name = attr.getName();
  if (name == getTypeTreeAttrName()) {
    if (!isa<TypeTreeAttr>(attr.getValue()))
      return op->emitError() << name << " must be a type tree";
    return success();
  }
  if (name == getArgTypeTreesAttrName()) {
    auto call = dyn_cast<CallOpInterface>(op);
    if (!call)
      return op->emitError() << name << " only applies to calls";
    return verifyTypeTreeArray(op, attr, call.getArgOperands().size(),
                               "arguments");
  }
  if (name == getRetTypeTreesAttrName()) {
    auto function = dyn_cast<FunctionOpInterface>(op);
    if (!function)
      return op->emitError() << name << " only applies to functions";
    return verifyTypeTreeArray(op, attr, function.getNumResults(), "results");
  }
  return success();
}

LogicalResult EnzymeDialect::verifyRegionArgAttribute(Operation *op,
                                                      unsigned /*regionIndex*/,
                                                      unsigned /*argIndex*/,
                                                      NamedAttribute attr) {
  if (attr.getName() == getTypeTreeAttrName() &&
      !isa<TypeTreeAttr>(attr.getValue()))
    return op->emitError() << attr.getName().getValue()
                           << " must be a type tree";
  return success();
}

void enzyme::detail::sortTypeTreeSubtrees(
    SmallVectorImpl<int64_t> &offsets,
    SmallVectorImpl<TypeTreeAttr> &subtrees) {
  assert(offsets.size() == subtrees.size() && "expected an offset per subtree");
  SmallVector<std::pair<int64_t, TypeTreeAttr>> children;
  for (auto [offset, subtree] : llvm::zip(offsets, subtrees))
    children.emplace_back(offset, subtree);
  llvm::stable_sort(children, llvm::less_first());
  for (auto [i, child] : llvm::enumerate(children)) {
    offsets[i] = child.first;
    subtrees[i] = child.second;
  }
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/EnzymeAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type trees
//===----------------------------------------------------------------------===//

LogicalResult enzyme::TypeTreeAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, BaseType kind, Type floatType,
    ArrayRef<int64_t> offsets, ArrayRef<TypeTreeAttr> subtrees) {
  assert(offsets.size() == subtrees.size() && "expected an offset per subtree");
  assert(llvm::is_sorted(offsets) &&
         "expected the builder or parser to sort the subtrees");
  if ((kind == BaseType::Float) != isa_and_nonnull<FloatType>(floatType))
    return emitError() << "a type tree node has a floating-point type "
                          "exactly if it is a float";
  for (auto [offset, subtree] : llvm::zip(offsets, subtrees)) {
    if (offset < -1)
      return emitError() << "type tree offset " << offset << " is below -1";
    if (subtree.getKind() == BaseType::Unknown && subtree.getSubtrees().empty())
      return emitError() << "type tree is empty at offset " << offset;
  }
  const int64_t *duplicate = std::adjacent_find(offsets.begin(), offsets.end());
  if (duplicate != offsets.end())
    return emitError() << "type tree offset " << *duplicate
                       << " appears more than once";
  return success();
}

/// The keyword of a type tree node of type `kind`, which is not a float.
static StringRef getTypeTreeKeyword(BaseType kind) {
  switch (kind) {
  case BaseType::Integer:
    return "integer";
  case BaseType::Pointer:
    return "pointer";
  case BaseType::Anything:
    return "anything";
  case BaseType::Float:
  case BaseType::Unknown:
    break;
  }
  llvm_unreachable("only integer, pointer and anything have a keyword");
}

/// Parse a type tree node `[<type>] [{<offset>: <node>, ...}]`, where <type>
/// is integer, pointer, anything, or a floating-point type.
static enzyme::TypeTreeAttr parseTypeTreeNode(AsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  BaseType kind = BaseType::Unknown;
  Type floatType;
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(
          &keyword, {"integer", "pointer", "anything"}))) {
    kind = llvm::StringSwitch<BaseType>(keyword)
               .Case("integer", BaseType::Integer)
               .Case("pointer", BaseType::Pointer)
               .Case("anything", BaseType::Anything);
  } else {
    OptionalParseResult result = parser.parseOptionalType(floatType);
    if (result.has_value()) {
      if (failed(*result))
        return {};
      kind = BaseType::Float;
    }
  }
  SmallVector<int64_t> offsets;
  SmallVector<enzyme::TypeTreeAttr> subtrees;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::OptionalBraces, [&]() -> ParseResult {
            if (parser.parseInteger(offsets.emplace_back()) ||
                parser.parseColon())
              return failure();
            enzyme::TypeTreeAttr subtree = parseTypeTreeNode(parser);
            if (!subtree)
              return failure();
            subtrees.push_back(subtree);
            return success();
          }))
    return {};
  enzyme::detail::sortTypeTreeSubtrees(offsets, subtrees);
  return enzyme::TypeTreeAttr::getChecked(
      [&]() { return parser.emitError(loc); }, parser.getContext(), kind,
      floatType, offsets, subtrees);
}

/// Print a type tree node in the syntax that parseTypeTreeNode reads.
static void printTypeTreeNode(AsmPrinter &printer, enzyme::TypeTreeAttr node) {
  if (node.getKind() == BaseType::Float)
    printer << node.getFloatType();
  else if (node.getKind() != BaseType::Unknown)
    printer << getTypeTreeKeyword(node.getKind());
  if (node.getSubtrees().empty())
    return;
  if (node.getKind() != BaseType::Unknown)
    printer << ' ';
  printer << '{';
  llvm::interleaveComma(llvm::zip(node.getOffsets(), node.getSubtrees()),
                        printer, [&](auto child) {
                          printer << std::get<0>(child) << ": ";
                          printTypeTreeNode(printer, std::get<1>(child));
                        });
  printer << '}';
}

Attribute enzyme::TypeTreeAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};
  TypeTreeAttr tree = parseTypeTreeNode(parser);
  if (!tree || parser.parseGreater())
    return {};
  return tree;
}

void enzyme::TypeTreeAttr::print(AsmPrinter &printer) const {
  printer << '<';
  printTypeTreeNode(printer, *this);
  printer << '>';
}
