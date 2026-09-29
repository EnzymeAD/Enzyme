#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Impulse/ImpulseEnums.cpp.inc"
#include "Dialect/Impulse/ImpulseOpsDialect.cpp.inc"

/// Symbol name only, so the operand list's `(` is not parsed as an attribute.
static mlir::ParseResult parseOptionalCallee(mlir::OpAsmParser &parser,
                                             mlir::FlatSymbolRefAttr &fn) {
  mlir::StringAttr name;
  if (succeeded(parser.parseOptionalSymbolName(name)))
    fn = mlir::FlatSymbolRefAttr::get(name);
  return mlir::success();
}

static void printOptionalCallee(mlir::OpAsmPrinter &printer, mlir::Operation *,
                                mlir::FlatSymbolRefAttr fn) {
  if (!fn)
    return;
  printer << ' ';
  printer.printAttributeWithoutType(fn);
}

#define GET_OP_CLASSES
#include "Dialect/Impulse/ImpulseOps.cpp.inc"

static mlir::ParseResult parseDistTail(mlir::AsmParser &parser,
                                       mlir::ArrayAttr &transforms,
                                       int64_t &eventDim) {
  transforms = {};
  eventDim = 0;
  if (failed(parser.parseOptionalComma()))
    return mlir::success();
  if (failed(parser.parseOptionalKeyword("event_dim"))) {
    if (parser.parseAttribute(transforms))
      return mlir::failure();
    if (failed(parser.parseOptionalComma()))
      return mlir::success();
    if (parser.parseKeyword("event_dim"))
      return mlir::failure();
  }
  return mlir::failure(parser.parseEqual() || parser.parseInteger(eventDim));
}

static void printDistTail(mlir::AsmPrinter &printer, mlir::ArrayAttr transforms,
                          int64_t eventDim) {
  if (transforms) {
    printer << ", ";
    printer.printStrippedAttrOrType(transforms);
  }
  if (eventDim != 0)
    printer << ", event_dim = " << eventDim;
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/Impulse/ImpulseAttributes.cpp.inc"

void mlir::impulse::ImpulseDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Impulse/ImpulseOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Impulse/ImpulseAttributes.cpp.inc"
      >();
}
