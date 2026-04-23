#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Impulse/ImpulseEnums.cpp.inc"
#include "Dialect/Impulse/ImpulseOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Impulse/ImpulseOps.cpp.inc"

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
