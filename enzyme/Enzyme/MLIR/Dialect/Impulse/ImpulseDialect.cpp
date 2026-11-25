#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::impulse;

#include "Dialect/Impulse/ImpulseEnums.cpp.inc"
#include "Dialect/Impulse/ImpulseOpsDialect.cpp.inc"

/// Symbol name only, so the operand list's `(` is not parsed as an attribute.
static ParseResult parseOptionalCallee(OpAsmParser &parser,
                                       FlatSymbolRefAttr &fn) {
  StringAttr name;
  if (succeeded(parser.parseOptionalSymbolName(name)))
    fn = FlatSymbolRefAttr::get(name);
  return success();
}

/// Prints `fn`, if set, as parseOptionalCallee reads it.
static void printOptionalCallee(OpAsmPrinter &printer, Operation *,
                                FlatSymbolRefAttr fn) {
  if (!fn)
    return;
  printer << ' ';
  printer.printAttributeWithoutType(fn);
}

#define GET_OP_CLASSES
#include "Dialect/Impulse/ImpulseOps.cpp.inc"

/// Parses `(, lower = x)? (, upper = y)? (, event_size = k)?`, the parts of a
/// SupportAttr after its kind.
static ParseResult parseSupportParams(AsmParser &parser, FloatAttr &lower,
                                      FloatAttr &upper,
                                      IntegerAttr &eventSize) {
  bool more = succeeded(parser.parseOptionalComma());
  if (more && succeeded(parser.parseOptionalKeyword("lower"))) {
    if (parser.parseEqual() || parser.parseCustomAttributeWithFallback(lower))
      return failure();
    more = succeeded(parser.parseOptionalComma());
  }
  if (more && succeeded(parser.parseOptionalKeyword("upper"))) {
    if (parser.parseEqual() || parser.parseCustomAttributeWithFallback(upper))
      return failure();
    more = succeeded(parser.parseOptionalComma());
  }
  if (!more)
    return success();
  return failure(parser.parseKeyword("event_size") || parser.parseEqual() ||
                 parser.parseCustomAttributeWithFallback(eventSize));
}

/// Prints what parseSupportParams reads.
static void printSupportParams(AsmPrinter &printer, FloatAttr lower,
                               FloatAttr upper, IntegerAttr eventSize) {
  if (lower) {
    printer << ", lower = ";
    printer.printStrippedAttrOrType(lower);
  }
  if (upper) {
    printer << ", upper = ";
    printer.printStrippedAttrOrType(upper);
  }
  if (eventSize) {
    printer << ", event_size = ";
    printer.printStrippedAttrOrType(eventSize);
  }
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

//===----------------------------------------------------------------------===//
// DistAttr
//===----------------------------------------------------------------------===//

/// The support of the image of a value with support `in` under `xform`.
static SupportKind applyXformToSupport(XformAttr xform, SupportKind in) {
  switch (xform.getKind()) {
  case XformKind::IDENTITY:
    return in;
  case XformKind::AFFINE:
    // loc + scale * [0, 1] is an interval with operand bounds.
    return in == SupportKind::UNIT_INTERVAL ? SupportKind::INTERVAL : in;
  case XformKind::EXP:
  case XformKind::SOFTPLUS:
    return SupportKind::POSITIVE;
  case XformKind::SIGMOID:
    return SupportKind::UNIT_INTERVAL;
  }
  llvm_unreachable("unknown XformKind");
}

/// Support of the base distribution before its transform chain.
static SupportKind baseSupport(DistBaseKind base) {
  switch (base) {
  case DistBaseKind::STD_UNIFORM:
    return SupportKind::UNIT_INTERVAL;
  case DistBaseKind::EXPONENTIAL:
  case DistBaseKind::GAMMA:
  case DistBaseKind::HALF_NORMAL:
  case DistBaseKind::HALF_CAUCHY:
  case DistBaseKind::HALF_STUDENT_T:
    return SupportKind::POSITIVE;
  case DistBaseKind::DIRICHLET:
    return SupportKind::SIMPLEX;
  case DistBaseKind::STD_NORMAL:
  case DistBaseKind::STUDENT_T:
  case DistBaseKind::BERNOULLI:
  case DistBaseKind::POISSON:
  case DistBaseKind::DISCRETE_UNIFORM:
  case DistBaseKind::MULTIVARIATE_NORMAL:
  case DistBaseKind::GUMBEL:
    return SupportKind::REAL;
  }
  llvm_unreachable("unknown DistBaseKind");
}

SupportAttr DistAttr::getSupport() const {
  SupportKind kind = baseSupport(getBase());
  if (ArrayAttr transforms = getTransforms())
    for (Attribute a : transforms)
      kind = applyXformToSupport(cast<XformAttr>(a), kind);
  return SupportAttr::get(getContext(), kind, /*lower=*/FloatAttr(),
                          /*upper=*/FloatAttr(), /*event_size=*/IntegerAttr());
}

int64_t SupportAttr::getUnconstrainedSize(int64_t constrainedSize) const {
  if (getKind() != SupportKind::SIMPLEX)
    return constrainedSize;
  int64_t eventSize = getEventSize().getInt();
  return constrainedSize / eventSize * (eventSize - 1);
}

unsigned DistAttr::getNumParameters() const {
  unsigned numParams = 0;
  switch (getBase()) {
  case DistBaseKind::STD_NORMAL:
  case DistBaseKind::STD_UNIFORM:
    break;
  case DistBaseKind::EXPONENTIAL:
  case DistBaseKind::BERNOULLI:
  case DistBaseKind::POISSON:
  case DistBaseKind::HALF_NORMAL:
  case DistBaseKind::HALF_CAUCHY:
  case DistBaseKind::DIRICHLET:
    numParams = 1;
    break;
  case DistBaseKind::GAMMA:
  case DistBaseKind::DISCRETE_UNIFORM:
  case DistBaseKind::MULTIVARIATE_NORMAL:
  case DistBaseKind::GUMBEL:
  case DistBaseKind::HALF_STUDENT_T:
    numParams = 2;
    break;
  case DistBaseKind::STUDENT_T:
    numParams = 3;
    break;
  }
  if (ArrayAttr transforms = getTransforms())
    for (Attribute a : transforms)
      if (cast<XformAttr>(a).getKind() == XformKind::AFFINE)
        numParams = std::max(numParams, 2u);
  return numParams;
}

RankedTensorType DistAttr::getSampleType(TypeRange paramTypes) const {
  // A MULTIVARIATE_NORMAL sample has the type of its mean; no other base
  // fixes the type.
  if (getBase() != DistBaseKind::MULTIVARIATE_NORMAL || paramTypes.empty())
    return RankedTensorType();
  return dyn_cast<RankedTensorType>(paramTypes.front());
}
