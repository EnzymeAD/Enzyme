//===- TransformUtils.cpp - Constraint transforms for HMC ------* C++ -*-===//
//
// This file implements constraint transforms for HMC inference.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py
//
//===----------------------------------------------------------------------===//

#include "TransformUtils.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::transforms;

Value transforms::createLogit(OpBuilder &builder, Location loc, Value x) {
  return impulse::emitLogit(builder, loc, x);
}

Value transforms::createLogSigmoid(OpBuilder &builder, Location loc, Value x) {
  return impulse::emitLogSigmoid(builder, loc, x);
}

int64_t transforms::getUnconstrainedSize(int64_t constrainedSize,
                                         impulse::SupportAttr support) {
  return impulse::bijectorInverseSize(constrainedSize, support);
}

int64_t transforms::getConstrainedSize(int64_t unconstrainedSize,
                                       impulse::SupportAttr support) {
  return impulse::bijectorForwardSize(unconstrainedSize, support);
}

Value transforms::unconstrain(OpBuilder &builder, Location loc,
                              Value constrained, impulse::SupportAttr support) {
  auto chain = impulse::bijectorFor(builder.getContext(), support);
  Value v = constrained;
  for (impulse::XformAttr xform : chain)
    v = impulse::emitBijectorInverse(builder, loc, xform, support, v);
  return v;
}

Value transforms::constrain(OpBuilder &builder, Location loc,
                            Value unconstrained, impulse::SupportAttr support) {
  auto chain = impulse::bijectorFor(builder.getContext(), support);
  Value v = unconstrained;
  for (impulse::XformAttr xform : chain)
    v = impulse::emitBijectorForward(builder, loc, xform, support, v);
  return v;
}

Value transforms::logAbsDetJacobian(OpBuilder &builder, Location loc,
                                    Value unconstrained,
                                    impulse::SupportAttr support) {
  // Support bijector chains have exactly one element.
  auto chain = impulse::bijectorFor(builder.getContext(), support);
  return impulse::emitBijectorLogAbsDetJac(builder, loc, chain.front(), support,
                                           unconstrained);
}
