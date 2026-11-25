//===- TransformUtils.cpp - Constraint transforms for HMC ------* C++ -*-===//
//
// This file implements constraint transforms for HMC inference.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py
//
//===----------------------------------------------------------------------===//

#include "TransformUtils.h"

#include "Interfaces/DistBuilder.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::transforms;

int64_t transforms::getUnconstrainedSize(int64_t constrainedSize,
                                         impulse::SupportAttr support) {
  if (!support || support.getKind() != impulse::SupportKind::SIMPLEX)
    return constrainedSize;
  int64_t eventSize = support.getEventSize().getInt();
  return constrainedSize / eventSize * (eventSize - 1);
}

Value transforms::unconstrain(OpBuilder &builder, Location loc,
                              Value constrained, impulse::SupportAttr support) {
  return impulse::emitBijectorInverse(
      builder, loc, impulse::bijectorFor(builder.getContext(), support),
      support, constrained);
}

Value transforms::constrain(OpBuilder &builder, Location loc,
                            Value unconstrained, impulse::SupportAttr support) {
  return impulse::emitBijectorForward(
      builder, loc, impulse::bijectorFor(builder.getContext(), support),
      support, unconstrained);
}

Value transforms::logAbsDetJacobian(OpBuilder &builder, Location loc,
                                    Value unconstrained,
                                    impulse::SupportAttr support) {
  return impulse::emitBijectorLogAbsDetJac(
      builder, loc, impulse::bijectorFor(builder.getContext(), support),
      support, unconstrained);
}
