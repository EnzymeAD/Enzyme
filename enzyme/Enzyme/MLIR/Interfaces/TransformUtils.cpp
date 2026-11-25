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
  return impulse::bijectorInverseSize(constrainedSize, support);
}

Value transforms::unconstrain(OpBuilder &builder, Location loc,
                              Value constrained, impulse::SupportAttr support) {
  SmallVector<impulse::XformAttr> chain =
      impulse::bijectorFor(builder.getContext(), support);
  Value v = constrained;
  for (impulse::XformAttr xform : chain)
    v = impulse::emitBijectorInverse(builder, loc, xform, support, v);
  return v;
}

Value transforms::constrain(OpBuilder &builder, Location loc,
                            Value unconstrained, impulse::SupportAttr support) {
  SmallVector<impulse::XformAttr> chain =
      impulse::bijectorFor(builder.getContext(), support);
  Value v = unconstrained;
  for (impulse::XformAttr xform : chain)
    v = impulse::emitBijectorForward(builder, loc, xform, support, v);
  return v;
}

Value transforms::logAbsDetJacobian(OpBuilder &builder, Location loc,
                                    Value unconstrained,
                                    impulse::SupportAttr support) {
  // Support bijector chains have exactly one element.
  SmallVector<impulse::XformAttr> chain =
      impulse::bijectorFor(builder.getContext(), support);
  return impulse::emitBijectorLogAbsDetJac(builder, loc, chain.front(), support,
                                           unconstrained);
}
