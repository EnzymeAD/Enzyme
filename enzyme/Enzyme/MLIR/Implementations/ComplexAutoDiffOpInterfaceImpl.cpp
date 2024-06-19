//===- ComplexAutoDiffOpInterfaceImpl.cpp - Interface external model --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR complex dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/ComplexDerivatives.inc"

class ComplexTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<ComplexTypeInterface,
                                                  ComplexType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto fltType = self.cast<ComplexType>().getElementType().cast<FloatType>();
    mlir::Attribute attrs[2] = {
        builder.getFloatAttr(fltType, APFloat(fltType.getFloatSemantics(), 0)),
        builder.getFloatAttr(fltType, APFloat(fltType.getFloatSemantics(), 0))};
    return builder.create<complex::ConstantOp>(loc, self,
                                               builder.getArrayAttr(attrs));
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return builder.create<complex::AddOp>(loc, a, b);
  }
  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return builder.create<complex::ConjOp>(loc, a);
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return false; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }
};
} // namespace

void mlir::enzyme::registerComplexDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, complex::ComplexDialect *) {
    registerInterfaces(context);
    ComplexType::attachInterface<ComplexTypeInterface>(*context);
  });
}
