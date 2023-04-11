//===- BuiltinAutoDiffOpInterfaceImpl.cpp - Interface external model ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation type interfaces for the upstream MLIR builtin dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
class FloatTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<FloatTypeInterface,
                                                  FloatType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto fltType = self.cast<FloatType>();
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(fltType.getFloatSemantics(), 0), fltType);
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return builder.create<arith::AddFOp>(loc, a, b);
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool requiresShadow(Type self) const { return false; }
};

template <typename T>
class IntegerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<IntegerTypeInterface<T>, T> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    if (isa<IndexType>(self)) {
      return builder.create<arith::ConstantIndexOp>(loc, 0);
    }
    return builder.create<arith::ConstantIntOp>(loc, 0, self);
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return builder.create<arith::AddIOp>(loc, a, b);
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool requiresShadow(Type self) const { return false; }
};
} // namespace

void mlir::enzyme::registerBuiltinDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, BuiltinDialect *) {
    BFloat16Type::attachInterface<FloatTypeInterface>(*context);
    Float16Type::attachInterface<FloatTypeInterface>(*context);
    Float32Type::attachInterface<FloatTypeInterface>(*context);
    Float64Type::attachInterface<FloatTypeInterface>(*context);
    IntegerType::attachInterface<IntegerTypeInterface<IntegerType>>(*context);
    IndexType::attachInterface<IntegerTypeInterface<IndexType>>(*context);
  });
}
