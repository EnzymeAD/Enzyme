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
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

static mlir::Type batchType(mlir::Type type, int64_t width) {
  if (width > 1 || ShapedType::isDynamic(width)) {
    if (auto TT = dyn_cast<mlir::TensorType>(type)) {
      SmallVector<int64_t> shape;
      shape.reserve(TT.getShape().size() + 1);
      shape.push_back(width);
      shape.append(TT.getShape().begin(), TT.getShape().end());
      return TT.clone(shape);
    }
    return RankedTensorType::get({width}, type);
  }
  return type;
}

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
  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return a;
  }

  Type getShadowType(Type self, int64_t width) const {
    return batchType(self, width);
  }

  bool isMutable(Type self) const { return false; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }
};

class TensorTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<TensorTypeInterface,
                                                  TensorType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto tenType = self.cast<TensorType>();
    auto ET = tenType.getElementType();

    if (auto F = dyn_cast<FloatType>(ET)) {
      APFloat apvalue(F.getFloatSemantics(), 0);
      auto attr = DenseElementsAttr::get(tenType, apvalue);
      return builder.create<arith::ConstantOp>(loc, tenType, attr);
    }
    if (auto G = dyn_cast<ComplexType>(ET)) {
      if (auto F = dyn_cast<FloatType>(G.getElementType())) {
        APFloat apvalue(F.getFloatSemantics(), 0);
        std::complex<APFloat> c(apvalue, apvalue);
        auto attr = DenseElementsAttr::get(tenType, c);
        return builder.create<arith::ConstantOp>(loc, tenType, attr);
      }
    }
    if (auto IT = dyn_cast<IntegerType>(ET)) {
      APInt apvalue(IT.getWidth(), 0);
      auto attr = DenseElementsAttr::get(tenType, apvalue);
      return builder.create<arith::ConstantOp>(loc, tenType, attr);
    }
    llvm::errs() << " cannot create null value of tensor type: " << tenType
                 << "\n";
    assert(0);
    return nullptr;
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    auto tenType = self.cast<TensorType>();
    auto ET = tenType.getElementType();
    auto iface = cast<AutoDiffTypeInterface>(ET);
    return iface.createAddOp(builder, loc, a, b);
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    auto tenType = self.cast<TensorType>();
    auto ET = tenType.getElementType();
    auto iface = cast<AutoDiffTypeInterface>(ET);
    auto added = iface.createConjOp(builder, loc, a);
    return added;
  }

  Type getShadowType(Type self, int64_t width) const {
    return batchType(self, width);
  }

  bool isMutable(Type self) const { return false; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }
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

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return a;
  }

  Type getShadowType(Type self, int64_t width) const {
    return batchType(self, width);
  }

  bool isMutable(Type self) const { return false; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }
};

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
    return builder.create<complex::AddOp>(loc, a, b)->getResult(0);
  }
  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return builder.create<complex::ConjOp>(loc, a)->getResult(0);
  }

  Type getShadowType(Type self, int64_t width) const {
    return batchType(self, width);
  }

  bool isMutable(Type self) const { return false; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }
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
    UnrankedTensorType::attachInterface<TensorTypeInterface>(*context);
    RankedTensorType::attachInterface<TensorTypeInterface>(*context);
    ComplexType::attachInterface<ComplexTypeInterface>(*context);
  });
}
