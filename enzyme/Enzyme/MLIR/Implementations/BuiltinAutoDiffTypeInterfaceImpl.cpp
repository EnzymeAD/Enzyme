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
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

static mlir::Type batchType(mlir::Type type, int64_t width) {
  if (width == 1)
    return type;

  if (auto TT = dyn_cast<mlir::TensorType>(type)) {
    SmallVector<int64_t> shape;
    shape.reserve(TT.getShape().size() + 1);
    shape.push_back(width);
    shape.append(TT.getShape().begin(), TT.getShape().end());
    return TT.clone(shape);
  }

  return RankedTensorType::get({width}, type);
}

template <typename ConcreteType>
class FloatTypeInterface : public AutoDiffTypeInterface::ExternalModel<
                               FloatTypeInterface<ConcreteType>, ConcreteType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto fltType = cast<ConcreteType>(self);
    return builder.create<arith::ConstantFloatOp>(
        loc, fltType, APFloat(fltType.getFloatSemantics(), 0));
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

  LogicalResult isZero(Type self, Value val) const {
    if (matchPattern(val, m_AnyZeroFloat())) {
      return success();
    }
    return failure();
  }

  int64_t getApproxSize(Type self) const {
    return self.getIntOrFloatBitWidth();
  }
};

class TensorTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<TensorTypeInterface,
                                                  TensorType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto tenType = cast<TensorType>(self);
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
    auto tenType = cast<TensorType>(self);
    auto ET = tenType.getElementType();
    auto iface = cast<AutoDiffTypeInterface>(ET);
    return iface.createAddOp(builder, loc, a, b);
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    auto tenType = cast<TensorType>(self);
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

  LogicalResult isZero(Type self, Value val) const {
    auto tenType = cast<TensorType>(self);
    auto ET = tenType.getElementType();
    DenseElementsAttr eAttr;

    if (!matchPattern(val, m_Constant(&eAttr)))
      return failure();

    if (eAttr.isSplat()) {
      auto splatVal = eAttr.getSplatValue<Attribute>();

      if (ET.isa<IntegerType>()) {
        return matchPattern(splatVal, m_Zero()) ? success() : failure();
      } else if (ET.isa<FloatType>()) {
        return matchPattern(splatVal, m_AnyZeroFloat()) ? success() : failure();
      } else {
        // TODO: handle complex
        return failure();
      }
    } else {
      if (ET.isa<IntegerType>()) {
        return llvm::all_of(eAttr.getValues<APInt>(),
                            [](const APInt &val) { return val.isZero(); })
                   ? success()
                   : failure();
      } else if (ET.isa<FloatType>()) {
        return llvm::all_of(eAttr.getValues<APFloat>(),
                            [](const APFloat &val) { return val.isZero(); })
                   ? success()
                   : failure();
      } else {
        // TODO: handle complex
        return failure();
      }
    }
    return failure();
  }

  int64_t getApproxSize(Type self) const {
    auto tenType = cast<TensorType>(self);
    auto elType = cast<AutoDiffTypeInterface>(tenType.getElementType());
    if (!elType)
      return INT64_MAX;
    int64_t sz = elType.getApproxSize();
    if (sz == INT64_MAX)
      return sz;
    for (auto n : tenType.getShape())
      sz *= n;
    return sz;
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
    return builder.create<arith::ConstantIntOp>(loc, self, 0);
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

  LogicalResult isZero(Type self, Value val) const { return failure(); }
  int64_t getApproxSize(Type self) const {
    return self.getIntOrFloatBitWidth();
  }
};

class ComplexTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<ComplexTypeInterface,
                                                  ComplexType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto fltType = cast<FloatType>(cast<ComplexType>(self).getElementType());
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

  LogicalResult isZero(Type self, Value val) const { return failure(); }

  int64_t getApproxSize(Type self) const {
    auto elType =
        cast<AutoDiffTypeInterface>(cast<ComplexType>(self).getElementType());
    auto elSize = elType.getApproxSize();
    if (elSize == INT64_MAX)
      return elSize;
    return 2 * elSize;
  }
};
} // namespace

void mlir::enzyme::registerBuiltinDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, BuiltinDialect *) {
    BFloat16Type::attachInterface<FloatTypeInterface<BFloat16Type>>(*context);
    Float16Type::attachInterface<FloatTypeInterface<Float16Type>>(*context);
    Float32Type::attachInterface<FloatTypeInterface<Float32Type>>(*context);
    Float64Type::attachInterface<FloatTypeInterface<Float64Type>>(*context);
    IntegerType::attachInterface<IntegerTypeInterface<IntegerType>>(*context);
    IndexType::attachInterface<IntegerTypeInterface<IndexType>>(*context);
    UnrankedTensorType::attachInterface<TensorTypeInterface>(*context);
    RankedTensorType::attachInterface<TensorTypeInterface>(*context);
    ComplexType::attachInterface<ComplexTypeInterface>(*context);
  });
}
