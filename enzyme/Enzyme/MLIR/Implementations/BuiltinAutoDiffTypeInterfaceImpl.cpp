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
  Attribute createNullAttr(Type self) const {
    auto fltType = cast<ConcreteType>(self);
    return FloatAttr::get(fltType, APFloat(fltType.getFloatSemantics(), 0));
  }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto fltType = cast<ConcreteType>(self);
    return arith::ConstantOp::create(builder, loc, fltType,
                                     cast<FloatAttr>(createNullAttr(self)));
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return arith::AddFOp::create(builder, loc, a, b);
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

  bool isZero(Type self, Value val) const {
    return matchPattern(val, m_AnyZeroFloat());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    return matchPattern(attr, m_AnyZeroFloat());
  }

  int64_t getApproxSize(Type self) const {
    return self.getIntOrFloatBitWidth();
  }
};

class TensorTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<TensorTypeInterface,
                                                  TensorType> {
public:
  Attribute createNullAttr(Type self) const {
    auto tenType = cast<TensorType>(self);
    auto ET = tenType.getElementType();

    if (auto F = dyn_cast<FloatType>(ET)) {
      APFloat apvalue(F.getFloatSemantics(), 0);
      return DenseElementsAttr::get(tenType, apvalue);
    }
    if (auto G = dyn_cast<ComplexType>(ET)) {
      if (auto F = dyn_cast<FloatType>(G.getElementType())) {
        APFloat apvalue(F.getFloatSemantics(), 0);
        std::complex<APFloat> c(apvalue, apvalue);
        return DenseElementsAttr::get(tenType, c);
      }
    }
    if (auto IT = dyn_cast<IntegerType>(ET)) {
      APInt apvalue(IT.getWidth(), 0);
      return DenseElementsAttr::get(tenType, apvalue);
    }
    llvm::errs() << " cannot create null value of tensor type: " << tenType
                 << "\n";
    assert(0);
    return nullptr;
  }
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto attr = createNullAttr(self);
    assert(attr);
    auto tenType = cast<TensorType>(self);
    return arith::ConstantOp::create(builder, loc, tenType,
                                     cast<TypedAttr>(attr));
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

  bool isZero(Type self, Value val) const {
    auto tenType = cast<TensorType>(self);
    auto ET = tenType.getElementType();
    DenseElementsAttr eAttr;

    if (!matchPattern(val, m_Constant(&eAttr)))
      return false;

    if (!eAttr.isSplat())
      return false;
    // recurse on the individual element type
    auto splatVal = eAttr.getSplatValue<Attribute>();
    auto ADET = dyn_cast<AutoDiffTypeInterface>(ET);
    return ADET && ADET.isZeroAttr(splatVal);
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto eAttr = dyn_cast<DenseElementsAttr>(attr);
    if (!eAttr)
      return false;

    if (!eAttr.isSplat())
      return false;

    auto ET = eAttr.getType().getElementType();
    auto ADET = dyn_cast<AutoDiffTypeInterface>(ET);

    if (!ADET)
      return false;

    return ADET.isZeroAttr(eAttr.getSplatValue<Attribute>());
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
  Attribute createNullAttr(Type self) const {
    if (isa<IndexType>(self)) {
      return IntegerAttr::get(self, APInt(64, 0));
    } else {
      return IntegerAttr::get(self, APInt(self.getIntOrFloatBitWidth(), 0));
    }
  }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    if (isa<IndexType>(self)) {
      return arith::ConstantIndexOp::create(builder, loc, 0);
    }
    return arith::ConstantIntOp::create(builder, loc, self, 0);
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return arith::AddIOp::create(builder, loc, a, b);
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

  bool isZero(Type self, Value val) const {
    return matchPattern(val, m_Zero());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    return matchPattern(attr, m_Zero());
  }

  int64_t getApproxSize(Type self) const {
    // Assume index is 64-bit for ease
    if (self.isIndex())
      return 64;

    return self.getIntOrFloatBitWidth();
  }
};

class ComplexTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<ComplexTypeInterface,
                                                  ComplexType> {
public:
  Attribute createNullAttr(Type self) const {
    auto fltType = cast<FloatType>(cast<ComplexType>(self).getElementType());
    auto zattr = cast<AutoDiffTypeInterface>(fltType).createNullAttr();
    mlir::Attribute attrs[2] = {zattr, zattr};
    return ArrayAttr::get(self.getContext(), attrs);
  }
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return complex::ConstantOp::create(builder, loc, self,
                                       cast<ArrayAttr>(createNullAttr(self)));
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return complex::AddOp::create(builder, loc, a, b)->getResult(0);
  }
  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return complex::ConjOp::create(builder, loc, a)->getResult(0);
  }

  Type getShadowType(Type self, int64_t width) const {
    return batchType(self, width);
  }

  bool isMutable(Type self) const { return false; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    ArrayAttr arrayAttr;

    if (!matchPattern(val, m_Constant(&arrayAttr))) {
      return false;
    }
    // reuse attributr check
    return this->isZeroAttr(self, arrayAttr);
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.size() != 2)
      return false;

    // get the element type
    auto compType = cast<ComplexType>(self);
    auto elType = compType.getElementType();
    auto eltIntf = dyn_cast<AutoDiffTypeInterface>(elType);

    if (!eltIntf)
      return false;

    // recurse and accumulate info per attribute
    for (auto eltAttr : arrayAttr) {
      if (!eltIntf.isZeroAttr(eltAttr))
        return false;
    }

    return true;
  }

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
