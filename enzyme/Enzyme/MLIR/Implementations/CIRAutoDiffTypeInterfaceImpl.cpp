#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffTypeInterface.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

template <typename ConcreteType>
class CIRFloatTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<
          CIRFloatTypeInterface<ConcreteType>, ConcreteType> {
public:
  Attribute createNullAttr(Type self) const {
    return cir::FPAttr::getZero(self);
  }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto fltType = cast<ConcreteType>(self);
    return cir::ConstantOp::create(builder, loc, fltType,
                                   cast<cir::FPAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return cir::FAddOp::create(builder, loc, a, b).getResult();
  }
  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return a;
  }

  Type getShadowType(Type self, int64_t width) const {
    assert(width > 0 && "batch width must be positive");
    if (width == 1)
      return self;

    return cir::VectorType::get(self, static_cast<uint64_t>(width));
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    auto constant = val.getDefiningOp<cir::ConstantOp>();
    if (!constant)
      return false;

    return isZeroAttr(self, constant.getValue());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto fpAttr = dyn_cast<cir::FPAttr>(attr);
    return fpAttr && fpAttr.getValue().isZero();
  }

  int64_t getApproxSize(Type self) const {
    return cast<cir::FPTypeInterface>(self).getWidth();
  }
};

class CIRVectorTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<CIRVectorTypeInterface,
                                                  cir::VectorType> {
public:
  Attribute createNullAttr(Type self) const {
    auto vecType = cast<cir::VectorType>(self);
    auto elemType = cast<AutoDiffTypeInterface>(vecType.getElementType());
    auto elemZero = elemType.createNullAttr();

    SmallVector<Attribute> elts(vecType.getSize(), elemZero);
    auto eltsAttr = ArrayAttr::get(self.getContext(), elts);
    return cir::ConstVectorAttr::get(vecType, eltsAttr);
  }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return cir::ConstantOp::create(builder, loc, self,
                                   cast<TypedAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    auto vecType = cast<cir::VectorType>(self);
    auto elemType = vecType.getElementType();

    if (isa<cir::FPTypeInterface>(elemType))
      return cir::FAddOp::create(builder, loc, a, b).getResult();

    if (isa<cir::IntType>(elemType))
      return cir::AddOp::create(builder, loc, a, b).getResult();

    llvm_unreachable("unsupported CIR vector element type for add");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return a;
  }

  Type getShadowType(Type self, int64_t width) const {
    if (width == 1)
      return self;
    llvm_unreachable(
        "batched shadows of CIR aggregate types are not supported");
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    auto constant = val.getDefiningOp<cir::ConstantOp>();
    if (!constant)
      return false;
    return isZeroAttr(self, constant.getValue());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto vecAttr = dyn_cast<cir::ConstVectorAttr>(attr);
    if (!vecAttr)
      return false;

    auto vecType = cast<cir::VectorType>(self);
    auto elemType = dyn_cast<AutoDiffTypeInterface>(vecType.getElementType());
    if (!elemType)
      return false;

    auto elts = vecAttr.getElts();
    if (elts.size() != vecType.getSize())
      return false;

    for (Attribute elt : elts)
      if (!elemType.isZeroAttr(elt))
        return false;

    return true;
  }

  int64_t getApproxSize(Type self) const {
    auto vecType = cast<cir::VectorType>(self);
    auto elemType = cast<AutoDiffTypeInterface>(vecType.getElementType());
    int64_t elemSize = elemType.getApproxSize();
    if (elemSize == INT64_MAX)
      return INT64_MAX;
    return vecType.getSize() * elemSize;
  }
};

class CIRIntTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<CIRIntTypeInterface,
                                                  cir::IntType> {
public:
  Attribute createNullAttr(Type self) const {
    auto intType = cast<cir::IntType>(self);
    return cir::IntAttr::get(intType, APInt(intType.getWidth(), 0));
  }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return cir::ConstantOp::create(builder, loc, self,
                                   cast<TypedAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return cir::AddOp::create(builder, loc, a, b).getResult();
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return a;
  }

  Type getShadowType(Type self, int64_t width) const {
    if (width == 1)
      return self;
    assert(width > 0 && "batch width must be positive");
    return cir::VectorType::get(self, static_cast<uint64_t>(width));
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    auto constant = val.getDefiningOp<cir::ConstantOp>();
    if (!constant)
      return false;
    return isZeroAttr(self, constant.getValue());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto intAttr = dyn_cast<cir::IntAttr>(attr);
    return intAttr && intAttr.getValue().isZero();
  }

  int64_t getApproxSize(Type self) const {
    return cast<cir::IntType>(self).getWidth();
  }
};

class CIRBoolTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<CIRBoolTypeInterface,
                                                  cir::BoolType> {
public:
  Attribute createNullAttr(Type self) const {
    return cir::BoolAttr::get(self.getContext(), false);
  }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return cir::ConstantOp::create(builder, loc, self,
                                   cast<TypedAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("cannot add CIR bool shadow values");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return a;
  }

  Type getShadowType(Type self, int64_t width) const {
    if (width == 1)
      return self;
    assert(width > 0 && "batch width must be positive");
    return cir::VectorType::get(self, static_cast<uint64_t>(width));
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    auto constant = val.getDefiningOp<cir::ConstantOp>();
    if (!constant)
      return false;
    return isZeroAttr(self, constant.getValue());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto boolAttr = dyn_cast<cir::BoolAttr>(attr);
    return boolAttr && !boolAttr.getValue();
  }

  int64_t getApproxSize(Type self) const { return 1; }
};

class CIRComplexTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<CIRComplexTypeInterface,
                                                  cir::ComplexType> {
public:
  Attribute createNullAttr(Type self) const {
    auto complexType = cast<cir::ComplexType>(self);
    auto elemType = cast<AutoDiffTypeInterface>(complexType.getElementType());

    auto zero = cast<TypedAttr>(elemType.createNullAttr());
    return cir::ConstComplexAttr::get(zero, zero);
  }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return cir::ConstantOp::create(builder, loc, self,
                                   cast<TypedAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return cir::ComplexAddOp::create(builder, loc, a, b).getResult();
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    return cir::ComplexConjOp::create(builder, loc, a).getResult();
  }

  Type getShadowType(Type self, int64_t width) const {
    if (width == 1)
      return self;
    llvm_unreachable(
        "batched shadows of CIR aggregate types are not supported");
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    auto constant = val.getDefiningOp<cir::ConstantOp>();
    if (!constant)
      return false;

    return isZeroAttr(self, constant.getValue());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto complexAttr = dyn_cast<cir::ConstComplexAttr>(attr);
    if (!complexAttr)
      return false;

    auto complexType = cast<cir::ComplexType>(self);
    auto elemType =
        dyn_cast<AutoDiffTypeInterface>(complexType.getElementType());
    if (!elemType)
      return false;

    return elemType.isZeroAttr(complexAttr.getReal()) &&
           elemType.isZeroAttr(complexAttr.getImag());
  }

  int64_t getApproxSize(Type self) const {
    auto complexType = cast<cir::ComplexType>(self);
    auto elemType = cast<AutoDiffTypeInterface>(complexType.getElementType());

    int64_t elemSize = elemType.getApproxSize();
    if (elemSize == INT64_MAX)
      return INT64_MAX;

    return 2 * elemSize;
  }
};

class CIRArrayTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<CIRArrayTypeInterface,
                                                  cir::ArrayType> {
public:
  Attribute createNullAttr(Type self) const { return cir::ZeroAttr::get(self); }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return cir::ConstantOp::create(builder, loc, self,
                                   cast<TypedAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("CIR array addition is not implemented yet");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("CIR array conjugation is not implemented yet");
  }

  Type getShadowType(Type self, int64_t width) const {
    if (width == 1)
      return self;
    llvm_unreachable(
        "batched shadows of CIR aggregate types are not supported");
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    auto constant = val.getDefiningOp<cir::ConstantOp>();
    return constant && isZeroAttr(self, constant.getValue());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    auto arrayType = cast<cir::ArrayType>(self);

    if (auto zero = dyn_cast<cir::ZeroAttr>(attr))
      return zero.getType() == self;

    auto arrayAttr = dyn_cast<cir::ConstArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.getType() != self)
      return false;

    if (auto str = dyn_cast<StringAttr>(arrayAttr.getElts())) {
      for (char ch : str.getValue())
        if (ch != '\0')
          return false;
      return true;
    }

    auto elements = dyn_cast<ArrayAttr>(arrayAttr.getElts());
    if (!elements)
      return false;

    if (elements.empty())
      return true;

    auto elemIface =
        dyn_cast<AutoDiffTypeInterface>(arrayType.getElementType());
    if (!elemIface)
      return false;

    for (Attribute element : elements)
      if (!elemIface.isZeroAttr(element))
        return false;

    return true;
  }

  int64_t getApproxSize(Type self) const {
    auto arrayType = cast<cir::ArrayType>(self);
    uint64_t count = arrayType.getSize();
    if (count == 0)
      return 0;

    auto elemIface =
        dyn_cast<AutoDiffTypeInterface>(arrayType.getElementType());
    if (!elemIface)
      return INT64_MAX;

    int64_t elemSize = elemIface.getApproxSize();
    if (elemSize < 0 || elemSize == INT64_MAX)
      return INT64_MAX;

    if (elemSize != 0 && count > static_cast<uint64_t>(INT64_MAX / elemSize))
      return INT64_MAX;

    return static_cast<int64_t>(count) * elemSize;
  }
};

class CIRPointerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<CIRPointerTypeInterface,
                                                  cir::PointerType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    auto i64 = IntegerType::get(self.getContext(), 64);
    return cir::ConstPtrAttr::get(self, IntegerAttr::get(i64, 0));
  }

  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    return cir::ConstantOp::create(builder, loc, self,
                                   cast<TypedAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("createAddOp on a CIR pointer shadow");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("createConjOp on a CIR pointer shadow");
  }

  Type getShadowType(Type self, int64_t width) const {
    if (width == 1)
      return self;
    llvm_unreachable("batched pointer shadows are not supported for CIR");
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    auto allocaOp = val.getDefiningOp<cir::AllocaOp>();
    if (!allocaOp)
      return failure();
    // A VLA needs a byte-sized memset; refuse for now.
    if (allocaOp.getDynAllocSize())
      return failure();
    Type elemTy = allocaOp.getAllocaType();
    Value zero;
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(elemTy)) {
      zero = iface.createNullValue(builder, loc);
    } else if (isa<cir::VPtrType>(elemTy)) {
      zero = cir::ConstantOp::create(builder, loc, elemTy,
                                     cir::ZeroAttr::get(elemTy));
    } else {
      return failure();
    }
    cir::StoreOp::create(builder, loc, zero, val);
    return success();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

/*
 * Handle shadow and union type
 */
template <typename ConcreteType>
class CIRRecordTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<
          CIRRecordTypeInterface<ConcreteType>, ConcreteType> {
public:
  Attribute createNullAttr(Type self) const { return cir::ZeroAttr::get(self); }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    return cir::ConstantOp::create(builder, loc, self,
                                   cast<TypedAttr>(createNullAttr(self)))
        .getResult();
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    auto recTy = cast<cir::RecordType>(self);
    if (recTy.isUnion())
      llvm_unreachable("adding shadows of a CIR union is not supported");
    Value result = createNullValue(self, builder, loc);
    for (auto &&[i, elemTy] : llvm::enumerate(recTy.getMembers())) {
      Value aElem = cir::ExtractMemberOp::create(builder, loc, a, i);
      Value sum = aElem;
      if (auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy)) {
        Value bElem = cir::ExtractMemberOp::create(builder, loc, b, i);
        sum = elemIface.createAddOp(builder, loc, aElem, bElem);
      }
      result = cir::InsertMemberOp::create(builder, loc, result, i, sum);
    }
    return result;
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("batched shadows of CIR records are not supported");
  }

  Type getShadowType(Type self, int64_t width) const {
    if (width == 1)
      return self;
    llvm_unreachable("batched shadows of CIR records are not supported");
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const {
    auto constant = val.getDefiningOp<cir::ConstantOp>();
    return constant && isZeroAttr(self, constant.getValue());
  }

  bool isZeroAttr(Type self, Attribute attr) const {
    if (auto zero = dyn_cast<cir::ZeroAttr>(attr))
      return zero.getType() == self;
    auto rec = dyn_cast<cir::ConstRecordAttr>(attr);
    if (!rec || rec.getType() != self)
      return false;
    auto recTy = cast<cir::RecordType>(self);
    for (auto &&[elemTy, elem] :
         llvm::zip(recTy.getMembers(), rec.getMembers())) {
      auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
      if (!elemIface || !elemIface.isZeroAttr(elem))
        return false;
    }
    return true;
  }

  int64_t getApproxSize(Type self) const {
    int64_t total = 0;
    for (Type elemTy : cast<cir::RecordType>(self).getMembers()) {
      auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
      if (!elemIface)
        return INT64_MAX;
      int64_t sz = elemIface.getApproxSize();
      if (sz == INT64_MAX)
        return INT64_MAX;
      total += sz;
    }
    return total;
  }
};

} // namespace

void mlir::enzyme::registerCIRAutoDiffTypeInterfaces(MLIRContext *context) {
  cir::SingleType::attachInterface<CIRFloatTypeInterface<cir::SingleType>>(
      *context);
  cir::DoubleType::attachInterface<CIRFloatTypeInterface<cir::DoubleType>>(
      *context);
  cir::FP16Type::attachInterface<CIRFloatTypeInterface<cir::FP16Type>>(
      *context);
  cir::BF16Type::attachInterface<CIRFloatTypeInterface<cir::BF16Type>>(
      *context);
  cir::FP80Type::attachInterface<CIRFloatTypeInterface<cir::FP80Type>>(
      *context);
  cir::FP128Type::attachInterface<CIRFloatTypeInterface<cir::FP128Type>>(
      *context);
  cir::LongDoubleType::attachInterface<
      CIRFloatTypeInterface<cir::LongDoubleType>>(*context);

  cir::IntType::attachInterface<CIRIntTypeInterface>(*context);
  cir::BoolType::attachInterface<CIRBoolTypeInterface>(*context);
  cir::VectorType::attachInterface<CIRVectorTypeInterface>(*context);
  cir::ComplexType::attachInterface<CIRComplexTypeInterface>(*context);
  cir::ArrayType::attachInterface<CIRArrayTypeInterface>(*context);
  cir::PointerType::attachInterface<CIRPointerTypeInterface>(*context);
  cir::StructType::attachInterface<CIRRecordTypeInterface<cir::StructType>>(
      *context);
  cir::UnionType::attachInterface<CIRRecordTypeInterface<cir::UnionType>>(
      *context);
}
