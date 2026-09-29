#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffTypeInterface.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"

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
    assert(width == 1 && "CIR batched derivatives are not supported yet");
    return self;
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

} // namespace