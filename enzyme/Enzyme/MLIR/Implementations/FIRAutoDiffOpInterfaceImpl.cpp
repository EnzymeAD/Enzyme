//===- FIRAutoDiffOpInterfaceImpl.cpp - FIR/HLFIR autodiff models --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external autodiff models needed by the Flang plugin
// while Enzyme runs in the early HLFIR pipeline.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

class FIRReferenceTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<FIRReferenceTypeInterface,
                                                  fir::ReferenceType> {
public:
  Attribute createNullAttr(Type self) const { return {}; }

  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto elementType = cast<fir::ReferenceType>(self).getElementType();
    Value allocation = fir::AllocaOp::create(builder, loc, elementType);
    auto elementInterface = cast<AutoDiffTypeInterface>(elementType);
    fir::StoreOp::create(builder, loc,
                         elementInterface.createNullValue(builder, loc),
                         allocation);
    return allocation;
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("cannot add FIR references");
  }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value value) const {
    auto elementType = cast<fir::ReferenceType>(self).getElementType();
    auto elementInterface = dyn_cast<AutoDiffTypeInterface>(elementType);
    if (!elementInterface || elementInterface.isMutable())
      return failure();
    fir::StoreOp::create(builder, loc,
                         elementInterface.createNullValue(builder, loc), value);
    return success();
  }

  bool isZero(Type self, Value value) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }

  Type getShadowType(Type self, int64_t width) const {
    assert(width == 1 && "FIR references do not support width != 1 yet");
    auto reference = cast<fir::ReferenceType>(self);
    auto elementInterface =
        cast<AutoDiffTypeInterface>(reference.getElementType());
    return fir::ReferenceType::get(elementInterface.getShadowType(),
                                   reference.isVolatile());
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value value) const {
    llvm_unreachable("cannot conjugate FIR references");
  }

  bool isMutable(Type self) const { return true; }
};

template <typename OpTy>
struct AllOperandsActive
    : public ActivityOpInterface::ExternalModel<AllOperandsActive<OpTy>, OpTy> {
  bool isInactive(Operation *op) const { return false; }
  bool isArgInactive(Operation *op, size_t index) const { return false; }
};

struct HLFIRDeclareActivity
    : public ActivityOpInterface::ExternalModel<HLFIRDeclareActivity,
                                                hlfir::DeclareOp> {
  bool isInactive(Operation *op) const { return false; }

  bool isArgInactive(Operation *op, size_t index) const {
    auto declare = cast<hlfir::DeclareOp>(op);
    Value operand = op->getOperand(index);
    if (operand == declare.getMemref())
      return false;
    Value storage = declare.getStorage();
    return !storage || operand != storage;
  }
};

struct FIRStoreLike
    : public StoreLikeInterface::ExternalModel<FIRStoreLike, fir::StoreOp> {
  Value getStoredValue(Operation *op) const {
    return cast<fir::StoreOp>(op).getValue();
  }
  Value getStoredPointer(Operation *op) const {
    return cast<fir::StoreOp>(op).getMemref();
  }
};

struct HLFIRAssignStoreLike
    : public StoreLikeInterface::ExternalModel<HLFIRAssignStoreLike,
                                               hlfir::AssignOp> {
  Value getStoredValue(Operation *op) const {
    return cast<hlfir::AssignOp>(op).getRhs();
  }
  Value getStoredPointer(Operation *op) const {
    return cast<hlfir::AssignOp>(op).getLhs();
  }
};

struct InactiveOperation
    : public ActivityOpInterface::ExternalModel<InactiveOperation,
                                                fir::DummyScopeOp> {
  bool isInactive(Operation *op) const { return true; }
  bool isArgInactive(Operation *op, size_t index) const { return true; }
};

} // namespace

void mlir::enzyme::registerFlangDialectAutoDiffInterfaces(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, fir::FIROpsDialect *) {
    fir::ReferenceType::attachInterface<FIRReferenceTypeInterface>(*context);

    fir::LoadOp::attachInterface<AllOperandsActive<fir::LoadOp>>(*context);
    fir::StoreOp::attachInterface<AllOperandsActive<fir::StoreOp>>(*context);
    fir::StoreOp::attachInterface<FIRStoreLike>(*context);
    fir::DummyScopeOp::attachInterface<InactiveOperation>(*context);

    registerAutoDiffUsingMemoryIdentityInterface<fir::LoadOp>(*context);
    registerAutoDiffUsingMemoryIdentityInterface<fir::StoreOp, 0>(*context);
    registerAutoDiffUsingAllocationInterface<fir::AllocaOp>(*context);
  });

  registry.addExtension(+[](MLIRContext *context, hlfir::hlfirDialect *) {
    hlfir::DeclareOp::attachInterface<HLFIRDeclareActivity>(*context);
    hlfir::AssignOp::attachInterface<AllOperandsActive<hlfir::AssignOp>>(
        *context);
    hlfir::AssignOp::attachInterface<HLFIRAssignStoreLike>(*context);

    registerAutoDiffUsingMemoryIdentityInterface<hlfir::DeclareOp>(*context);
    registerAutoDiffUsingMemoryIdentityInterface<hlfir::AssignOp, 0>(*context);
  });
}
