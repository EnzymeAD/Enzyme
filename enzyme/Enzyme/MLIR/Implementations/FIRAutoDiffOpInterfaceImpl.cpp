//===- FIRAutoDiffOpInterfaceImpl.cpp - FIR active-memory AD --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enzyme autodiff support for FIR's by-reference memory model, so whole Fortran
// functions differentiate: their arguments and locals are !fir.ref<T> and the
// body is fir.load / hlfir.assign / fir.alloca / hlfir.declare over that
// memory.
//
//   * AutoDiffTypeInterface for !fir.ref<T>: mutable, shadow = a parallel ref.
//   * The memory ops are registered as active-memory identities via the
//     declarative FIRDerivatives.td (mirrors the memref dialect).
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/FIRDerivatives.inc"

// The adjoint of a !fir.ref<T> is a parallel !fir.ref<T> (its shadow buffer).
struct FIRReferenceTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<FIRReferenceTypeInterface,
                                                  fir::ReferenceType> {
  Type getShadowType(Type self, int64_t width) const {
    assert(width == 1 && "unsupported width != 1 for !fir.ref");
    return self;
  }

  bool isMutable(Type self) const { return true; }

  // A zeroed shadow buffer: allocate the pointee and store its null value.
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto refTy = cast<fir::ReferenceType>(self);
    Type eleTy = refTy.getEleTy();
    Value ref = fir::AllocaOp::create(builder, loc, eleTy, /*uniqName=*/"",
                                      /*bindcName=*/"");
    if (auto eIface = dyn_cast<AutoDiffTypeInterface>(eleTy);
        eIface && !eIface.isMutable())
      fir::StoreOp::create(builder, loc, eIface.createNullValue(builder, loc),
                           ref);
    return ref;
  }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    auto refTy = cast<fir::ReferenceType>(self);
    auto eIface = dyn_cast<AutoDiffTypeInterface>(refTy.getEleTy());
    if (!eIface || eIface.isMutable())
      return failure();
    fir::StoreOp::create(builder, loc, eIface.createNullValue(builder, loc),
                         val);
    return success();
  }

  // Not used for a mutable type (memory ops accumulate through load/store).
  Attribute createNullAttr(Type self) const { return {}; }
  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("createAddOp on mutable !fir.ref");
  }
  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("createConjOp on mutable !fir.ref");
  }
  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

// Expose the (storedValue, pointer) pair so activity analysis can reason about
// these stores without hard-coding the FIR/HLFIR dialects.
struct FIRStoreActiveStore
    : public StoreLikeInterface::ExternalModel<FIRStoreActiveStore,
                                               fir::StoreOp> {
  Value getStoredValue(Operation *op) const {
    return cast<fir::StoreOp>(op).getValue();
  }
  Value getStoredPointer(Operation *op) const {
    return cast<fir::StoreOp>(op).getMemref();
  }
};

struct HLFIRAssignActiveStore
    : public StoreLikeInterface::ExternalModel<HLFIRAssignActiveStore,
                                               hlfir::AssignOp> {
  Value getStoredValue(Operation *op) const {
    return cast<hlfir::AssignOp>(op).getRhs();
  }
  Value getStoredPointer(Operation *op) const {
    return cast<hlfir::AssignOp>(op).getLhs();
  }
};

//===----------------------------------------------------------------------===//
// Reverse-mode adjoints for the FIR/HLFIR by-reference memory ops.
//
// The forward tangents come from the declarative FIRDerivatives.td rules; the
// reverse pass instead requires each active memory op to carry a
// ReverseAutoDiffOpInterface. These mirror the LLVM dialect handlers, which are
// the closest analog: !fir.ref is a scalar pointer with no explicit index
// operands (unlike memref).
//===----------------------------------------------------------------------===//

// fir.load reads through a reference; its adjoint accumulates the loaded
// value's gradient back into the shadow reference (shadow += diffe(load)).
struct FIRLoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          FIRLoadOpInterfaceReverse, fir::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<fir::LoadOp>(op);
    Value ref = loadOp.getMemref();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      if (!gutils->isConstantValue(loadOp) && !gutils->isConstantValue(ref)) {
        Value gradient = gutils->diffe(loadOp, builder);
        Value refGradient = gutils->popCache(caches.front(), builder);

        Value loadedGradient =
            fir::LoadOp::create(builder, loadOp.getLoc(), refGradient);
        Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                loadedGradient, gradient);
        fir::StoreOp::create(builder, loadOp.getLoc(), addedGradient,
                             refGradient);
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<fir::LoadOp>(op);
    Value ref = loadOp.getMemref();
    if (!(isa<AutoDiffTypeInterface>(loadOp.getType()) &&
          !gutils->isConstantValue(loadOp) && !gutils->isConstantValue(ref)))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(ref, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

// fir.store writes a value through a reference; its adjoint reads the shadow
// reference's gradient into the stored value's diffe, then zeroes the shadow.
struct FIRStoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          FIRStoreOpInterfaceReverse, fir::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<fir::StoreOp>(op);
    Value val = storeOp.getValue();
    Value ref = storeOp.getMemref();

    auto iface = cast<AutoDiffTypeInterface>(val.getType());

    if (!gutils->isConstantValue(ref)) {
      Value refGradient = gutils->popCache(caches.front(), builder);

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient =
              fir::LoadOp::create(builder, storeOp.getLoc(), refGradient);
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        Value zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());
        fir::StoreOp::create(builder, storeOp.getLoc(), zero, refGradient);
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<fir::StoreOp>(op);
    Value ref = storeOp.getMemref();
    if (gutils->isConstantValue(ref))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(ref, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

// hlfir.assign is store-like (RHS value written through the LHS variable); its
// scalar adjoint matches fir.store, reading/zeroing the shadow LHS reference.
struct HLFIRAssignOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          HLFIRAssignOpInterfaceReverse, hlfir::AssignOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto assignOp = cast<hlfir::AssignOp>(op);
    Value val = assignOp.getRhs();
    Value ref = assignOp.getLhs();

    if (!gutils->isConstantValue(ref)) {
      Value refGradient = gutils->popCache(caches.front(), builder);

      auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType());
      if (iface && !iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient =
              fir::LoadOp::create(builder, assignOp.getLoc(), refGradient);
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        Value zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());
        fir::StoreOp::create(builder, assignOp.getLoc(), zero, refGradient);
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto assignOp = cast<hlfir::AssignOp>(op);
    Value val = assignOp.getRhs();
    Value ref = assignOp.getLhs();
    auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType());
    if (!iface || iface.isMutable() || gutils->isConstantValue(ref))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(ref, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

// hlfir.declare re-derives variable/base handles from a reference; it forwards
// the pointer, so in reverse mode we create a shadow declare over the shadow
// input and register both shadow handles (mirrors LLVM's GEP handler).
struct HLFIRDeclareOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          HLFIRDeclareOpInterfaceReverse, hlfir::DeclareOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto declareOp = cast<hlfir::DeclareOp>(op);
    Value input = declareOp.getMemref();
    if (gutils->isConstantValue(input))
      return;
    auto newDeclare = cast<hlfir::DeclareOp>(gutils->getNewFromOriginal(op));
    auto shadowDeclare = cast<hlfir::DeclareOp>(builder.clone(*newDeclare));
    shadowDeclare.getMemrefMutable().assign(
        gutils->invertPointerM(input, builder));
    for (auto &&[orig, shadow] :
         llvm::zip(declareOp.getResults(), shadowDeclare.getResults())) {
      if (!gutils->isConstantValue(orig))
        gutils->setInvertedPointer(orig, shadow);
    }
  }
};

} // namespace

void mlir::enzyme::registerFIRDialectAutoDiffInterface(
    DialectRegistry &registry) {
  // registerInterfaces attaches to both FIR (fir.load/store/alloca) and HLFIR
  // (hlfir.declare/assign) ops, so run once both dialects are loaded.
  registry.addExtension(+[](MLIRContext *context, fir::FIROpsDialect *,
                            hlfir::hlfirDialect *) {
    fir::ReferenceType::attachInterface<FIRReferenceTypeInterface>(*context);
    fir::StoreOp::attachInterface<FIRStoreActiveStore>(*context);
    hlfir::AssignOp::attachInterface<HLFIRAssignActiveStore>(*context);
    // Reverse-mode adjoints (the .td rules only cover forward tangents).
    fir::LoadOp::attachInterface<FIRLoadOpInterfaceReverse>(*context);
    fir::StoreOp::attachInterface<FIRStoreOpInterfaceReverse>(*context);
    hlfir::AssignOp::attachInterface<HLFIRAssignOpInterfaceReverse>(*context);
    hlfir::DeclareOp::attachInterface<HLFIRDeclareOpInterfaceReverse>(*context);
    registerInterfaces(context);
  });
}
