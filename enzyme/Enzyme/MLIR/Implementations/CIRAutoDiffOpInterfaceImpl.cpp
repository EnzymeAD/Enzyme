//===- CIRAutoDiffOpInterfaceImpl.cpp - Interface external model ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "Dialect/Ops.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/CIRDerivatives.inc"
} // namespace

namespace {
struct SwitchFlatBranchOpInterface
    : public BranchOpInterface::ExternalModel<SwitchFlatBranchOpInterface,
                                              cir::SwitchFlatOp> {

  SuccessorOperands getSuccessorOperands(Operation *op, unsigned index) const {
    auto sw = cast<cir::SwitchFlatOp>(op);
    assert(index < sw->getNumSuccessors() && "invalid successor index");
    if (index == 0)
      return SuccessorOperands(sw.getDefaultOperandsMutable());
    return SuccessorOperands(sw.getCaseOperandsMutable()[index - 1]);
  }

  std::optional<BlockArgument>
  getSuccessorBlockArgument(Operation *op, unsigned operandIndex) const {
    for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
      if (auto arg = mlir::detail::getBranchSuccessorArgument(
              getSuccessorOperands(op, i), operandIndex, op->getSuccessor(i)))
        return arg;
    }
    return std::nullopt;
  }
};

struct CIRStoreLike
    : public StoreLikeInterface::ExternalModel<CIRStoreLike, cir::StoreOp> {
  Value getStoredValue(Operation *op) const {
    return cast<cir::StoreOp>(op).getValue();
  }
  Value getStoredPointer(Operation *op) const {
    return cast<cir::StoreOp>(op).getAddr();
  }
};

struct CIRLoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          CIRLoadOpInterfaceReverse, cir::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<cir::LoadOp>(op);
    Value addr = loadOp.getAddr();
    auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType());
    if (!iface || iface.isMutable() || gutils->isConstantValue(loadOp) ||
        gutils->isConstantValue(addr))
      return success();
    // shadow[p] += v;
    Value gradient = gutils->diffe(loadOp, builder);
    Value addrGradient = gutils->popCache(caches.front(), builder);
    if (gutils->AtomicAdd) {
      // cir.atomic.fetch add accepts floating-point values and lowers to
      // atomicrmw fadd, like the llvm.load model's accumulation.
      cir::AtomicFetchOp::create(
          builder, loadOp.getLoc(), loadOp.getType(), addrGradient, gradient,
          cir::AtomicFetchKind::Add, cir::MemOrder::Relaxed,
          cir::SyncScopeKind::System);
      return success();
    }
    Value loaded = cir::LoadOp::create(builder, loadOp.getLoc(),
                                       loadOp.getType(), addrGradient);
    Value added = iface.createAddOp(builder, loadOp.getLoc(), loaded, gradient);
    cir::StoreOp::create(builder, loadOp.getLoc(), added, addrGradient);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<cir::LoadOp>(op);
    Value addr = loadOp.getAddr();
    if (!isa<AutoDiffTypeInterface>(loadOp.getType()) ||
        gutils->isConstantValue(loadOp) || gutils->isConstantValue(addr))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<cir::LoadOp>(op);
    Value addr = loadOp.getAddr();
    auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType());
    if (!iface)
      return op->emitError() << "could not compute the shadow of a load of a "
                                "type without autodiff semantics "
                             << *op;
    if (!iface.isMutable() || gutils->isConstantValue(loadOp))
      return success();
    if (gutils->isConstantValue(addr))
      return op->emitError()
             << "cannot load a non-constant value out of a constant address "
             << *op;
    Value addrShadow = gutils->invertPointerM(addr, builder);
    auto newLoad = cast<cir::LoadOp>(gutils->getNewFromOriginal(op));
    auto shadowLoad = cast<cir::LoadOp>(builder.clone(*newLoad));
    shadowLoad.getAddrMutable().assign(addrShadow);
    gutils->setInvertedPointer(loadOp.getResult(), shadowLoad.getResult());
    return success();
  }
};

struct CIRStoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          CIRStoreOpInterfaceReverse, cir::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<cir::StoreOp>(op);
    Value val = storeOp.getValue();
    Value addr = storeOp.getAddr();
    auto iface = cast<AutoDiffTypeInterface>(val.getType());

    if (!gutils->isConstantValue(addr)) {
      Value addrGradient = gutils->popCache(caches.front(), builder);
      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loaded = cir::LoadOp::create(builder, storeOp.getLoc(),
                                             val.getType(), addrGradient);
          gutils->addToDiffe(val, loaded, builder);
        }
        Value zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());
        cir::StoreOp::create(builder, storeOp.getLoc(), zero, addrGradient);
      }
    }
    // A store into memory the caller declared unneeded (enzyme_dupnoneed)
    // need not happen in the augmented forward pass either.
    if (gutils->primalStoreElidable(addr))
      gutils->erase(gutils->getNewFromOriginal(op));
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<cir::StoreOp>(op);
    Value addr = storeOp.getAddr();
    if (gutils->isConstantValue(addr))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<cir::StoreOp>(op);
    Value val = storeOp.getValue();
    Value addr = storeOp.getAddr();
    auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType());
    if (!iface)
      return op->emitError() << "could not compute the shadow of a store of a "
                                "type without autodiff semantics "
                             << *op;
    if (gutils->isConstantValue(addr) || !iface.isMutable())
      return success();
    Value addrShadow = gutils->invertPointerM(addr, builder);
    Value valShadow =
        gutils->isConstantValue(val)
            ? oputils::inactiveStoredValueShadow(op, *gutils, val, builder)
            : gutils->invertPointerM(val, builder);
    auto newOp = cast<cir::StoreOp>(gutils->getNewFromOriginal(op));
    auto shadowOp = cast<cir::StoreOp>(builder.clone(*newOp));
    shadowOp.getValueMutable().assign(valShadow);
    shadowOp.getAddrMutable().assign(addrShadow);
    return success();
  }
};

template <typename OpTy>
struct CIRPointerArithmeticReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          CIRPointerArithmeticReverse<OpTy>, OpTy> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    Value base = op->getOperand(0);
    if (gutils->isConstantValue(base))
      return success();
    Value baseShadow = gutils->invertPointerM(base, builder);
    Operation *shadowOp = builder.clone(*gutils->getNewFromOriginal(op));
    shadowOp->setOperand(0, baseShadow);
    gutils->setInvertedPointer(op->getResult(0), shadowOp->getResult(0));
    return success();
  }
};

// Any int or bool to and from conversion is drop as they need to be marked as
// constant.
// TODO: Support Complex
static bool isPointerCast(cir::CastKind kind) {
  return kind == cir::CastKind::bitcast ||
         kind == cir::CastKind::address_space ||
         kind == cir::CastKind::array_to_ptrdecay;
}

// Cast always mimic source so it is impossible to create active.
struct CIRCastOpActivity
    : public ActivityOpInterface::ExternalModel<CIRCastOpActivity,
                                                cir::CastOp> {
  bool isInactive(Operation *) const { return false; }
  bool isArgInactive(Operation *, size_t) const { return false; }
};

struct CIRCastOpForward
    : public AutoDiffOpInterface::ExternalModel<CIRCastOpForward, cir::CastOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto castOp = cast<cir::CastOp>(op);
    if (gutils->isConstantValue(castOp.getResult()))
      return success();
    cir::CastKind kind = castOp.getKind();
    if (kind != cir::CastKind::floating && !isPointerCast(kind))
      return op->emitError() << "unsupported active cir.cast kind '"
                             << cir::stringifyCastKind(kind) << "' " << *op;
    return mlir::enzyme::detail::memoryIdentityForwardHandler(
        op, builder, gutils, /*storedVals=*/{});
  }
};

struct CIRCastOpReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<CIRCastOpReverse,
                                                       cir::CastOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto castOp = cast<cir::CastOp>(op);
    Value src = castOp.getSrc();
    Value res = castOp.getResult();
    if (gutils->isConstantValue(res) || gutils->isConstantValue(src))
      return success();
    cir::CastKind kind = castOp.getKind();
    if (isPointerCast(kind))
      return success();
    if (kind != cir::CastKind::floating)
      return op->emitError() << "unsupported active cir.cast kind '"
                             << cir::stringifyCastKind(kind) << "' " << *op;
    Value dres = gutils->diffe(res, builder);
    Value dsrc = cir::CastOp::create(builder, op->getLoc(), src.getType(),
                                     cir::CastKind::floating, dres);
    gutils->addToDiffe(src, dsrc, builder);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto castOp = cast<cir::CastOp>(op);
    if (!isPointerCast(castOp.getKind()))
      return success();
    Value src = castOp.getSrc();
    if (gutils->isConstantValue(src))
      return success();
    auto newCast = cast<cir::CastOp>(gutils->getNewFromOriginal(op));
    auto shadowCast = cast<cir::CastOp>(builder.clone(*newCast));
    shadowCast.getSrcMutable().assign(gutils->invertPointerM(src, builder));
    gutils->setInvertedPointer(castOp.getResult(), shadowCast.getResult());
    return success();
  }
};

struct CIRCopyOpForward
    : public AutoDiffOpInterface::ExternalModel<CIRCopyOpForward, cir::CopyOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto copyOp = cast<cir::CopyOp>(op);
    Value dst = copyOp.getDst();
    Value src = copyOp.getSrc();
    if (gutils->isConstantValue(dst))
      return success();
    Value dstShadow = gutils->invertPointerM(dst, builder);
    auto newOp = cast<cir::CopyOp>(gutils->getNewFromOriginal(op));
    if (gutils->isConstantValue(src)) {
      Type elemTy = cast<cir::PointerType>(dst.getType()).getPointee();
      auto iface = dyn_cast<AutoDiffTypeInterface>(elemTy);
      if (!iface)
        return op->emitError() << "could not compute the tangent of a copy "
                                  "from an undifferentiated source of type "
                               << elemTy << " " << *op;
      Value zero = iface.createNullValue(builder, op->getLoc());
      cir::StoreOp::create(builder, op->getLoc(), zero, dstShadow);
    } else {
      Value srcShadow = gutils->invertPointerM(src, builder);
      auto shadowOp = cast<cir::CopyOp>(builder.clone(*newOp));
      shadowOp.getDstMutable().assign(dstShadow);
      shadowOp.getSrcMutable().assign(srcShadow);
    }
    if (gutils->primalStoreElidable(dst))
      gutils->erase(newOp);
    return success();
  }
};

struct CIRCopyOpReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<CIRCopyOpReverse,
                                                       cir::CopyOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto copyOp = cast<cir::CopyOp>(op);
    Value dst = copyOp.getDst();
    Value src = copyOp.getSrc();
    if (gutils->isConstantValue(dst))
      return success();
    Type elemTy = cast<cir::PointerType>(dst.getType()).getPointee();
    auto iface = dyn_cast<AutoDiffTypeInterface>(elemTy);
    if (!iface)
      return op->emitError() << "could not compute the adjoint of a copy of "
                             << elemTy << " " << *op;
    Location loc = op->getLoc();
    Value dstShadow = gutils->popCache(caches[0], builder);
    Value dstAdj = cir::LoadOp::create(builder, loc, elemTy, dstShadow);
    if (!gutils->isConstantValue(src)) {
      Value srcShadow = gutils->popCache(caches[1], builder);
      Value srcAdj = cir::LoadOp::create(builder, loc, elemTy, srcShadow);
      Value sum = iface.createAddOp(builder, loc, srcAdj, dstAdj);
      cir::StoreOp::create(builder, loc, sum, srcShadow);
    }
    Value zero = iface.createNullValue(builder, loc);
    cir::StoreOp::create(builder, loc, zero, dstShadow);
    if (gutils->primalStoreElidable(dst))
      gutils->erase(gutils->getNewFromOriginal(op));
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto copyOp = cast<cir::CopyOp>(op);
    Value dst = copyOp.getDst();
    Value src = copyOp.getSrc();
    if (gutils->isConstantValue(dst))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    SmallVector<Value> caches{gutils->initAndPushCache(
        gutils->invertPointerM(dst, cacheBuilder), cacheBuilder)};
    if (!gutils->isConstantValue(src))
      caches.push_back(gutils->initAndPushCache(
          gutils->invertPointerM(src, cacheBuilder), cacheBuilder));
    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto copyOp = cast<cir::CopyOp>(op);
    Value dst = copyOp.getDst();
    Value src = copyOp.getSrc();
    if (gutils->isConstantValue(dst) || gutils->isConstantValue(src))
      return success();
    auto newOp = cast<cir::CopyOp>(gutils->getNewFromOriginal(op));
    auto shadowOp = cast<cir::CopyOp>(builder.clone(*newOp));
    shadowOp.getDstMutable().assign(gutils->invertPointerM(dst, builder));
    shadowOp.getSrcMutable().assign(gutils->invertPointerM(src, builder));
    return success();
  }
};

} // namespace

class AutoDiffCIRFuncOpFunctionInterface
    : public AutoDiffFunctionInterface::ExternalModel<
          AutoDiffCIRFuncOpFunctionInterface, cir::FuncOp> {
public:
  static Type packedType(MLIRContext *ctx, TypeRange types) {
    SmallVector<Type> members(types.begin(), types.end());
    SmallVector<cir::RecordMemberKind> kinds(members.size(),
                                             cir::RecordMemberKind::Data);
    return cir::StructType::get(ctx, members, /*packed=*/false,
                                /*is_class=*/false, kinds);
  }

  void transformResultTypes(Operation *self,
                            SmallVectorImpl<Type> &resultTypes) const {
    if (resultTypes.size() <= 1)
      return;
    Type packed = packedType(self->getContext(), resultTypes);
    resultTypes.clear();
    resultTypes.push_back(packed);
  }
  void detachFromPrimalDefinition(Operation *self) const {
    // cir.func has comdat as a unit attr, not a symbol ref: nothing to
    // retarget.
  }
  Operation *createCall(Operation *self, OpBuilder &b, Location loc,
                        ValueRange args) const {
    auto fn = cast<cir::FuncOp>(self);
    Type res = fn.getFunctionType().getReturnTypes().empty()
                   ? Type()
                   : fn.getFunctionType().getReturnType();
    return cir::CallOp::create(b, loc, SymbolRefAttr::get(fn), res, args);
  }
  Operation *createReturn(Operation *, OpBuilder &b, Location loc,
                          ValueRange args) const {
    if (args.size() <= 1)
      return cir::ReturnOp::create(b, loc, args);
    Type packed = packedType(b.getContext(), args.getTypes());
    Value result =
        cir::ConstantOp::create(b, loc, packed, cir::ZeroAttr::get(packed));
    for (auto &&[i, v] : llvm::enumerate(args))
      result = cir::InsertMemberOp::create(b, loc, result, i, v);
    return cir::ReturnOp::create(b, loc, ValueRange{result});
  }
};

template <typename OpTy>
struct CIRLifetimeForward
    : public AutoDiffOpInterface::ExternalModel<CIRLifetimeForward<OpTy>,
                                                OpTy> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto ltOp = cast<OpTy>(op);
    if (gutils->isConstantValue(ltOp.getPtr()))
      return success();

    Value shadow = gutils->invertPointerM(ltOp.getPtr(), builder);
    auto newOp = cast<OpTy>(gutils->getNewFromOriginal(op));
    auto shadowOp = cast<OpTy>(builder.clone(*newOp));
    shadowOp.getPtrMutable().assign(shadow);
    return success();
  }
};

struct CIRReturnOpFunctionReturnInterface
    : public FunctionReturnOpInterface::ExternalModel<
          CIRReturnOpFunctionReturnInterface, cir::ReturnOp> {};

void mlir::enzyme::registerCIRDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, cir::CIRDialect *) {
    cir::SwitchFlatOp::attachInterface<SwitchFlatBranchOpInterface>(*context);
    cir::StoreOp::attachInterface<CIRStoreLike>(*context);
    cir::LoadOp::attachInterface<CIRLoadOpInterfaceReverse>(*context);
    cir::StoreOp::attachInterface<CIRStoreOpInterfaceReverse>(*context);
    cir::CastOp::attachInterface<CIRCastOpActivity>(*context);
    cir::CastOp::attachInterface<CIRCastOpForward>(*context);
    cir::CastOp::attachInterface<CIRCastOpReverse>(*context);
    cir::CopyOp::attachInterface<CIRCopyOpForward>(*context);
    cir::CopyOp::attachInterface<CIRCopyOpReverse>(*context);
    cir::LifetimeStartOp::attachInterface<
        CIRLifetimeForward<cir::LifetimeStartOp>>(*context);
    cir::LifetimeEndOp::attachInterface<CIRLifetimeForward<cir::LifetimeEndOp>>(
        *context);
    // Control never reaches cir.unreachable, so there is nothing to undo; the
    // reverse sweep still has to be told so for a terminator, which the
    // activity tables alone do not cover.
    cir::UnreachableOp::attachInterface<
        detail::NoopRevAutoDiffInterface<cir::UnreachableOp>>(*context);
    cir::PtrStrideOp::attachInterface<
        CIRPointerArithmeticReverse<cir::PtrStrideOp>>(*context);
    cir::GetMemberOp::attachInterface<
        CIRPointerArithmeticReverse<cir::GetMemberOp>>(*context);
    cir::GetElementOp::attachInterface<
        CIRPointerArithmeticReverse<cir::GetElementOp>>(*context);
    cir::BaseClassAddrOp::attachInterface<
        CIRPointerArithmeticReverse<cir::BaseClassAddrOp>>(*context);
    cir::DerivedClassAddrOp::attachInterface<
        CIRPointerArithmeticReverse<cir::DerivedClassAddrOp>>(*context);
    cir::ComplexRealPtrOp::attachInterface<
        CIRPointerArithmeticReverse<cir::ComplexRealPtrOp>>(*context);
    cir::ComplexImagPtrOp::attachInterface<
        CIRPointerArithmeticReverse<cir::ComplexImagPtrOp>>(*context);
    cir::PtrMaskOp::attachInterface<
        CIRPointerArithmeticReverse<cir::PtrMaskOp>>(*context);
    cir::GetRuntimeMemberOp::attachInterface<
        CIRPointerArithmeticReverse<cir::GetRuntimeMemberOp>>(*context);
    registerInterfaces(context);
    registerCIRAutoDiffTypeInterfaces(context);
    cir::FuncOp::attachInterface<AutoDiffCIRFuncOpFunctionInterface>(*context);
    cir::ReturnOp::attachInterface<CIRReturnOpFunctionReturnInterface>(
        *context);
  });
}
