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
#include "Interfaces/AutoDiffControlFlowInterface.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/MemTransferOpInterface.h"
#include "Interfaces/OffsetViewInterface.h"
#include "Interfaces/Utils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/StringSwitch.h"

#include <set>

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

struct CIRLoadLike
    : public LoadLikeInterface::ExternalModel<CIRLoadLike, cir::LoadOp> {
  Value getLoadedPointer(Operation *op) const {
    return cast<cir::LoadOp>(op).getAddr();
  }
};

struct CIRAllocaLike
    : public AllocaLikeInterface::ExternalModel<CIRAllocaLike, cir::AllocaOp> {
};

static bool returnsFreshAllocation(cir::CallOp call);
static Value forwardedTo(BranchOpInterface br, unsigned succIndex,
                         BlockArgument arg);

// Whether the shadow of `ptr` can be `ptr` itself, as far as the derivative
// checks: only with runtime activity on.
static bool shadowMayBePrimal(Value ptr, MGradientUtils *gutils) {
  if (!gutils->Logic.runtimeActivity)
    return false;
  SmallVector<Value> todo{ptr};
  DenseSet<Value> seen;
  while (!todo.empty()) {
    Value v = todo.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *def = v.getDefiningOp();
    if (!def) {
      auto arg = cast<BlockArgument>(v);
      Block *owner = arg.getOwner();
      if (owner->isEntryBlock())
        return true;
      for (auto it = owner->pred_begin(), e = owner->pred_end(); it != e;
           ++it) {
        auto br = dyn_cast<BranchOpInterface>((*it)->getTerminator());
        Value in = br ? forwardedTo(br, it.getSuccessorIndex(), arg) : Value();
        if (!in)
          return true;
        todo.push_back(in);
      }
      continue;
    }
    if (isa<cir::AllocaOp>(def) || def->hasTrait<OpTrait::ConstantLike>())
      continue;
    if (auto call = dyn_cast<cir::CallOp>(def)) {
      if (!returnsFreshAllocation(call))
        return true;
      continue;
    }
    if (auto cast = dyn_cast<cir::CastOp>(def);
        cast && isa<cir::PointerType>(cast.getSrc().getType())) {
      todo.push_back(cast.getSrc());
      continue;
    }
    if (auto stride = dyn_cast<cir::PtrStrideOp>(def)) {
      todo.push_back(stride.getBase());
      continue;
    }
    if (auto member = dyn_cast<cir::GetMemberOp>(def)) {
      todo.push_back(member.getAddr());
      continue;
    }
    if (auto element = dyn_cast<cir::GetElementOp>(def)) {
      todo.push_back(element.getBase());
      continue;
    }
    if (auto select = dyn_cast<cir::SelectOp>(def)) {
      todo.push_back(select.getTrueValue());
      todo.push_back(select.getFalseValue());
      continue;
    }
    return true;
  }
  return false;
}

// Memory holding a `ty`, which nothing reads, served as a garbage can for any write
static Value getScratch(OpBuilder &builder, Location loc, Type ty) {
  auto module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto ptrTy = cir::PointerType::get(ty);
  constexpr StringLiteral prefix = "__enzyme_cir_scratch";
  unsigned count = 0;
  for (auto global : module.getOps<cir::GlobalOp>()) {
    if (!global.getSymName().starts_with(prefix))
      continue;
    if (global.getSymType() == ty)
      return cir::GetGlobalOp::create(builder, loc, ptrTy, global.getSymName());
    ++count;
  }
  OpBuilder moduleBuilder = OpBuilder::atBlockBegin(module.getBody());
  std::string name = (prefix + Twine(count)).str();
  auto global = cir::GlobalOp::create(moduleBuilder, loc, name, ty,
                                      /*isConstant=*/false, /*addrSpace=*/{},
                                      cir::GlobalLinkageKind::InternalLinkage);
  global.setInitialValueAttr(cast<AutoDiffTypeInterface>(ty).createNullAttr());
  global.setPrivate();
  return cir::GetGlobalOp::create(builder, loc, ptrTy, name);
}

static void cachePrimalAddress(Value addr, OpBuilder &cacheBuilder,
                               MGradientUtilsReverse *gutils,
                               SmallVectorImpl<Value> &caches) {
  if (shadowMayBePrimal(addr, gutils))
    caches.push_back(gutils->initAndPushCache(gutils->getNewFromOriginal(addr),
                                              cacheBuilder));
}

// Whether `shadow` is `primal`, at run time, or null when it cannot be.
static Value shadowIsPrimal(OpBuilder &builder, Location loc, Value shadow,
                            Value primal) {
  if (!primal)
    return Value();
  if (primal.getType() != shadow.getType())
    primal = cir::CastOp::create(builder, loc, shadow.getType(),
                                 cir::CastKind::bitcast, primal);
  return cir::CmpOp::create(builder, loc, cir::CmpOpKind::eq, shadow, primal);
}

// Where to write the adjoint `shadow` holds: `shadow`, or scratch memory
// when it is the primal.
static Value adjointAddress(OpBuilder &builder, Location loc, Value same,
                            Value shadow) {
  if (!same)
    return shadow;
  Type ty = cast<cir::PointerType>(shadow.getType()).getPointee();
  return cir::SelectOp::create(builder, loc, same, getScratch(builder, loc, ty),
                               shadow);
}

// The adjoint `loaded` is read from a shadow: zero when that is the primal.
static Value adjointValue(OpBuilder &builder, Location loc, Value same,
                          Value loaded) {
  if (!same)
    return loaded;
  Value zero = cast<AutoDiffTypeInterface>(loaded.getType())
                   .createNullValue(builder, loc);
  return cir::SelectOp::create(builder, loc, same, zero, loaded);
}

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
    gutils->zeroDiffe(loadOp, builder);
    Value addrGradient = gutils->popCache(caches.front(), builder);
    // Do a write, if it is inactive, go to trash can.
    if (caches.size() > 1)
      addrGradient =
          adjointAddress(builder, loadOp.getLoc(),
                         shadowIsPrimal(builder, loadOp.getLoc(), addrGradient,
                                        gutils->popCache(caches[1], builder)),
                         addrGradient);
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
    SmallVector<Value> caches{gutils->initAndPushCache(
        gutils->invertPointerM(addr, cacheBuilder), cacheBuilder)};
    if (!cast<AutoDiffTypeInterface>(loadOp.getType()).isMutable())
      cachePrimalAddress(addr, cacheBuilder, gutils, caches);
    return caches;
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
        Location loc = storeOp.getLoc();
        Value same = caches.size() > 1
                         ? shadowIsPrimal(builder, loc, addrGradient,
                                          gutils->popCache(caches[1], builder))
                         : Value();
        if (!gutils->isConstantValue(val)) {
          Value loaded =
              cir::LoadOp::create(builder, loc, val.getType(), addrGradient);
          gutils->addToDiffe(val, adjointValue(builder, loc, same, loaded),
                             builder);
        }
        Value zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());
        cir::StoreOp::create(builder, loc, zero,
                             adjointAddress(builder, loc, same, addrGradient));
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
    SmallVector<Value> caches{gutils->initAndPushCache(
        gutils->invertPointerM(addr, cacheBuilder), cacheBuilder)};
    auto iface = dyn_cast<AutoDiffTypeInterface>(storeOp.getValue().getType());
    if (iface && !iface.isMutable())
      cachePrimalAddress(addr, cacheBuilder, gutils, caches);
    return caches;
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
    // A shadow store through a shadow that is the primal would overwrite the
    // primal.
    if (shadowMayBePrimal(addr, gutils))
      addrShadow =
          adjointAddress(builder, op->getLoc(),
                         shadowIsPrimal(builder, op->getLoc(), addrShadow,
                                        gutils->getNewFromOriginal(addr)),
                         addrShadow);
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

static LogicalResult createGlobalShadow(cir::GetGlobalOp op, OpBuilder &builder,
                                        MGradientUtils *gutils) {
  auto global =
      dyn_cast_or_null<cir::GlobalOp>(oputils::getGlobalForAddress(op));
  if (!global)
    return op.emitError() << "could not find the global of " << *op;
  auto shadowName = global->getAttrOfType<FlatSymbolRefAttr>("enzyme_shadow");
  if (!shadowName) {
    // As LLVM does, create a zero shadow next to a global this module defines
    // or that cannot change; only a global defined elsewhere must name one.
    if (!global.getConstant() && !cir::isLocalLinkage(global.getLinkage()) &&
        !(cir::isExternalLinkage(global.getLinkage()) &&
          global.hasInitializer()))
      return op.emitError() << "global @" << global.getSymName()
                            << " is active and defined outside this module, "
                               "so it needs a shadow global named by an "
                               "enzyme_shadow attribute";
    OpBuilder globalBuilder(op.getContext());
    auto shadow = cir::GlobalOp::create(
        globalBuilder, global.getLoc(), global.getSymName().str() + "_shadow",
        global.getSymType(), global.getConstant(), global.getAddrSpaceAttr(),
        global.getLinkage());
    shadow.setSymVisibilityAttr(global.getSymVisibilityAttr());
    shadow.setGlobalVisibility(global.getGlobalVisibility());
    shadow.setTlsModelAttr(global.getTlsModelAttr());
    shadow.setDsoLocal(global.getDsoLocal());
    shadow.setAlignmentAttr(global.getAlignmentAttr());
    shadow.setInitialValueAttr(cir::CIRBaseBuilderTy(*op.getContext())
                                   .getZeroInitAttr(global.getSymType()));
    // Renames the shadow if the name is taken.
    SymbolTable(SymbolTable::getNearestSymbolTable(global))
        .insert(shadow, std::next(Block::iterator(global)));
    shadowName = FlatSymbolRefAttr::get(shadow.getSymNameAttr());
    global->setAttr("enzyme_shadow", shadowName);
  }
  auto shadow =
      SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(op, shadowName);
  if (!shadow || shadow.getSymType() != global.getSymType())
    return op.emitError() << "enzyme_shadow of @" << global.getSymName()
                          << " must name a cir.global of type "
                          << global.getSymType();
  if (gutils->width != 1)
    return op.emitError() << "vector mode is not supported for global @"
                          << global.getSymName();
  auto shadowOp = cast<cir::GetGlobalOp>(
      builder.clone(*gutils->getNewFromOriginal(op.getOperation())));
  shadowOp.setNameAttr(shadowName);
  gutils->setInvertedPointer(op.getAddr(), shadowOp.getAddr());
  return success();
}

struct CIRGetGlobalOpForward
    : public AutoDiffOpInterface::ExternalModel<CIRGetGlobalOpForward,
                                                cir::GetGlobalOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    return createGlobalShadow(cast<cir::GetGlobalOp>(op), builder, gutils);
  }
};

struct CIRGetGlobalOpReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<CIRGetGlobalOpReverse,
                                                       cir::GetGlobalOp> {
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
    return createGlobalShadow(cast<cir::GetGlobalOp>(op), builder, gutils);
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
    gutils->zeroDiffe(res, builder);
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

static FailureOr<Type> uniformFloatType(Type t) {
  if (isa<cir::FPTypeInterface>(t))
    return t;
  if (isa<cir::IntType, cir::BoolType, cir::PointerType>(t))
    return Type();
  if (auto a = dyn_cast<cir::ArrayType>(t))
    return uniformFloatType(a.getElementType());
  if (auto v = dyn_cast<cir::VectorType>(t))
    return uniformFloatType(v.getElementType());
  if (auto c = dyn_cast<cir::ComplexType>(t))
    return uniformFloatType(c.getElementType());
  if (auto s = dyn_cast<cir::StructType>(t)) {
    std::optional<Type> acc;
    for (Type member : s.getMembers()) {
      FailureOr<Type> f = uniformFloatType(member);
      if (failed(f) || (acc && *acc != *f))
        return failure();
      acc = *f;
    }
    return acc.value_or(Type());
  }
  return failure();
}

static bool copiesOnlyFloatFreeData(Type t) {
  FailureOr<Type> f = uniformFloatType(t);
  return succeeded(f) && !*f;
}

static Type memcpyCopiedType(cir::MemCpyOp memcpy) {
  for (Value ptr : {memcpy.getDst(), memcpy.getSrc()}) {
    if (auto castOp = ptr.getDefiningOp<cir::CastOp>();
        castOp && castOp.getKind() == cir::CastKind::bitcast)
      ptr = castOp.getSrc();
    Type pointee = cast<cir::PointerType>(ptr.getType()).getPointee();
    if (!isa<cir::VoidType>(pointee))
      return pointee;
  }
  return Type();
}

static FailureOr<Type> memcpyFloatType(cir::MemCpyOp memcpy) {
  Type copied = memcpyCopiedType(memcpy);
  if (!copied)
    return failure();
  return uniformFloatType(copied);
}

static void zeroBytes(OpBuilder &builder, Location loc, Value dst, Value len) {
  auto u8 = cir::IntType::get(builder.getContext(), 8, /*isSigned=*/false);
  Value zero = cir::ConstantOp::create(builder, loc, u8,
                                       cir::IntAttr::get(u8, APInt(8, 0)));
  cir::MemSetOp::create(builder, loc, dst, /*alignment=*/nullptr, zero, len);
}

// dsrc[i] += ddst[i]; ddst[i] = 0 for each floatTy element in len bytes.
static void accumulateAndZero(OpBuilder &builder, Location loc, Type floatTy,
                              Value dstShadow, Value srcShadow, Value len) {
  auto ptrTy = cir::PointerType::get(floatTy);
  Value dst = cir::CastOp::create(builder, loc, ptrTy, cir::CastKind::bitcast,
                                  dstShadow);
  Value src = cir::CastOp::create(builder, loc, ptrTy, cir::CastKind::bitcast,
                                  srcShadow);
  auto lenTy = cast<cir::IntType>(len.getType());
  auto constLen = [&](OpBuilder &b, uint64_t v) -> Value {
    return cir::ConstantOp::create(
        b, loc, lenTy, cir::IntAttr::get(lenTy, APInt(lenTy.getWidth(), v)));
  };
  DataLayout layout =
      DataLayout::closest(dstShadow.getParentBlock()->getParentOp());
  Value count = cir::DivOp::create(
      builder, loc, len, constLen(builder, layout.getTypeSize(floatTy)));

  // The builder may be filling a region whose op is not built yet; the
  // shadow is defined in the function already.
  Operation *fn = dstShadow.getParentBlock()->getParentOp();
  if (!isa<FunctionOpInterface>(fn))
    fn = fn->getParentOfType<FunctionOpInterface>();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(&fn->getRegion(0).front());
  Value iv = cir::AllocaOp::create(
      entryBuilder, loc, cir::PointerType::get(lenTy), "memcpy.idx",
      entryBuilder.getI64IntegerAttr(layout.getTypeABIAlignment(lenTy)));
  cir::StoreOp::create(builder, loc, constLen(builder, 0), iv);

  auto iface = cast<AutoDiffTypeInterface>(floatTy);
  cir::ForOp::create(
      builder, loc,
      [&](OpBuilder &b, Location l) {
        Value i = cir::LoadOp::create(b, l, lenTy, iv);
        Value more = cir::CmpOp::create(b, l, cir::CmpOpKind::lt, i, count);
        cir::ConditionOp::create(b, l, more);
      },
      [&](OpBuilder &b, Location l) {
        Value i = cir::LoadOp::create(b, l, lenTy, iv);
        Value d = cir::PtrStrideOp::create(b, l, ptrTy, dst, i);
        Value s = cir::PtrStrideOp::create(b, l, ptrTy, src, i);
        Value dAdj = cir::LoadOp::create(b, l, floatTy, d);
        Value sAdj = cir::LoadOp::create(b, l, floatTy, s);
        cir::StoreOp::create(b, l, iface.createAddOp(b, l, sAdj, dAdj), s);
        cir::StoreOp::create(b, l, iface.createNullValue(b, l), d);
        cir::YieldOp::create(b, l);
      },
      [&](OpBuilder &b, Location l) {
        Value i = cir::LoadOp::create(b, l, lenTy, iv);
        cir::StoreOp::create(b, l, cir::AddOp::create(b, l, i, constLen(b, 1)),
                             iv);
        cir::YieldOp::create(b, l);
      });
}

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
    bool srcActive = !gutils->isConstantValue(src);
    Value dstShadow = gutils->popCache(caches[0], builder);
    Value srcShadow =
        srcActive ? gutils->popCache(caches[1], builder) : Value();
    unsigned next = srcActive ? 2 : 1;
    Value dstSame, srcSame;
    if (shadowMayBePrimal(dst, gutils))
      dstSame = shadowIsPrimal(builder, loc, dstShadow,
                               gutils->popCache(caches[next++], builder));
    if (srcActive && shadowMayBePrimal(src, gutils))
      srcSame = shadowIsPrimal(builder, loc, srcShadow,
                               gutils->popCache(caches[next++], builder));
    Value dstAdj =
        adjointValue(builder, loc, dstSame,
                     cir::LoadOp::create(builder, loc, elemTy, dstShadow));
    if (srcActive) {
      Value srcAddr = adjointAddress(builder, loc, srcSame, srcShadow);
      Value srcAdj = cir::LoadOp::create(builder, loc, elemTy, srcAddr);
      Value sum = iface.createAddOp(builder, loc, srcAdj, dstAdj);
      cir::StoreOp::create(builder, loc, sum, srcAddr);
    }
    Value zero = iface.createNullValue(builder, loc);
    cir::StoreOp::create(builder, loc, zero,
                         adjointAddress(builder, loc, dstSame, dstShadow));
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
    cachePrimalAddress(dst, cacheBuilder, gutils, caches);
    if (!gutils->isConstantValue(src))
      cachePrimalAddress(src, cacheBuilder, gutils, caches);
    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto copyOp = cast<cir::CopyOp>(op);
    Value dst = copyOp.getDst();
    Value src = copyOp.getSrc();
    if (gutils->isConstantValue(dst) || gutils->isConstantValue(src))
      return success();
    // Copying float shadows now would overwrite the adjoint seed of dst.
    if (!copiesOnlyFloatFreeData(
            cast<cir::PointerType>(dst.getType()).getPointee()))
      return success();
    auto newOp = cast<cir::CopyOp>(gutils->getNewFromOriginal(op));
    auto shadowOp = cast<cir::CopyOp>(builder.clone(*newOp));
    shadowOp.getDstMutable().assign(gutils->invertPointerM(dst, builder));
    shadowOp.getSrcMutable().assign(gutils->invertPointerM(src, builder));
    return success();
  }
};

struct CIRMemCpyOpForward
    : public AutoDiffOpInterface::ExternalModel<CIRMemCpyOpForward,
                                                cir::MemCpyOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto memcpy = cast<cir::MemCpyOp>(op);
    if (gutils->isConstantValue(memcpy.getDst()))
      return success();
    Value dstShadow = gutils->invertPointerM(memcpy.getDst(), builder);
    auto newOp = cast<cir::MemCpyOp>(gutils->getNewFromOriginal(op));
    if (!gutils->isConstantValue(memcpy.getSrc())) {
      auto shadowOp = cast<cir::MemCpyOp>(builder.clone(*newOp));
      shadowOp.getDstMutable().assign(dstShadow);
      shadowOp.getSrcMutable().assign(
          gutils->invertPointerM(memcpy.getSrc(), builder));
    } else {
      FailureOr<Type> floatTy = memcpyFloatType(memcpy);
      if (failed(floatTy))
        return op->emitError()
               << "could not compute the tangent of a memcpy from an "
                  "undifferentiated source without knowing what it copies "
               << *op;
      if (*floatTy) {
        zeroBytes(builder, op->getLoc(), dstShadow, newOp.getLen());
      } else {
        auto shadowOp = cast<cir::MemCpyOp>(builder.clone(*newOp));
        shadowOp.getDstMutable().assign(dstShadow);
      }
    }
    if (gutils->primalStoreElidable(memcpy.getDst()))
      gutils->erase(newOp);
    return success();
  }
};

struct CIRMemCpyOpReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<CIRMemCpyOpReverse,
                                                       cir::MemCpyOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto memcpy = cast<cir::MemCpyOp>(op);
    if (gutils->isConstantValue(memcpy.getDst()))
      return success();
    FailureOr<Type> floatTy = memcpyFloatType(memcpy);
    if (failed(floatTy))
      return op->emitError() << "could not compute the adjoint of a memcpy "
                                "without knowing what it copies "
                             << *op;
    if (*floatTy) {
      Location loc = op->getLoc();
      bool srcActive = !gutils->isConstantValue(memcpy.getSrc());
      Value dstShadow = gutils->popCache(caches[0], builder);
      Value len = gutils->popCache(caches[1], builder);
      Value srcShadow =
          srcActive ? gutils->popCache(caches[2], builder) : Value();
      unsigned next = srcActive ? 3 : 2;
      Value dstSame, srcSame;
      if (shadowMayBePrimal(memcpy.getDst(), gutils))
        dstSame = shadowIsPrimal(builder, loc, dstShadow,
                                 gutils->popCache(caches[next++], builder));
      if (srcActive && shadowMayBePrimal(memcpy.getSrc(), gutils))
        srcSame = shadowIsPrimal(builder, loc, srcShadow,
                                 gutils->popCache(caches[next++], builder));
      // Nothing is done when the destination's shadow is its primal, and
      // nothing is added up when the source's is.
      auto adjoint = [&](OpBuilder &b) {
        if (!srcActive) {
          zeroBytes(b, loc, dstShadow, len);
          return;
        }
        if (!srcSame) {
          accumulateAndZero(b, loc, *floatTy, dstShadow, srcShadow, len);
          return;
        }
        cir::IfOp::create(
            b, loc, srcSame, /*withElseRegion=*/true,
            [&](OpBuilder &tb, Location l) {
              zeroBytes(tb, l, dstShadow, len);
              cir::YieldOp::create(tb, l);
            },
            [&](OpBuilder &eb, Location l) {
              accumulateAndZero(eb, l, *floatTy, dstShadow, srcShadow, len);
              cir::YieldOp::create(eb, l);
            });
      };
      if (!dstSame)
        adjoint(builder);
      else
        cir::IfOp::create(
            builder, loc, dstSame, /*withElseRegion=*/true,
            [&](OpBuilder &tb, Location l) { cir::YieldOp::create(tb, l); },
            [&](OpBuilder &eb, Location l) {
              adjoint(eb);
              cir::YieldOp::create(eb, l);
            });
    }
    if (gutils->primalStoreElidable(memcpy.getDst()))
      gutils->erase(gutils->getNewFromOriginal(op));
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto memcpy = cast<cir::MemCpyOp>(op);
    if (gutils->isConstantValue(memcpy.getDst()))
      return {};
    FailureOr<Type> floatTy = memcpyFloatType(memcpy);
    if (failed(floatTy) || !*floatTy)
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    SmallVector<Value> caches{
        gutils->initAndPushCache(
            gutils->invertPointerM(memcpy.getDst(), cacheBuilder),
            cacheBuilder),
        gutils->initAndPushCache(gutils->getNewFromOriginal(memcpy.getLen()),
                                 cacheBuilder)};
    if (!gutils->isConstantValue(memcpy.getSrc()))
      caches.push_back(gutils->initAndPushCache(
          gutils->invertPointerM(memcpy.getSrc(), cacheBuilder), cacheBuilder));
    cachePrimalAddress(memcpy.getDst(), cacheBuilder, gutils, caches);
    if (!gutils->isConstantValue(memcpy.getSrc()))
      cachePrimalAddress(memcpy.getSrc(), cacheBuilder, gutils, caches);
    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto memcpy = cast<cir::MemCpyOp>(op);
    if (gutils->isConstantValue(memcpy.getDst()))
      return success();
    // Float shadows hold adjoint seeds now; leave them for the reverse sweep.
    FailureOr<Type> floatTy = memcpyFloatType(memcpy);
    if (failed(floatTy) || *floatTy)
      return success();
    auto newOp = cast<cir::MemCpyOp>(gutils->getNewFromOriginal(op));
    Value dstShadow = gutils->invertPointerM(memcpy.getDst(), builder);
    Value srcShadow = gutils->isConstantValue(memcpy.getSrc())
                          ? Value()
                          : gutils->invertPointerM(memcpy.getSrc(), builder);
    auto copyShadows = [&](OpBuilder &b) {
      auto shadowOp = cast<cir::MemCpyOp>(b.clone(*newOp));
      shadowOp.getDstMutable().assign(dstShadow);
      if (srcShadow)
        shadowOp.getSrcMutable().assign(srcShadow);
    };
    // Copying shadows over a destination whose shadow is its primal would
    // overwrite the primal.
    if (!shadowMayBePrimal(memcpy.getDst(), gutils)) {
      copyShadows(builder);
      return success();
    }
    Value same = shadowIsPrimal(builder, op->getLoc(), dstShadow,
                                gutils->getNewFromOriginal(memcpy.getDst()));
    cir::IfOp::create(
        builder, op->getLoc(), same, /*withElseRegion=*/true,
        [&](OpBuilder &tb, Location l) { cir::YieldOp::create(tb, l); },
        [&](OpBuilder &eb, Location l) {
          copyShadows(eb);
          cir::YieldOp::create(eb, l);
        });
    return success();
  }
};

// The C and C++ allocators have no body to differentiate. An active
// allocation gets a shadow allocation of the same size, zeroed, which lives
// as long as the primal does and is freed the way the primal is.
enum class AllocatorFn { None, Alloc, ZeroedAlloc, Realloc, Free };

static AllocatorFn getAllocatorFn(cir::CallOp call) {
  FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee)
    return AllocatorFn::None;
  auto fn = SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(call, callee);
  if (!fn || !fn.isDeclaration())
    return AllocatorFn::None;
  return llvm::StringSwitch<AllocatorFn>(fn.getSymName())
      .Cases({"malloc", "_Znwm", "_Znam", "_Znwj", "_Znaj"}, AllocatorFn::Alloc)
      .Cases({"_ZnwmRKSt9nothrow_t", "_ZnamRKSt9nothrow_t",
              "_ZnwjRKSt9nothrow_t", "_ZnajRKSt9nothrow_t"},
             AllocatorFn::Alloc)
      .Cases({"_ZnwmSt11align_val_t", "_ZnamSt11align_val_t",
              "_ZnwjSt11align_val_t", "_ZnajSt11align_val_t"},
             AllocatorFn::Alloc)
      .Cases({"_ZnwmSt11align_val_tRKSt9nothrow_t",
              "_ZnamSt11align_val_tRKSt9nothrow_t",
              "_ZnwjSt11align_val_tRKSt9nothrow_t",
              "_ZnajSt11align_val_tRKSt9nothrow_t"},
             AllocatorFn::Alloc)
      .Case("calloc", AllocatorFn::ZeroedAlloc)
      .Case("realloc", AllocatorFn::Realloc)
      .Cases({"free", "_ZdlPv", "_ZdaPv", "_ZdlPvm", "_ZdaPvm", "_ZdlPvj",
              "_ZdaPvj"},
             AllocatorFn::Free)
      .Cases({"_ZdlPvSt11align_val_t", "_ZdaPvSt11align_val_t",
              "_ZdlPvmSt11align_val_t", "_ZdaPvmSt11align_val_t",
              "_ZdlPvjSt11align_val_t", "_ZdaPvjSt11align_val_t"},
             AllocatorFn::Free)
      .Cases({"_ZdlPvRKSt9nothrow_t", "_ZdaPvRKSt9nothrow_t",
              "_ZdlPvSt11align_val_tRKSt9nothrow_t",
              "_ZdaPvSt11align_val_tRKSt9nothrow_t"},
             AllocatorFn::Free)
      .Default(AllocatorFn::None);
}

static bool isAllocation(AllocatorFn fn) {
  return fn == AllocatorFn::Alloc || fn == AllocatorFn::ZeroedAlloc;
}

static bool returnsFreshAllocation(cir::CallOp call) {
  AllocatorFn fn = getAllocatorFn(call);
  return isAllocation(fn) || fn == AllocatorFn::Realloc;
}

// Allocates the shadow of primal.
static Value createShadowAllocation(cir::CallOp primal, AllocatorFn fn,
                                    OpBuilder &builder) {
  Location loc = primal.getLoc();
  Value shadow = builder.clone(*primal)->getResult(0);
  // calloc's memory is zero already.
  if (fn == AllocatorFn::Alloc) {
    MLIRContext *ctx = builder.getContext();
    auto voidPtr = cir::PointerType::get(cir::VoidType::get(ctx));
    Value dst = shadow.getType() == voidPtr
                    ? shadow
                    : cir::CastOp::create(builder, loc, voidPtr,
                                          cir::CastKind::bitcast, shadow)
                          .getResult();
    auto u8 = cir::IntType::get(ctx, 8, /*isSigned=*/false);
    Value zero =
        cir::ConstantOp::create(builder, loc, cir::IntAttr::get(u8, 0));
    cir::MemSetOp::create(builder, loc, dst, /*alignment=*/nullptr, zero,
                          primal.getArgOperands().front());
  }
  return shadow;
}

static bool isBitcast(cir::CastOp cast) {
  return cast && cast.getKind() == cir::CastKind::bitcast;
}

static bool isNullPointer(Value v) {
  auto c = v.getDefiningOp<cir::ConstantOp>();
  auto attr = c ? dyn_cast<cir::ConstPtrAttr>(c.getValue()) : nullptr;
  return attr && attr.isNullValue();
}

static Value stripBitcasts(Value v) {
  for (auto cast = v.getDefiningOp<cir::CastOp>(); isBitcast(cast);
       cast = v.getDefiningOp<cir::CastOp>())
    v = cast.getSrc();
  return v;
}

// Whether v is one of held, or a bitcast of one.
static bool isHeld(Value v, const DenseSet<Value> &held) {
  return held.contains(stripBitcasts(v));
}

// What `br` passes to `arg` when it branches to its successor `succIndex`, or
// null when `br` produces that operand itself.
static Value forwardedTo(BranchOpInterface br, unsigned succIndex,
                         BlockArgument arg) {
  SuccessorOperands operands = br.getSuccessorOperands(succIndex);
  unsigned produced = operands.getProducedOperandCount();
  if (arg.getArgNumber() < produced)
    return Value();
  return operands.getForwardedOperands()[arg.getArgNumber() - produced];
}

// C++ requires `delete` be called non null. Find the null path and return
// directly.
static Block *nullSuccessor(Block *block, const DenseSet<Value> &held) {
  auto br = dyn_cast<cir::BrCondOp>(block->getTerminator());
  if (!br)
    return nullptr;
  auto cmp = br.getCond().getDefiningOp<cir::CmpOp>();
  if (!cmp || (cmp.getKind() != cir::CmpOpKind::eq &&
               cmp.getKind() != cir::CmpOpKind::ne))
    return nullptr;
  if (!((isHeld(cmp.getLhs(), held) && isNullPointer(cmp.getRhs())) ||
        (isHeld(cmp.getRhs(), held) && isNullPointer(cmp.getLhs()))))
    return nullptr;
  return cmp.getKind() == cir::CmpOpKind::eq ? br.getDestTrue()
                                             : br.getDestFalse();
}

// Check whether a alloc is actually freed in all BB used it.
// This go through all BB contains the transformation of allocated SSA value
// (from cast or itself). Returns every free found, or none if some path does
// not free it.
static SmallVector<cir::CallOp> getGuaranteedFrees(cir::CallOp alloc) {
  // The first free in `block` of a pointer `held` there, up to and
  // including `last`.
  auto freeIn = [](Block *block, const DenseSet<Value> &held,
                   Operation *last = nullptr) {
    for (Operation &op : *block) {
      if (auto call = dyn_cast<cir::CallOp>(op)) {
        AllocatorFn fn = getAllocatorFn(call);
        bool frees =
            fn == AllocatorFn::Realloc ||
            (fn == AllocatorFn::Free &&
             llvm::all_of(call.getArgOperands().drop_front(), [](Value arg) {
               return arg.getDefiningOp<cir::ConstantOp>();
             }));
        if (frees && isHeld(call.getArgOperands().front(), held))
          return call;
      }
      if (&op == last)
        break;
    }
    return cir::CallOp();
  };

  // Bounds the paths looked at; past it the shadow is just not freed.
  constexpr unsigned maxStates = 256;
  Block *start = alloc->getBlock();
  using State = std::pair<Block *, DenseSet<Value>>;
  SmallVector<State> todo;
  std::set<std::pair<Block *, std::vector<void *>>> visited;
  SmallVector<cir::CallOp> found;
  auto add = [&](cir::CallOp free) {
    if (!llvm::is_contained(found, free))
      found.push_back(free);
  };
  DenseSet<Value> init{alloc->getResult(0)};
  if (cir::CallOp free = freeIn(start, init))
    return {free};
  todo.push_back({start, init});
  while (!todo.empty()) {
    auto [block, held] = todo.pop_back_val();
    if (block != start) {
      std::vector<void *> key;
      for (Value v : held)
        key.push_back(v.getAsOpaquePointer());
      llvm::sort(key);
      if (!visited.insert({block, std::move(key)}).second)
        continue;
      if (visited.size() > maxStates)
        return {};
      if (cir::CallOp free = freeIn(block, held)) {
        add(free);
        continue;
      }
    }
    if (block->getNumSuccessors() == 0)
      return {};
    auto br = dyn_cast<BranchOpInterface>(block->getTerminator());
    if (!br)
      return {};
    Block *ifNull = nullSuccessor(block, held);
    for (unsigned i = 0, e = block->getNumSuccessors(); i < e; ++i) {
      // Skip nullptr check branch
      Block *succ = block->getSuccessor(i);
      if (succ == ifNull)
        continue;
      DenseSet<Value> next = held;
      for (BlockArgument arg : succ->getArguments()) {
        Value in = forwardedTo(br, i, arg);
        if (in && isHeld(in, held))
          next.insert(arg);
        else
          next.erase(arg);
      }
      // Back where it allocates: fine if it frees the pointer first, as
      // `b = realloc(b, n)` in a loop does.
      if (succ == start) {
        cir::CallOp free = freeIn(start, next, alloc);
        if (!free)
          return {};
        add(free);
        continue;
      }
      todo.push_back({succ, std::move(next)});
    }
  }
  return found;
}

// getGuaranteedFrees for every allocation of the function being
// differentiated, found once for all.
static const MGradientUtilsReverse::GuaranteedFrees &
guaranteedFrees(MGradientUtilsReverse *gutils) {
  if (!gutils->guaranteedFrees) {
    auto &found = gutils->guaranteedFrees.emplace();
    gutils->oldFunc.walk([&](cir::CallOp call) {
      if (!returnsFreshAllocation(call))
        return;
      SmallVector<cir::CallOp> frees = getGuaranteedFrees(call);
      if (frees.empty())
        return;
      found.byAllocation[call] = frees.front();
      for (cir::CallOp free : frees)
        found.frees.insert(free);
    });
  }
  return *gutils->guaranteedFrees;
}

static cir::CallOp getGuaranteedFree(cir::CallOp alloc,
                                     MGradientUtilsReverse *gutils) {
  return cast_if_present<cir::CallOp>(
      guaranteedFrees(gutils).byAllocation.lookup(alloc));
}

static cir::FuncOp getLibraryFunction(Operation *from, StringRef name,
                                      cir::FuncType type) {
  auto module = from->getParentOfType<ModuleOp>();
  if (auto fn = module.lookupSymbol<cir::FuncOp>(name))
    return fn;
  OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
  auto fn = cir::FuncOp::create(builder, from->getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

static cir::FuncOp getFreeFunction(Operation *from) {
  MLIRContext *ctx = from->getContext();
  return getLibraryFunction(
      from, "free",
      cir::FuncType::get({cir::PointerType::get(cir::VoidType::get(ctx))},
                         cir::VoidType::get(ctx)));
}

static cir::FuncOp getMallocFunction(Operation *from) {
  MLIRContext *ctx = from->getContext();
  return getLibraryFunction(
      from, "malloc",
      cir::FuncType::get({cir::IntType::get(ctx, 64, /*isSigned=*/false)},
                         cir::PointerType::get(cir::VoidType::get(ctx))));
}

static Value bitcastTo(OpBuilder &builder, Location loc, Value v, Type ty) {
  if (v.getType() == ty)
    return v;
  return cir::CastOp::create(builder, loc, ty, cir::CastKind::bitcast, v);
}

// Free shadow in the reverse path as the same as free in primal space.
static void createShadowFree(cir::CallOp free, Value shadow, OpBuilder &builder,
                             MGradientUtilsReverse *gutils) {
  Location loc = free.getLoc();
  if (getAllocatorFn(free) == AllocatorFn::Realloc) {
    cir::FuncOp fn = getFreeFunction(free);
    cir::CallOp::create(
        builder, loc, SymbolRefAttr::get(fn), Type(),
        ValueRange{bitcastTo(builder, loc, shadow,
                             fn.getFunctionType().getInputs().front())});
    return;
  }
  SmallVector<Value> args{
      bitcastTo(builder, loc, shadow, free.getArgOperands().front().getType())};
  for (Value arg : free.getArgOperands().drop_front())
    args.push_back(builder.clone(*arg.getDefiningOp())->getResult(0));
  builder.clone(*gutils->getNewFromOriginal(free))->setOperands(args);
}

// Create stack allocation and store the orignal size inside from every source.
static LogicalResult storeAllocationSize(Value ptr, Operation *at, Value slot,
                                         Type sizeTy, MGradientUtils *gutils) {
  SmallVector<std::pair<Value, Operation *>> todo{{ptr, at}};
  DenseSet<std::pair<Value, Operation *>> seen;
  while (!todo.empty()) {
    auto [v, where] = todo.pop_back_val();
    if (!seen.insert({v, where}).second)
      continue;
    OpBuilder builder(gutils->getNewFromOriginal(where));
    Location loc = where->getLoc();
    auto store = [&](Value size) {
      if (size.getType() != sizeTy)
        return false;
      cir::StoreOp::create(builder, loc, size, slot);
      return true;
    };
    if (auto cast = v.getDefiningOp<cir::CastOp>(); isBitcast(cast)) {
      todo.push_back({cast.getSrc(), where});
      continue;
    }
    // Realloc on nullptr, previous size always == 0.
    if (isNullPointer(v)) {
      store(cir::ConstantOp::create(
          builder, loc, cir::IntAttr::get(cast<cir::IntType>(sizeTy), 0)));
      continue;
    }
    // Store the source.
    if (auto call = v.getDefiningOp<cir::CallOp>()) {
      auto args = call.getArgOperands();
      bool stored = false;
      switch (getAllocatorFn(call)) {
      case AllocatorFn::Alloc:
        stored = store(gutils->getNewFromOriginal(args[0]));
        break;
      case AllocatorFn::ZeroedAlloc:
        stored = store(cir::MulOp::create(builder, loc,
                                          gutils->getNewFromOriginal(args[0]),
                                          gutils->getNewFromOriginal(args[1])));
        break;
      case AllocatorFn::Realloc:
        stored = store(gutils->getNewFromOriginal(args[1]));
        break;
      default:
        break;
      }
      if (!stored)
        return failure();
      continue;
    }
    auto arg = dyn_cast<BlockArgument>(v);
    if (!arg || arg.getOwner()->isEntryBlock())
      return failure();
    Block *owner = arg.getOwner();
    for (auto it = owner->pred_begin(), e = owner->pred_end(); it != e; ++it) {
      // A block nothing branches to never passes anything.
      if (!(*it)->isEntryBlock() && (*it)->hasNoPredecessors())
        continue;
      auto br = dyn_cast<BranchOpInterface>((*it)->getTerminator());
      if (!br)
        return failure();
      SuccessorOperands operands =
          br.getSuccessorOperands(it.getSuccessorIndex());
      unsigned produced = operands.getProducedOperandCount();
      if (arg.getArgNumber() < produced)
        return failure();
      todo.push_back(
          {operands.getForwardedOperands()[arg.getArgNumber() - produced], br});
    }
  }
  return success();
}

// The type the memory realloc resizes holds, from the casts of the new
// pointer or of the old one, or null.
static Type reallocElementType(cir::CallOp realloc) {
  auto pointee = [](Value v) -> Type {
    Type t = cast<cir::PointerType>(v.getType()).getPointee();
    return isa<cir::VoidType>(t) ? Type() : t;
  };
  for (Operation *user : realloc->getResult(0).getUsers())
    if (auto cast = dyn_cast<cir::CastOp>(user); isBitcast(cast))
      if (Type t = pointee(cast.getResult()))
        return t;
  if (auto cast = realloc.getArgOperands().front().getDefiningOp<cir::CastOp>();
      isBitcast(cast))
    return pointee(cast.getSrc());
  return Type();
}

// Realloc a data need to copy as the shadow space value extend its lifetime
// until the reverse space. We have no choice but allocate a new one so that
// previous address is fine.
static LogicalResult reallocReverse(cir::CallOp call, OpBuilder &builder,
                                    MGradientUtilsReverse *gutils) {
  Location loc = call.getLoc();
  Value p = call.getArgOperands()[0], n = call.getArgOperands()[1];
  Type elemTy = reallocElementType(call);
  FailureOr<Type> floatTy =
      elemTy ? uniformFloatType(elemTy) : FailureOr<Type>(failure());
  if (failed(floatTy))
    return call.emitError()
           << "cannot differentiate a realloc in reverse mode without knowing "
              "whether it holds floats or other data";

  auto newCall = cast<cir::CallOp>(gutils->getNewFromOriginal(call));
  OpBuilder fwd(newCall);
  Type sizeTy = n.getType();
  OpBuilder entry =
      OpBuilder::atBlockBegin(&gutils->newFunc.getFunctionBody().front());
  DataLayout layout = DataLayout::closest(newCall);
  Value slot = cir::AllocaOp::create(
      entry, loc, cir::PointerType::get(sizeTy), "realloc.size",
      entry.getI64IntegerAttr(layout.getTypeABIAlignment(sizeTy)));
  if (failed(storeAllocationSize(p, call, slot, sizeTy, gutils)))
    return call.emitError()
           << "cannot differentiate a realloc in reverse mode unless what it "
              "resizes comes from malloc, calloc, realloc or null in this "
              "function";
  Value old = cir::LoadOp::create(fwd, loc, sizeTy, slot);
  Value size = newCall.getArgOperands()[1];
  Value smaller = cir::CmpOp::create(fwd, loc, cir::CmpOpKind::lt, old, size);
  Value copied = cir::SelectOp::create(fwd, loc, smaller, old, size);

  Value oldShadow = gutils->invertPointerM(p, fwd);
  // realloc(NULL, n) allocates like malloc(n).
  Value null =
      cast<AutoDiffTypeInterface>(p.getType()).createNullValue(fwd, loc);
  auto shadowCall = cast<cir::CallOp>(fwd.clone(*newCall));
  shadowCall->setOperand(0, null);
  Value shadow = shadowCall->getResult(0);

  MLIRContext *ctx = fwd.getContext();
  auto voidPtr = cir::PointerType::get(cir::VoidType::get(ctx));
  zeroBytes(fwd, loc, bitcastTo(fwd, loc, shadow, voidPtr), size);
  if (!*floatTy)
    cir::MemCpyOp::create(fwd, loc, bitcastTo(fwd, loc, shadow, voidPtr),
                          bitcastTo(fwd, loc, oldShadow, voidPtr), copied,
                          /*dst_alignment=*/nullptr,
                          /*src_alignment=*/nullptr);
  gutils->setInvertedPointer(call->getResult(0), shadow);

  Value shadowCache = gutils->initAndPushCache(shadow, fwd);
  Value oldShadowCache = gutils->initAndPushCache(oldShadow, fwd);
  Value copiedCache = gutils->initAndPushCache(copied, fwd);
  Value revShadow = gutils->popCache(shadowCache, builder);
  Value revOldShadow = gutils->popCache(oldShadowCache, builder);
  Value revCopied = gutils->popCache(copiedCache, builder);
  if (*floatTy)
    accumulateAndZero(builder, loc, *floatTy, revShadow, revOldShadow,
                      revCopied);
  if (cir::CallOp free = getGuaranteedFree(call, gutils))
    createShadowFree(free, revShadow, builder, gutils);
  return success();
}

// The forward sweep of a derivative that is split runs in the augmented
// function, whose frame is gone by the time the reverse function runs. We
// switch to heap allocated value so that both forward sweep and backward sweep
// can share it.
struct CIRAllocaOpReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<CIRAllocaOpReverse,
                                                       cir::AllocaOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    if (caches.empty())
      return success();
    cir::FuncOp free = getFreeFunction(op);
    for (Value cache : caches)
      cir::CallOp::create(builder, op->getLoc(), SymbolRefAttr::get(free),
                          Type(), ValueRange{gutils->popCache(cache, builder)});
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    if (!gutils->splitting)
      return {};
    auto alloca = cast<cir::AllocaOp>(op);
    auto newAlloca = cast<cir::AllocaOp>(gutils->getNewFromOriginal(op));
    OpBuilder builder(newAlloca);
    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();
    auto u64 = cir::IntType::get(ctx, 64, /*isSigned=*/false);
    uint64_t bytes =
        DataLayout::closest(newAlloca).getTypeSize(alloca.getAllocaType());
    Value size =
        cir::ConstantOp::create(builder, loc, cir::IntAttr::get(u64, bytes));
    if (Value count = newAlloca.getDynAllocSize()) {
      if (count.getType() != u64)
        count = cir::CastOp::create(builder, loc, u64, cir::CastKind::integral,
                                    count);
      size = cir::MulOp::create(builder, loc, size, count);
    }
    cir::FuncOp malloc = getMallocFunction(op);
    auto allocate = [&] {
      return cir::CallOp::create(builder, loc, SymbolRefAttr::get(malloc),
                                 malloc.getFunctionType().getReturnType(),
                                 ValueRange{size})
          ->getResult(0);
    };

    SmallVector<Value> caches;
    Value primal = allocate();
    Value typed = bitcastTo(builder, loc, primal, alloca.getType());
    // Lifetime markers only apply to stack slots. The slot itself stays,
    // unused, until it is cleaned up: the derivative still maps the original
    // op to it.
    for (Operation *user : llvm::make_early_inc_range(newAlloca->getUsers()))
      if (isa<cir::LifetimeStartOp, cir::LifetimeEndOp>(user))
        gutils->erase(user);
    newAlloca->replaceAllUsesWith(ValueRange{typed});
    gutils->originalToNewFn.map(alloca.getResult(), typed);
    caches.push_back(gutils->initAndPushCache(primal, builder));

    if (!gutils->isConstantValue(alloca.getResult())) {
      Value shadow = allocate();
      zeroBytes(builder, loc, shadow, size);
      gutils->setInvertedPointer(
          alloca.getResult(),
          bitcastTo(builder, loc, shadow, alloca.getType()));
      caches.push_back(gutils->initAndPushCache(shadow, builder));
    }
    return caches;
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    // A split derivative's shadow is made in cacheValues, with the primal.
    if (gutils->splitting)
      return success();
    (void)enzyme::detail::allocationForwardHandler(
        op, builder, (MGradientUtils *)gutils, /*zero*/ true);
    return success();
  }
};

struct CIRCallOpForward
    : public AutoDiffOpInterface::ExternalModel<CIRCallOpForward, cir::CallOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto call = cast<cir::CallOp>(op);
    AllocatorFn fn = getAllocatorFn(call);
    switch (fn) {
    case AllocatorFn::None:
      return enzyme::detail::callForwardHandler(op, builder, gutils);
    case AllocatorFn::Alloc:
    case AllocatorFn::ZeroedAlloc:
      if (!gutils->isConstantValue(call->getResult(0)))
        gutils->setInvertedPointer(
            call->getResult(0),
            createShadowAllocation(
                cast<cir::CallOp>(gutils->getNewFromOriginal(op)), fn,
                builder));
      return success();
    case AllocatorFn::Realloc: {
      // The shadow is resized with the primal. Like a fresh malloc's, the
      // grown part is only read once the primal has written it, and the
      // tangent of that write fills it in.
      if (gutils->isConstantValue(call->getResult(0)))
        return success();
      Operation *shadow = builder.clone(*gutils->getNewFromOriginal(op));
      shadow->setOperand(
          0, gutils->invertPointerM(call.getArgOperands().front(), builder));
      gutils->setInvertedPointer(call->getResult(0), shadow->getResult(0));
      return success();
    }
    case AllocatorFn::Free: {
      Value ptr = call.getArgOperands().front();
      if (gutils->isConstantValue(ptr))
        return success();
      Operation *shadowFree = builder.clone(*gutils->getNewFromOriginal(op));
      shadowFree->setOperand(0, gutils->invertPointerM(ptr, builder));
      return success();
    }
    }
    llvm_unreachable("unknown allocator function");
  }
};

// Allocation can be aligned, get the orignal parameter of alignment and return
// the mangled name
static std::string deferredDeallocation(cir::CallOp free) {
  StringRef callee = free.getCallee().value_or("");
  StringRef unaligned =
      llvm::StringSwitch<StringRef>(callee)
          .Case("free", "free")
          .Cases({"_ZdlPv", "_ZdlPvm", "_ZdlPvj", "_ZdlPvRKSt9nothrow_t"},
                 "_ZdlPv")
          .Cases({"_ZdaPv", "_ZdaPvm", "_ZdaPvj", "_ZdaPvRKSt9nothrow_t"},
                 "_ZdaPv")
          .Default("");
  if (!unaligned.empty())
    return unaligned.str();
  // The alignment is the operand after the pointer, and the size if any.
  auto [aligned, alignOperand] =
      llvm::StringSwitch<std::pair<StringRef, unsigned>>(callee)
          .Cases(
              {"_ZdlPvSt11align_val_t", "_ZdlPvSt11align_val_tRKSt9nothrow_t"},
              {"_ZdlPvSt11align_val_t", 1})
          .Cases({"_ZdlPvmSt11align_val_t", "_ZdlPvjSt11align_val_t"},
                 {"_ZdlPvSt11align_val_t", 2})
          .Cases(
              {"_ZdaPvSt11align_val_t", "_ZdaPvSt11align_val_tRKSt9nothrow_t"},
              {"_ZdaPvSt11align_val_t", 1})
          .Cases({"_ZdaPvmSt11align_val_t", "_ZdaPvjSt11align_val_t"},
                 {"_ZdaPvSt11align_val_t", 2})
          .Default({"", 0});
  if (aligned.empty() || free.getArgOperands().size() <= alignOperand)
    return "";
  auto constant =
      free.getArgOperands()[alignOperand].getDefiningOp<cir::ConstantOp>();
  auto alignment =
      constant ? dyn_cast<cir::IntAttr>(constant.getValue()) : cir::IntAttr();
  if (!alignment)
    return "";
  return (aligned + ":" + Twine(alignment.getUInt())).str();
}

// Create deferred shadow. Function name is in compile time, set to the
// attribute directly. Only object is different.
static void deferShadowFree(cir::CallOp free, OpBuilder &builder,
                            MGradientUtilsReverse *gutils) {
  Value ptr = free.getArgOperands().front();
  if (gutils->isConstantValue(ptr) ||
      guaranteedFrees(gutils).frees.contains(free))
    return;
  std::string dealloc = deferredDeallocation(free);
  if (dealloc.empty())
    return;
  Location loc = free.getLoc();
  auto voidPtr = cir::PointerType::get(cir::VoidType::get(free.getContext()));
  Value shadow =
      bitcastTo(builder, loc, gutils->invertPointerM(ptr, builder), voidPtr);
  Value primal =
      bitcastTo(builder, loc, gutils->getNewFromOriginal(ptr), voidPtr);
  Value same =
      cir::CmpOp::create(builder, loc, cir::CmpOpKind::eq, shadow, primal);
  Value null =
      cast<AutoDiffTypeInterface>(voidPtr).createNullValue(builder, loc);
  Value toFree = cir::SelectOp::create(builder, loc, same, null, shadow);
  enzyme::PushOp::create(builder, loc,
                         gutils->getDeferredFrees(voidPtr, dealloc), toFree);
}

struct CIRCallOpReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<CIRCallOpReverse,
                                                       cir::CallOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto call = cast<cir::CallOp>(op);
    switch (getAllocatorFn(call)) {
    case AllocatorFn::None:
      return enzyme::detail::callReverseHandler(op, builder, gutils, caches);
    case AllocatorFn::Alloc:
    case AllocatorFn::ZeroedAlloc: {
      if (gutils->isConstantValue(call->getResult(0)))
        return success();
      cir::CallOp free = getGuaranteedFree(call, gutils);
      if (!free)
        return success();
      OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
      Value cache = gutils->initAndPushCache(
          gutils->invertedPointers.lookupOrNull(call->getResult(0)),
          cacheBuilder);
      createShadowFree(free, gutils->popCache(cache, builder), builder, gutils);
      return success();
    }
    case AllocatorFn::Realloc:
      if (gutils->isConstantValue(call->getResult(0)))
        return success();
      return reallocReverse(call, builder, gutils);
    case AllocatorFn::Free:
      return success();
    }
    llvm_unreachable("unknown allocator function");
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    if (getAllocatorFn(cast<cir::CallOp>(op)) == AllocatorFn::None)
      return enzyme::detail::callCacheValues(op, gutils);
    return {};
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    auto call = cast<cir::CallOp>(op);
    AllocatorFn fn = getAllocatorFn(call);
    if (isAllocation(fn) && !gutils->isConstantValue(call->getResult(0)))
      gutils->setInvertedPointer(
          call->getResult(0),
          createShadowAllocation(
              cast<cir::CallOp>(gutils->getNewFromOriginal(op)), fn, builder));
    if (fn == AllocatorFn::Free)
      deferShadowFree(call, builder, gutils);
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
  SmallVector<Value> unpackCallResults(Operation *, OpBuilder &b,
                                       Operation *call,
                                       unsigned numResults) const {
    if (numResults == call->getNumResults())
      return SmallVector<Value>(call->getResults());
    assert(call->getNumResults() == 1 && "expected one packed result");
    Value packed = call->getResult(0);
    SmallVector<Value> results;
    for (unsigned i = 0; i < numResults; ++i)
      results.push_back(
          cir::ExtractMemberOp::create(b, call->getLoc(), packed, i));
    return results;
  }
  Type getTapeType(Operation *self, Type cacheType) const {
    MLIRContext *ctx = self->getContext();
    return cir::PointerType::get(cir::VoidType::get(ctx));
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

static bool cirIsZeroInt(Value v) {
  if (auto c = v.getDefiningOp<cir::ConstantOp>())
    if (auto ia = dyn_cast<cir::IntAttr>(c.getValue()))
      return ia.getValue().isZero();
  return false;
}

static bool cirZeroOffset(cir::GetElementOp op) {
  return cirIsZeroInt(op.getIndex());
}
static bool cirZeroOffset(cir::PtrStrideOp op) {
  return cirIsZeroInt(op.getStride());
}

static bool cirZeroOffset(cir::GetMemberOp op) { return op.getIndex() == 0; }
static bool cirZeroOffset(cir::BaseClassAddrOp op) {
  return op.getOffset().isZero();
}
static bool cirZeroOffset(cir::DerivedClassAddrOp op) {
  return op.getOffset().isZero();
}
static bool cirZeroOffset(cir::ComplexRealPtrOp) { return true; }
static bool cirZeroOffset(cir::ComplexImagPtrOp) { return false; }
static bool cirZeroOffset(cir::PtrMaskOp) { return false; }
static bool cirZeroOffset(cir::GetRuntimeMemberOp) { return false; }
static bool cirZeroOffset(cir::CastOp op) {
  return isPointerCast(op.getKind());
}

template <typename OpTy>
struct CIRAddressView
    : public ViewLikeOpInterface::ExternalModel<CIRAddressView<OpTy>, OpTy> {
  Value getViewSource(Operation *op) const { return op->getOperand(0); }
};

template <typename OpTy>
struct CIRAddressOffset
    : public OffsetViewInterface::ExternalModel<CIRAddressOffset<OpTy>, OpTy> {
  bool isZeroOffset(Operation *op) const {
    return cirZeroOffset(cast<OpTy>(op));
  }
};

template <typename OpTy> static void registerCIRAddressView(MLIRContext &ctx) {
  OpTy::template attachInterface<CIRAddressView<OpTy>>(ctx);
  OpTy::template attachInterface<CIRAddressOffset<OpTy>>(ctx);
}

template <typename OpTy>
struct CIRMemTransfer
    : public MemTransferOpInterface::ExternalModel<CIRMemTransfer<OpTy>, OpTy> {
  Value getTransferDest(Operation *op) const { return cast<OpTy>(op).getDst(); }
  Value getTransferSource(Operation *op) const {
    return cast<OpTy>(op).getSrc();
  }
};

class CIRAutoDiffControlFlow : public AutoDiffControlFlowInterface {
public:
  using AutoDiffControlFlowInterface::AutoDiffControlFlowInterface;

  Type getBranchIndexType(MLIRContext *context) const override {
    return cir::IntType::get(context, 32, /*isSigned=*/true);
  }
  Value createBranchIndex(OpBuilder &b, Location loc,
                          int64_t value) const override {
    return cir::ConstantOp::create(
        b, loc, cir::IntAttr::get(getBranchIndexType(b.getContext()), value));
  }
  Operation *createBranch(OpBuilder &b, Location loc,
                          Block *dest) const override {
    return cir::BrOp::create(b, loc, dest);
  }
  Operation *createSwitch(OpBuilder &b, Location loc, Value flag,
                          Block *defaultDest, ArrayRef<int64_t> caseValues,
                          BlockRange caseDests) const override {
    // Switch with no case will be erased, fallback to normal br
    if (caseValues.empty())
      return cir::BrOp::create(b, loc, defaultDest);
    SmallVector<APInt> values;
    for (int64_t v : caseValues)
      values.push_back(APInt(32, v, /*isSigned=*/true));
    return cir::SwitchFlatOp::create(b, loc, flag, defaultDest, ValueRange(),
                                     values, caseDests,
                                     SmallVector<ValueRange>(values.size()));
  }
  Value createSelectOnBranchIndex(OpBuilder &b, Location loc, Value flag,
                                  int64_t value, Value ifTrue,
                                  Value ifFalse) const override {
    Value isValue = cir::CmpOp::create(b, loc, cir::CmpOpKind::eq, flag,
                                       createBranchIndex(b, loc, value));
    return cir::SelectOp::create(b, loc, isValue, ifTrue, ifFalse);
  }
};

struct CIRReturnOpFunctionReturnInterface
    : public FunctionReturnOpInterface::ExternalModel<
          CIRReturnOpFunctionReturnInterface, cir::ReturnOp> {};

template <typename OpTy>
struct CIRNoMemoryEffect
    : public MemoryEffectOpInterface::ExternalModel<CIRNoMemoryEffect<OpTy>,
                                                    OpTy> {
  void
  getEffects(Operation *,
             SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &) const {}
};

void mlir::enzyme::registerCIRDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, cir::CIRDialect *dialect) {
    dialect->addInterfaces<CIRAutoDiffControlFlow>();
    cir::SwitchFlatOp::attachInterface<SwitchFlatBranchOpInterface>(*context);
    cir::StoreOp::attachInterface<CIRStoreLike>(*context);
    cir::StoreOp::attachInterface<CIRStoreOpInterfaceReverse>(*context);
    cir::LoadOp::attachInterface<CIRLoadLike>(*context);
    cir::LoadOp::attachInterface<CIRLoadOpInterfaceReverse>(*context);
    cir::AllocaOp::attachInterface<CIRAllocaLike>(*context);
    cir::AllocaOp::attachInterface<
        detail::AutoDiffUsingAllocationFwd<cir::AllocaOp>>(*context);
    cir::AllocaOp::attachInterface<CIRAllocaOpReverse>(*context);
    cir::CallOp::attachInterface<CIRCallOpForward>(*context);
    cir::CallOp::attachInterface<CIRCallOpReverse>(*context);
    cir::CastOp::attachInterface<CIRCastOpActivity>(*context);
    cir::CastOp::attachInterface<CIRCastOpForward>(*context);
    cir::CastOp::attachInterface<CIRCastOpReverse>(*context);
    cir::CopyOp::attachInterface<CIRCopyOpForward>(*context);
    cir::CopyOp::attachInterface<CIRCopyOpReverse>(*context);
    cir::CopyOp::attachInterface<CIRMemTransfer<cir::CopyOp>>(*context);
    cir::MemCpyOp::attachInterface<CIRMemTransfer<cir::MemCpyOp>>(*context);
    cir::MemMoveOp::attachInterface<CIRMemTransfer<cir::MemMoveOp>>(*context);
    cir::MemCpyOp::attachInterface<CIRMemCpyOpForward>(*context);
    cir::MemCpyOp::attachInterface<CIRMemCpyOpReverse>(*context);
    cir::GetGlobalOp::attachInterface<CIRGetGlobalOpForward>(*context);
    cir::GetGlobalOp::attachInterface<CIRGetGlobalOpReverse>(*context);
    cir::LifetimeStartOp::attachInterface<
        CIRLifetimeForward<cir::LifetimeStartOp>>(*context);
    cir::LifetimeEndOp::attachInterface<CIRLifetimeForward<cir::LifetimeEndOp>>(
        *context);
    // Control never gets past cir.unreachable or cir.trap, so there is
    // nothing to undo; the reverse sweep still has to be told so for a
    // terminator, which the activity tables alone do not cover.
    cir::UnreachableOp::attachInterface<
        detail::NoopRevAutoDiffInterface<cir::UnreachableOp>>(*context);
    cir::TrapOp::attachInterface<detail::NoopRevAutoDiffInterface<cir::TrapOp>>(
        *context);
    registerCIRAddressView<cir::GetElementOp>(*context);
    registerCIRAddressView<cir::GetMemberOp>(*context);
    registerCIRAddressView<cir::PtrStrideOp>(*context);
    registerCIRAddressView<cir::BaseClassAddrOp>(*context);
    registerCIRAddressView<cir::DerivedClassAddrOp>(*context);
    registerCIRAddressView<cir::ComplexRealPtrOp>(*context);
    registerCIRAddressView<cir::ComplexImagPtrOp>(*context);
    registerCIRAddressView<cir::PtrMaskOp>(*context);
    registerCIRAddressView<cir::GetRuntimeMemberOp>(*context);
    // TODO: it might be wrong as CastOp create new View.
    registerCIRAddressView<cir::CastOp>(*context);
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
    cir::ReturnOp::attachInterface<CIRNoMemoryEffect<cir::ReturnOp>>(*context);
    cir::SwitchFlatOp::attachInterface<CIRNoMemoryEffect<cir::SwitchFlatOp>>(
        *context);
    cir::UnreachableOp::attachInterface<CIRNoMemoryEffect<cir::UnreachableOp>>(
        *context);
    cir::TrapOp::attachInterface<CIRNoMemoryEffect<cir::TrapOp>>(*context);
  });
}
