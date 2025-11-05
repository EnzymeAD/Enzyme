//===- LLVMAutoDiffOpInterfaceImpl.cpp - Interface external model  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect/LLVMExt/LLVMExt.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/LLVMDerivatives.inc"

struct InlineAsmActivityInterface
    : public ActivityOpInterface::ExternalModel<InlineAsmActivityInterface,
                                                LLVM::InlineAsmOp> {
  bool isInactive(Operation *op) const {
    auto asmOp = cast<LLVM::InlineAsmOp>(op);
    auto str = asmOp.getAsmString();
    return str.contains("cpuid") || str.contains("exit");
  }
  bool isArgInactive(Operation *op, size_t) const { return isInactive(op); }
};

class PointerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<PointerTypeInterface,
                                                  LLVM::LLVMPointerType> {
public:
  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    return LLVM::ZeroOp::create(builder, loc, self);
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("TODO");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    // TODO inspect val and memset corresponding size
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

struct GEPOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<GEPOpInterfaceReverse,
                                                       LLVM::GEPOp> {
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
    auto gep = cast<LLVM::GEPOp>(op);
    auto newGep = cast<LLVM::GEPOp>(gutils->getNewFromOriginal(op));
    auto base = gep.getBase();
    if (!gutils->isConstantValue(base)) {
      auto baseShadow = gutils->invertPointerM(base, builder);
      auto shadowGep = cast<LLVM::GEPOp>(builder.clone(*newGep));
      shadowGep.getBaseMutable().assign(baseShadow);
      gutils->setInvertedPointer(gep.getRes(), shadowGep->getResult(0));
    }
  }
};

struct LoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<LoadOpInterfaceReverse,
                                                       LLVM::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    Value addr = loadOp.getAddr();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      if (!gutils->isConstantValue(loadOp) && !gutils->isConstantValue(addr)) {
        Value gradient = gutils->diffe(loadOp, builder);
        Value addrGradient = gutils->popCache(caches.front(), builder);

        if (!gutils->AtomicAdd) {
          Value loadedGradient = LLVM::LoadOp::create(builder, loadOp.getLoc(),
                                                      iface, addrGradient);
          Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                  loadedGradient, gradient);

          LLVM::StoreOp::create(builder, loadOp.getLoc(), addedGradient,
                                addrGradient);
        } else {
          LLVM::AtomicRMWOp::create(builder, loadOp.getLoc(),
                                    LLVM::AtomicBinOp::fadd, addrGradient,
                                    gradient, LLVM::AtomicOrdering::monotonic);
        }
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    auto addr = loadOp.getAddr();
    if (!(isa<AutoDiffTypeInterface>(loadOp.getType()) &&
          (!gutils->isConstantValue(loadOp) && !gutils->isConstantValue(addr))))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto loadOp = cast<LLVM::LoadOp>(op);
    // Value ptr = loadOp.getAddr();
  }
};

struct StoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<StoreOpInterfaceReverse,
                                                       LLVM::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    Value val = storeOp.getValue();
    Value addr = storeOp.getAddr();

    auto iface = cast<AutoDiffTypeInterface>(val.getType());

    if (!gutils->isConstantValue(addr)) {
      Value addrGradient = gutils->popCache(caches.front(), builder);

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient = LLVM::LoadOp::create(
              builder, storeOp.getLoc(), val.getType(), addrGradient);
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        LLVM::StoreOp::create(builder, storeOp.getLoc(), zero, addrGradient);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    auto addr = storeOp.getAddr();
    if (gutils->isConstantValue(addr))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

std::optional<Value> findPtrSize(Value ptr) {
  if (auto allocOp = ptr.getDefiningOp<llvm_ext::AllocOp>())
    return allocOp.getSize();

  for (auto user : ptr.getUsers()) {
    if (auto psh = dyn_cast<llvm_ext::PtrSizeHintOp>(user)) {
      return psh.getSize();
    }
  }

  return std::nullopt;
}

struct PointerClonableTypeInterface
    : public ClonableTypeInterface::ExternalModel<PointerClonableTypeInterface,
                                                  LLVM::LLVMPointerType> {
  mlir::Value cloneValue(Type self, OpBuilder &builder, Value value) const {
    auto ptrSize = findPtrSize(value);
    if (!ptrSize) {
      llvm::errs() << "cannot find size of ptr: " << value << "\n";
      return nullptr;
    }

    auto clone = builder.create<llvm_ext::AllocOp>(
        value.getLoc(), LLVM::LLVMPointerType::get(value.getContext()),
        *ptrSize);
    builder.create<LLVM::MemcpyOp>(value.getLoc(), clone, value, *ptrSize,
                                   /*isVolatile*/ false);

    return clone;
  }

  void freeClonedValue(Type self, OpBuilder &builder, Value value) const {
    builder.create<llvm_ext::FreeOp>(value.getLoc(), value);
  }
};

} // namespace

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    LLVM::LLVMPointerType::attachInterface<PointerTypeInterface>(*context);
    LLVM::LLVMPointerType::attachInterface<PointerClonableTypeInterface>(
        *context);
    LLVM::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    LLVM::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
    LLVM::GEPOp::attachInterface<GEPOpInterfaceReverse>(*context);
    registerInterfaces(context);
    LLVM::UnreachableOp::template attachInterface<
        detail::NoopRevAutoDiffInterface<LLVM::UnreachableOp>>(*context);
  });
}
