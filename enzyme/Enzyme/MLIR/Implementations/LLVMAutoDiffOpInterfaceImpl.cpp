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
    return builder.create<LLVM::ZeroOp>(loc, self);
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
        Value addrGradient = gutils->invertPointerM(addr, builder);

        if (!gutils->AtomicAdd) {
          Value loadedGradient = builder.create<LLVM::LoadOp>(
              loadOp.getLoc(), iface, addrGradient);
          Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                  loadedGradient, gradient);

          builder.create<LLVM::StoreOp>(loadOp.getLoc(), addedGradient,
                                        addrGradient);
        } else {
          builder.create<LLVM::AtomicRMWOp>(
              loadOp.getLoc(), LLVM::AtomicBinOp::fadd, addrGradient, gradient,
              LLVM::AtomicOrdering::monotonic);
        }
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
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
      Value addrGradient = gutils->invertPointerM(addr, builder);

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient = builder.create<LLVM::LoadOp>(
              storeOp.getLoc(), val.getType(), addrGradient);
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        builder.create<LLVM::StoreOp>(storeOp.getLoc(), zero, addrGradient);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

std::optional<Value> findPtrSize(Value ptr, OpBuilder &builder) {
  if (auto ba = dyn_cast<BlockArgument>(ptr)) {
    Block *owner = ba.getOwner();
    Operation *op = owner->getParentOp();

    FunctionOpInterface fn = dyn_cast<FunctionOpInterface>(op);
    if (!fn)
      fn = op->getParentOfType<FunctionOpInterface>();

    if (&fn.front() != owner)
      return std::nullopt;

    auto argAttrs = fn.getArgAttrs(ba.getArgNumber());
    for (auto nattr : argAttrs) {
      if (nattr.getName() == "enzyme.ptr_size_hint") {
        auto attr = cast<IntegerAttr>(nattr.getValue());
        return builder.create<LLVM::ConstantOp>(ptr.getLoc(), attr.getType(),
                                                attr);
      }
    }
  }
  return std::nullopt;
}

struct PointerCachableTypeInterface
    : public CachableTypeInterface::ExternalModel<PointerCachableTypeInterface,
                                                  LLVM::LLVMPointerType> {
  mlir::Operation *createPush(Type self, OpBuilder &builder, Value cache,
                              Value value) const {
    auto ptrSize = findPtrSize(value, builder);
    if (!ptrSize) {
      llvm::errs() << "cannot find size of ptr: " << value << "\n";
      return nullptr;
    }

    auto clone = builder.create<enzyme::AllocOp>(
        cache.getLoc(), LLVM::LLVMPointerType::get(cache.getContext()),
        *ptrSize);
    builder.create<LLVM::MemcpyOp>(cache.getLoc(), clone, value, *ptrSize,
                                   /*isVolatile*/ false);

    return builder.create<enzyme::PushOp>(cache.getLoc(), cache, clone);
  }

  Value createPop(Type self, OpBuilder &builder, Value cache) const {
    return builder.create<enzyme::PopOp>(cache.getLoc(), self, cache);
  }

  void deletePoppedValue(Type self, OpBuilder &builder, Value value) const {
    builder.create<enzyme::FreeOp>(value.getLoc(), value);
  }
};

} // namespace

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    LLVM::LLVMPointerType::attachInterface<PointerTypeInterface>(*context);
    LLVM::LLVMPointerType::attachInterface<PointerCachableTypeInterface>(
        *context);
    LLVM::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    LLVM::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
    registerInterfaces(context);
    LLVM::UnreachableOp::template attachInterface<
        detail::NoopRevAutoDiffInterface<LLVM::UnreachableOp>>(*context);
  });
}
