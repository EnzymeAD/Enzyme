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

struct LoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<LoadOpInterfaceReverse,
                                                       LLVM::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    Value addr = loadOp.getAddr();
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      if (!gutils->isConstantValue(addr) && !gutils->isConstantValue(loadOp)) {
        Value grad = gutils->diffe(loadOp, builder);
        Value shadow = gutils->invertPointerM(addr, builder);
        gutils->zeroDiffe(loadOp, builder);

        // TODO: emit serial += where possible
        builder.create<LLVM::AtomicRMWOp>(loadOp.getLoc(),
                                          LLVM::AtomicBinOp::fadd, shadow, grad,
                                          LLVM::AtomicOrdering::monotonic);
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

struct StoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<StoreOpInterfaceReverse,
                                                       LLVM::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    Value storedVal = storeOp.getValue();
    if (auto iface = dyn_cast<AutoDiffTypeInterface>(storedVal.getType())) {
      if (!gutils->isConstantValue(storeOp.getAddr()) &&
          !gutils->isConstantValue(storedVal)) {
        Value daddr = gutils->invertPointerM(storeOp.getAddr(), builder);
        Value tmp = builder.create<LLVM::LoadOp>(storeOp.getLoc(),
                                                 storedVal.getType(), daddr);
        gutils->zeroDiffe(storeOp.getAddr(), builder);
        gutils->addToDiffe(storedVal, tmp, builder);
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
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto gepOp = cast<LLVM::GEPOp>(op);
    auto newGepOp = cast<LLVM::GEPOp>(gutils->getNewFromOriginal(op));
    using llvm::errs;
    errs() << "[debug] creating shadow for gep\n";
    if (!gutils->isConstantValue(gepOp.getBase())) {
      SmallVector<LLVM::GEPArg> indices;
      indices.reserve(newGepOp.getIndices().size());
      for (llvm::PointerUnion<IntegerAttr, Value> idx : newGepOp.getIndices()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(idx)) {
          indices.push_back(intAttr.getInt());
        } else {
          indices.push_back(cast<Value>(idx));
        }
      }

      Value shadow = builder.create<LLVM::GEPOp>(
          gepOp.getLoc(), gepOp.getType(), gepOp.getElemType(),
          gutils->invertPointerM(gepOp.getBase(), builder), indices);
      gutils->setDiffe(gepOp, shadow, builder);
    }
  }
};

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
};
} // namespace

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    LLVM::LLVMPointerType::attachInterface<PointerTypeInterface>(*context);
    registerInterfaces(context);

    LLVM::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    LLVM::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
    LLVM::GEPOp::attachInterface<GEPOpInterfaceReverse>(*context);
  });
}
