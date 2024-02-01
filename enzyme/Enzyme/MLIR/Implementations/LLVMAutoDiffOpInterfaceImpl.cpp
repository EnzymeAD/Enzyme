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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/LLVMDerivatives.inc"
} // namespace

namespace {
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

struct StoreOpInterface
    : public AutoDiffOpInterface::ExternalModel<StoreOpInterface,
                                                LLVM::StoreOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    if (!gutils->isConstantValue(storeOp.getAddr())) {
      builder.create<LLVM::StoreOp>(
          storeOp.getLoc(), gutils->invertPointerM(storeOp.getValue(), builder),
          gutils->invertPointerM(storeOp.getAddr(), builder));
    }
    gutils->eraseIfUnused(op);
    return success();
  }
};

struct AllocaOpInterface
    : public AutoDiffOpInterface::ExternalModel<AllocaOpInterface,
                                                LLVM::AllocaOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto allocOp = cast<LLVM::AllocaOp>(op);
    if (!gutils->isConstantValue(allocOp)) {
      Operation *nop = gutils->cloneWithNewOperands(builder, op);
      gutils->setDiffe(allocOp, nop->getResult(0), builder);
    }
    gutils->eraseIfUnused(op);
    return success();
  }
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

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool requiresShadow(Type self) const { return true; }
};
} // namespace

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    LLVM::StoreOp::attachInterface<StoreOpInterface>(*context);
    LLVM::AllocaOp::attachInterface<AllocaOpInterface>(*context);
    LLVM::LLVMPointerType::attachInterface<PointerTypeInterface>(*context);
    registerInterfaces(context);
  });
}
