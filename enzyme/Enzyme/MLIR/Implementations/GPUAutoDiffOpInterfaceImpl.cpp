//===- GPUAutoDiffOpInterfaceImpl.cpp - Interface external model ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct GPUAllocOpInterface
    : public MultidimensionalAllocInterface::ExternalModel<GPUAllocOpInterface,
                                                           gpu::AllocOp> {
  Value allocate(Operation *op, OpBuilder &rewriter, Location loc, Type newType,
                 ValueRange dynamicDims) const {
    return gpu::AllocOp::create(rewriter, loc, cast<MemRefType>(newType),
                                /*asyncDependencies=*/ValueRange{}, dynamicDims,
                                /*symbolOperands=*/ValueRange{})
        .getMemref();
  }

  void deallocate(Operation *op, OpBuilder &rewriter, Location loc,
                  Value val) const {
    gpu::DeallocOp::create(rewriter, loc, /*resultTypes=*/TypeRange(),
                           /*asyncDependencies=*/ValueRange(), val);
  }

  bool isDeallocation(Operation *op, Operation *user) const {
    return isa<gpu::DeallocOp>(user);
  }
};
} // namespace

void mlir::enzyme::registerGPUDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, gpu::GPUDialect *) {
    gpu::AllocOp::attachInterface<GPUAllocOpInterface>(*context);
  });
}
