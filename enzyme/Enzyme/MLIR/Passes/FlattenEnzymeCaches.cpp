//===- FlattenEnzymeCaches.cpp - Flatten multi-dim enzyme memory ops -------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert multi-dimensional memory operations
// to equivalent 1-D operations.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_FLATTENENZYMECACHESPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
// Recursively flatten the MemRefType for nested memrefs (e.g.
// memref<?xmemref<?x?xf32, 1>>, which represents a host memref of pointers to
// GPU memrefs)
Type flattenType(Type oldType, OpBuilder &builder) {
  if (auto memrefType = dyn_cast<MemRefType>(oldType)) {
    return MemRefType::get({ShapedType::kDynamic},
                           flattenType(memrefType.getElementType(), builder),
                           builder.getDimIdentityMap(),
                           memrefType.getMemorySpace());
  }
  return oldType;
}

Operation *createFlatAlloc(MemRefType oldType, Operation *alloc,
                           ImplicitLocOpBuilder &builder) {
  auto newType = cast<MemRefType>(flattenType(oldType, builder));
  // Compute the size of the flattened allocation
  unsigned opidx = 0;
  auto materializeDimSize = [&](unsigned dim) -> Value {
    if (oldType.getDimSize(dim) == ShapedType::kDynamic)
      return alloc->getOperand(opidx++);
    return arith::ConstantIndexOp::create(builder, alloc->getLoc(),
                                          oldType.getDimSize(dim));
  };

  Value size = materializeDimSize(0);
  for (unsigned dim = 1; dim < oldType.getRank(); dim++)
    size = arith::MulIOp::create(builder, size, materializeDimSize(dim));

  StringRef operandSegmentSizeAttrName = OpTrait::AttrSizedOperandSegments<
      memref::AllocOp>::getOperandSegmentSizeAttr();

  // Create the flattened allocation
  Operation *flatAllocOp = builder.clone(*alloc);
  auto oldSegments =
      alloc->getAttrOfType<DenseI32ArrayAttr>(operandSegmentSizeAttrName);
  SmallVector<int32_t> operandSegmentSizes(oldSegments.asArrayRef());
  // Making this dialect-agnostic is a pain because the operand segment index
  // representing dynamic dims is 0 for memref.alloc, 1 for gpu.alloc
  if (isa<gpu::AllocOp>(alloc)) {
    operandSegmentSizes[1] = 1;
  } else {
    operandSegmentSizes[0] = 1;
  }
  flatAllocOp->setOperands(size);
  flatAllocOp->getResult(0).setType(newType);
  flatAllocOp->setAttr(operandSegmentSizeAttrName,
                       builder.getDenseI32ArrayAttr(operandSegmentSizes));
  return flatAllocOp;
}

Value computeFlatIndex(ValueRange indices, ValueRange dynamicSizes,
                       ArrayRef<int64_t> oldShape,
                       ImplicitLocOpBuilder &builder) {
  // Compute the flat index by iterating over indices in reverse
  // We assume the caches have identity layouts, so strides can be
  // computed from sizes.
  Value flatIndex = arith::ConstantIndexOp::create(builder, 0);
  int64_t dynamicIndex = dynamicSizes.size();
  Value runningStride = arith::ConstantIndexOp::create(builder, 1);
  for (int64_t dim = oldShape.size(); dim-- > 0;) {
    Value mul = arith::MulIOp::create(builder, indices[dim], runningStride);
    flatIndex = arith::AddIOp::create(builder, mul, flatIndex);

    // Update the stride
    if (oldShape[dim] == ShapedType::kDynamic) {
      runningStride = arith::MulIOp::create(builder, runningStride,
                                            dynamicSizes[--dynamicIndex]);
    } else {
      runningStride = arith::MulIOp::create(
          builder, runningStride,
          arith::ConstantIndexOp::create(builder, oldShape[dim]));
    }
  }

  return flatIndex;
}

struct FlattenEnzymeCaches
    : public enzyme::impl::FlattenEnzymeCachesPassBase<FlattenEnzymeCaches> {
  void runOnOperation() override {
    SetVector<Operation *> allocations;
    getOperation()->walk([&allocations](enzyme::StoreOp storeOp) {
      Operation *alloc = storeOp.getMemref().getDefiningOp();
      assert(alloc && alloc->getNumResults() == 1);
      MemRefType oldType = storeOp.getMemref().getType();

      if (alloc->getNumOperands() == 0)
        return;
      if (oldType.getRank() == 0)
        return;
      allocations.insert(alloc);
    });

    for (Operation *alloc : llvm::make_early_inc_range(allocations)) {
      auto oldType = cast<MemRefType>(alloc->getResultTypes().front());
      ImplicitLocOpBuilder abuilder(alloc->getLoc(), alloc);
      Operation *flatAllocOp = createFlatAlloc(oldType, alloc, abuilder);
      alloc->replaceAllUsesWith(flatAllocOp->getResults());
      alloc->erase();
    }

    // Update users
    getOperation()->walk([](enzyme::StoreOp storeOp) {
      ImplicitLocOpBuilder sbuilder(storeOp.getLoc(), storeOp);
      Value flatIndex =
          computeFlatIndex(storeOp.getIndices(), storeOp.getSizes(),
                           storeOp.getStaticSizes(), sbuilder);
      memref::StoreOp::create(sbuilder, storeOp.getValue(), storeOp.getMemref(),
                              flatIndex);
      storeOp.erase();
    });
    getOperation()->walk([](enzyme::LoadOp loadOp) {
      ImplicitLocOpBuilder lbuilder(loadOp.getLoc(), loadOp);
      Value flatIndex = computeFlatIndex(loadOp.getIndices(), loadOp.getSizes(),
                                         loadOp.getStaticSizes(), lbuilder);
      auto flatLoad =
          memref::LoadOp::create(lbuilder, loadOp.getMemref(), flatIndex);
      loadOp.replaceAllUsesWith(flatLoad.getResult());
      loadOp.erase();
    });
  }
};
} // namespace
