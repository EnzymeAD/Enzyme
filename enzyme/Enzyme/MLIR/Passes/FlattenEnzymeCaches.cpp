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
#include "Interfaces/AutoDiffOpInterface.h"
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

Value createFlatAlloc(MemRefType oldType,
                      enzyme::MultidimensionalAllocInterface allocOp,
                      ImplicitLocOpBuilder &builder) {
  auto newType = cast<MemRefType>(flattenType(oldType, builder));
  // Compute the size of the flattened allocation
  unsigned opidx = 0;

  SmallVector<Value> dynamicDims;
  allocOp.appendDynamicDims(dynamicDims);

  Value size = nullptr;
  for (unsigned dim = 0; dim < oldType.getRank(); dim++) {
    Value bound;
    if (oldType.getDimSize(dim) == ShapedType::kDynamic) {
      bound = dynamicDims[opidx];
      opidx++;
    } else {
      bound = arith::ConstantIndexOp::create(builder, oldType.getDimSize(dim));
    }
    if (size == nullptr) {
      size = bound;
    } else {
      size = arith::MulIOp::create(builder, size, bound);
    }
  }

  if (size == nullptr) {
    size = arith::ConstantIndexOp::create(builder, 0);
  }

  return allocOp.allocate(builder, builder.getLoc(), newType, size);
}

Value computeFlatIndex(ValueRange indices, ValueRange dynamicSizes,
                       ArrayRef<int64_t> oldShape,
                       ImplicitLocOpBuilder &builder) {
  if (oldShape.size() == 0) {
    return arith::ConstantIndexOp::create(builder, 0);
  }

  // Compute the flat index by iterating over indices in reverse
  // We assume the caches have identity layouts, so strides can be
  // computed from sizes.
  Value flatIndex = indices[0];
  int64_t dynamicIndex = 0;
  if (oldShape[0] == ShapedType::kDynamic) {
    dynamicIndex++;
  }

  for (int64_t dim = 1; dim < oldShape.size(); dim++) {
    Value bound;
    if (oldShape[dim] == ShapedType::kDynamic) {
      bound = dynamicSizes[dynamicIndex];
      dynamicIndex++;
    } else {
      bound = arith::ConstantIndexOp::create(builder, oldShape[dim]);
    }

    flatIndex = arith::MulIOp::create(builder, flatIndex, bound);

    flatIndex = arith::AddIOp::create(builder, flatIndex, indices[dim]);
  }

  return flatIndex;
}

struct FlattenEnzymeCaches
    : public enzyme::impl::FlattenEnzymeCachesPassBase<FlattenEnzymeCaches> {
  void runOnOperation() override {
    SetVector<enzyme::MultidimensionalAllocInterface> allocations;
    getOperation()->walk([&allocations](enzyme::StoreOp storeOp) {
      Operation *alloc = storeOp.getMemref().getDefiningOp();
      if (!alloc)
        return;

      MemRefType oldType = storeOp.getMemref().getType();
      if (oldType.getRank() <= 1)
        return;

      if (auto allocOp =
              dyn_cast<enzyme::MultidimensionalAllocInterface>(alloc)) {
        allocations.insert(allocOp);
      }
    });

    for (auto alloc : llvm::make_early_inc_range(allocations)) {
      auto oldType = cast<MemRefType>(alloc->getResultTypes().front());
      ImplicitLocOpBuilder abuilder(alloc->getLoc(), alloc);
      Value flatAlloc = createFlatAlloc(oldType, alloc, abuilder);
      alloc->replaceAllUsesWith(ValueRange(flatAlloc));
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
      // TODO add alignment
      storeOp.erase();
    });

    getOperation()->walk([](enzyme::LoadOp loadOp) {
      ImplicitLocOpBuilder lbuilder(loadOp.getLoc(), loadOp);
      Value flatIndex = computeFlatIndex(loadOp.getIndices(), loadOp.getSizes(),
                                         loadOp.getStaticSizes(), lbuilder);
      auto flatLoad =
          memref::LoadOp::create(lbuilder, loadOp.getMemref(), flatIndex);
      loadOp.replaceAllUsesWith(flatLoad.getResult());
      // TODO add alignment
      loadOp.erase();
    });

    // Trying to get lowering working, remember to delete this
    llvm::errs() << "***Deleting leftover placeholders***\n";

    getOperation()->walk([](enzyme::PlaceholderOp placeholder) {
      SmallVector<Operation *> frontier{placeholder};
      SetVector<Operation *> visited;
      while (!frontier.empty()) {
        Operation *curr = frontier.pop_back_val();
        visited.insert(curr);

        for (Operation *user : curr->getUsers()) {
          if (!visited.contains(user)) {
            frontier.push_back(user);
          }
        }
      }

      for (Operation *toDelete : llvm::reverse(visited)) {
        toDelete->erase();
      }
    });
  }
};
} // namespace
