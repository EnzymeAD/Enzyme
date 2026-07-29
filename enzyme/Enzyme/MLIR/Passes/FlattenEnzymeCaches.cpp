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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

/// Is `sv` a "row slice" of `buf`, i.e. `buf[i, 0][1, n][1, 1]`, rank-reduced
/// to 1-D? Returns the slice's size as an OpFoldResult, or nullopt.
static std::optional<OpFoldResult> matchRowSlice(memref::SubViewOp sv,
                                                 MemRefType bufTy) {
  if (cast<MemRefType>(sv.getType()).getRank() != 1)
    return std::nullopt;
  auto offsets = sv.getMixedOffsets();
  auto sizes = sv.getMixedSizes();
  auto strides = sv.getMixedStrides();
  if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2)
    return std::nullopt;
  if (!isConstantIntValue(offsets[1], 0) || !isConstantIntValue(sizes[0], 1) ||
      !isConstantIntValue(strides[0], 1) || !isConstantIntValue(strides[1], 1))
    return std::nullopt;
  return sizes[1];
}

/// A rank-2 cache buffer whose *inner* dimension is dynamic cannot be lowered
/// under the bare-pointer calling convention used for GPU kernel arguments:
/// there is nowhere to put the (dynamic) inner stride. Such a buffer -- the
/// checkpoint/cache storage `memref<Nx?xT>` produced when caching a
/// dynamically-sized memref across a loop -- is only ever accessed as a row
/// slice `buf[slot, 0][1, n][1, 1]`, so flatten the allocation to 1-D and
/// rewrite each slice as an offset slice `flat[slot * inner][n][1]`. The result
/// type is unchanged (`memref<?xT, strided<[1], offset: ?>>`), which the
/// bare-pointer conversion *does* accept, so no user needs updating.
///
/// Restricted to rank 2: for higher ranks the row slice would itself be
/// multi-dimensional and could not keep its type across the rewrite.
static void flattenRowSlicedCaches(Operation *root) {
  SmallVector<memref::AllocOp> allocs;
  root->walk([&](memref::AllocOp alloc) {
    MemRefType MT = alloc.getType();
    if (MT.getRank() != 2 || !MT.getLayout().isIdentity())
      return;
    if (!MT.isDynamicDim(1) || MT.isDynamicDim(0))
      return;
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user))
        continue;
      auto sv = dyn_cast<memref::SubViewOp>(user);
      if (!sv || !matchRowSlice(sv, MT))
        return;
    }
    allocs.push_back(alloc);
  });

  for (memref::AllocOp alloc : allocs) {
    MemRefType MT = alloc.getType();
    // The single dynamic size operand is the inner extent, which doubles as
    // the row stride of the flattened buffer.
    Value inner = alloc.getDynamicSizes().front();

    ImplicitLocOpBuilder b(alloc.getLoc(), alloc);
    Value rows = arith::ConstantIndexOp::create(b, MT.getDimSize(0));
    Value flatSize = arith::MulIOp::create(b, rows, inner);
    auto flatTy = MemRefType::get({ShapedType::kDynamic}, MT.getElementType(),
                                  MemRefLayoutAttrInterface{},
                                  MT.getMemorySpace());
    Value flat = memref::AllocOp::create(b, flatTy, ValueRange{flatSize},
                                         alloc.getAlignmentAttr());

    for (Operation *user : llvm::make_early_inc_range(alloc->getUsers())) {
      auto sv = dyn_cast<memref::SubViewOp>(user);
      if (!sv) {
        // memref.dealloc -- retype in place.
        user->setOperand(0, flat);
        continue;
      }
      OpFoldResult size = *matchRowSlice(sv, MT);
      ImplicitLocOpBuilder sb(sv.getLoc(), sv);
      Value slot = getValueOrCreateConstantIndexOp(sb, sv.getLoc(),
                                                   sv.getMixedOffsets()[0]);
      Value offset = arith::MulIOp::create(sb, slot, inner);
      Value newSv = memref::SubViewOp::create(
          sb, cast<MemRefType>(sv.getType()), flat, ArrayRef<OpFoldResult>{offset},
          ArrayRef<OpFoldResult>{size},
          ArrayRef<OpFoldResult>{sb.getIndexAttr(1)});
      sv.getResult().replaceAllUsesWith(newSv);
      sv.erase();
    }
    alloc.erase();
  }
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

    flattenRowSlicedCaches(getOperation());
  }
};
} // namespace
