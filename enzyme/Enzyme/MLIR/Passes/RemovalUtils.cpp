//===- RemovalUtils.cpp - Utilities to remove Enzyme ops -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RemovalUtils.h"
#include "Interfaces/AutoDiffOpInterface.h"

mlir::Type mlir::enzyme::CacheInfo::batchType() {
  return mlir::enzyme::CacheInfo::batchType(mlir::ShapedType::kDynamic);
}

mlir::Type mlir::enzyme::CacheInfo::batchType(int64_t dim) {
  auto T = pushedValue().getType();

  if (auto TT = dyn_cast<mlir::TensorType>(T)) {
    SmallVector<int64_t> shape;
    shape.push_back(dim);
    shape.append(TT.getShape().begin(), TT.getShape().end());
    return TT.clone(shape);
  }

  return mlir::RankedTensorType::get({dim}, T);
}

mlir::LogicalResult mlir::enzyme::removeOpsWithinBlock(mlir::Block *block) {
  bool valid = true;

  for (auto &it : *block) {
    mlir::Operation *op = &it;
    if (auto iface = dyn_cast<mlir::enzyme::EnzymeOpsRemoverOpInterface>(op)) {
      valid &= iface.removeEnzymeOps().succeeded();
    }
  }

  return success(valid);
}
