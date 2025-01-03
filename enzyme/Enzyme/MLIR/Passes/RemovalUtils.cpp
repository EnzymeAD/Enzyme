//===- RemovalUtils.cpp - Utilities to remove Enzyme ops -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RemovalUtils.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include <cassert>

mlir::enzyme::CacheInfo
mlir::enzyme::CacheInfo::merge(mlir::enzyme::CacheInfo other) {
  assert(other.pushOp->getBlock() == pushOp->getBlock());
  assert(other.popOp->getBlock() == popOp->getBlock());

  enzyme::InitOp newInitOp;
  if (other.initOp->isBeforeInBlock(initOp)) {
    newInitOp = other.initOp;
    initOp.getResult().replaceAllUsesWith(newInitOp.getResult());
    initOp->erase();
  } else {
    newInitOp = initOp;
    other.initOp.getResult().replaceAllUsesWith(newInitOp.getResult());
    other.initOp->erase();
  }

  enzyme::PushOp newPushOp = pushOp;
  other.pushOp->erase();

  enzyme::PopOp newPopOp;
  if (other.popOp->isBeforeInBlock(popOp)) {
    newPopOp = other.popOp;
    popOp.getResult().replaceAllUsesWith(newPopOp.getResult());
    popOp->erase();
  } else {
    newPopOp = popOp;
    other.popOp.getResult().replaceAllUsesWith(newPopOp.getResult());
    other.popOp->erase();
  }

  CacheInfo newInfo{newInitOp};
  return newInfo;
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
