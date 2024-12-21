//===- RemovalUtils.cpp - Utilities to remove Enzyme ops -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RemovalUtils.h"
#include "Interfaces/AutoDiffOpInterface.h"

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
