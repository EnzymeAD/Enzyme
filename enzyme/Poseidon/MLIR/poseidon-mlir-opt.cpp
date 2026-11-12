//===- poseidon-mlir-opt.cpp - The poseidon-mlir-opt driver ---------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// mlir-opt with Poseidon's passes and the dialects they read and write.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::tensor::TensorDialect>();

  mlir::poseidon::registerPoseidonPasses();
  // For tests that clean up after the rewrite.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Poseidon MLIR optimizer driver", registry));
}
