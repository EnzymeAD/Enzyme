//===- Passes.td - Poseidon MLIR passes -----------------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MLIR_PASSES
#define POSEIDON_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def PoseidonRewritePass : Pass<"poseidon-rewrite", "func::FuncOp"> {
  let summary = "Rewrite chains of elementwise floating-point tensor ops";
  let description = [{
    Finds chains of elementwise arith and math ops on floating-point tensors,
    writes each chain as an FPCore expression, asks Herbie for a more accurate
    expression, and replaces the chain with the one Herbie returns.
  }];
  let dependentDialects = ["arith::ArithDialect", "math::MathDialect"];
  let options = [
    Option<"dumpFPCore", "dump-fpcore", "bool", /*default=*/"false",
           "Print each chain's inputs and FPCore expressions to stderr">,
    Option<"herbieResults", "herbie-results", "std::string",
           /*default=*/"\"\"",
           "A results.json from an earlier Herbie run, read instead of "
           "running Herbie">,
    Option<"seed", "seed", "uint64_t", /*default=*/"0",
           "Seed of the random choice among Herbie's candidates">,
  ];
}

#endif // POSEIDON_MLIR_PASSES
