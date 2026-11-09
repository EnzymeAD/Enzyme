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

    Herbie's answers come from a results.json of an earlier run
    (herbie-results), or from running Herbie on each function (run-herbie).
    Without either, the chains are only found.
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
    Option<"runHerbie", "run-herbie", "bool", /*default=*/"false",
           "Run Herbie on each function's chains (ignored with "
           "herbie-results)">,
    Option<"herbieBinary", "herbie-binary", "std::string",
           /*default=*/"\"\"",
           "The Herbie executable (empty: the one configured at build time)">,
    Option<"herbiePlatform", "herbie-platform", "std::string",
           /*default=*/"\"\"",
           "A Herbie platform: a .rkt file, or a name compiled into Herbie "
           "(empty: Herbie's default)">,
    Option<"herbieTimeout", "herbie-timeout", "unsigned", /*default=*/"1800",
           "Seconds Herbie may spend on one function (0: no limit)">,
    Option<"herbieThreads", "herbie-threads", "int", /*default=*/"8",
           "Herbie worker threads, one process each">,
    Option<"herbieSeed", "herbie-seed", "unsigned", /*default=*/"239778888",
           "Herbie's own random seed">,
    Option<"herbieCache", "herbie-cache", "std::string",
           /*default=*/"\"./.poseidon\"",
           "Directory of cached Herbie results (empty: no cache)">,
  ];
}

#endif // POSEIDON_MLIR_PASSES
