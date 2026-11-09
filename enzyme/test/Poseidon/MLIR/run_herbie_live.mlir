// REQUIRES: poseidon-mlir, herbie
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_mlir_opt --mlir-disable-threading \
// RUN:   --poseidon-rewrite="dump-fpcore run-herbie herbie-cache=%t/c" \
// RUN:   %s 2>&1 | FileCheck %s

// The Herbie configured at build time, run for real. Which candidate is
// chosen depends on what this Herbie returns, so only its answer is checked:
// that it improved the core, with expm1 among its candidates.

// CHECK:     (FPCore (x0) :name "sqrt_expm1/0" :precision binary64 (sqrt (- (exp x0) 1)))
// CHECK:     herbie: imp-start, {{[1-9][0-9]*}} candidate(s)
// CHECK:     expm1
// CHECK:     chose
// CHECK:     func.func @sqrt_expm1
// CHECK:     return
func.func @sqrt_expm1(%x: tensor<16xf64>) -> tensor<16xf64> {
  %one = arith.constant dense<1.0> : tensor<16xf64>
  %0 = math.exp %x : tensor<16xf64>
  %1 = arith.subf %0, %one : tensor<16xf64>
  %2 = math.sqrt %1 : tensor<16xf64>
  return %2 : tensor<16xf64>
}
