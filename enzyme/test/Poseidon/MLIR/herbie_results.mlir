// REQUIRES: poseidon-mlir
// RUN: %poseidon_mlir_opt --mlir-disable-threading \
// RUN:   --poseidon-rewrite="dump-fpcore herbie-results=%S/Inputs/herbie_results.json" \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="herbie-results=%S/Inputs/no_such_file.json" \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING

// Herbie's answers are read from a results.json in the format a real run
// writes, matched to each core by its :name, and parsed. Nothing is rewritten
// yet.

// MISSING: error: poseidon-rewrite: herbie-results {{.*}}no_such_file.json: cannot read

// The best rewrite first, then the alternatives, each with Herbie's error in
// bits; precisions are dropped when printed.
// CHECK-LABEL: (FPCore (x0) :name "log_one_plus/0"
// CHECK-NEXT:    herbie: imp-start, 2 candidate(s), original error 29.4 bits
// CHECK-NEXT:      0: 0.1 bits, cost 0.6: (log1p x0)
// CHECK-NEXT:      1: 29.4 bits, cost 1: (log (+ 1 x0))
func.func @log_one_plus(%x: tensor<8xf64>) -> tensor<8xf64> {
  %one = arith.constant dense<1.0> : tensor<8xf64>
  %0 = arith.addf %one, %x : tensor<8xf64>
  %1 = math.log %0 : tensor<8xf64>
  return %1 : tensor<8xf64>
}

// A rational literal.
// CHECK-LABEL: (FPCore (x0 x1) :name "half_sum/0"
// CHECK-NEXT:    herbie: imp-start, 1 candidate(s), original error 1.2 bits
// CHECK-NEXT:      0: 0.4 bits, cost 0.7: (* (+ x0 x1) 1/2)
func.func @half_sum(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %half = arith.constant dense<0.5> : tensor<4xf64>
  %0 = arith.mulf %x, %half : tensor<4xf64>
  %1 = arith.mulf %y, %half : tensor<4xf64>
  %2 = arith.addf %0, %1 : tensor<4xf64>
  return %2 : tensor<4xf64>
}

// An approx wrapper reads as the expression it computes; an expression that
// does not parse is reported, not dropped silently.
// CHECK-LABEL: (FPCore (x0) :name "expm1_over/0"
// CHECK-NEXT:    herbie: imp-start, 3 candidate(s), original error 33 bits
// CHECK-NEXT:      0: 0.2 bits, cost 0.9: (/ (expm1 x0) x0)
// CHECK-NEXT:      1: 12 bits, cost 0.5: (+ 1 (* x0 1/2))
// CHECK-NEXT:      2: 5 bits, cost 0.8: unparseable (expression ends early): (/.f64 (expm1.f64 x0) x0
func.func @expm1_over(%x: tensor<4xf64>) -> tensor<4xf64> {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  %2 = arith.divf %1, %x : tensor<4xf64>
  return %2 : tensor<4xf64>
}

// A branch on the input: Herbie's regimes.
// CHECK-LABEL: (FPCore (x0) :name "branch/0"
// CHECK-NEXT:    herbie: imp-start, 1 candidate(s), original error 20 bits
// CHECK-NEXT:      0: 0.3 bits, cost 1.1: (if (<= x0 -1/10) (- (exp x0) 1) (expm1 x0))
func.func @branch(%x: tensor<4xf64>) -> tensor<4xf64> {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  return %1 : tensor<4xf64>
}

// Herbie answered with no expression, and a core it never saw.
// CHECK-LABEL: (FPCore (x0) :name "nothing/0"
// CHECK-NEXT:    herbie: timeout, 0 candidate(s), original error 15 bits
// CHECK-LABEL: (FPCore (x0) :name "missing/0"
// CHECK-NEXT:    herbie: no result
func.func @nothing(%x: tensor<4xf64>) -> tensor<4xf64> {
  %0 = math.sin %x : tensor<4xf64>
  %1 = math.cos %0 : tensor<4xf64>
  return %1 : tensor<4xf64>
}
func.func @missing(%x: tensor<4xf64>) -> tensor<4xf64> {
  %0 = math.tan %x : tensor<4xf64>
  %1 = math.atan %0 : tensor<4xf64>
  return %1 : tensor<4xf64>
}
