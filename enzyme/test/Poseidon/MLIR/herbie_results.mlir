// REQUIRES: poseidon-mlir
// RUN: %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="seed=2 herbie-results=%S/Inputs/herbie_results.json" \
// RUN:   %s | FileCheck %s
// RUN: not %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="herbie-results=%S/Inputs/no_such_file.json" \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING

// Herbie's answers are read from a results.json in the format a real run
// writes, and matched to each core by its :name. The seed is one that picks
// the candidates checked here; how candidates are built is tested in
// rewrite.mlir.

// MISSING: error: poseidon-rewrite: herbie-results {{.*}}no_such_file.json: cannot read

// The best rewrite and the alternatives are all candidates.
// CHECK-LABEL: func.func @log_one_plus
// CHECK:         %[[R:.*]] = math.log1p %arg0
// CHECK-NEXT:    return %[[R]]
func.func @log_one_plus(%x: tensor<8xf64>) -> tensor<8xf64> {
  %one = arith.constant dense<1.0> : tensor<8xf64>
  %0 = arith.addf %one, %x : tensor<8xf64>
  %1 = math.log %0 : tensor<8xf64>
  return %1 : tensor<8xf64>
}

// A rational literal.
// CHECK-LABEL: func.func @half_sum
// CHECK:         %[[S:.*]] = arith.addf %arg0, %arg1
// CHECK-NEXT:    %[[H:.*]] = arith.constant dense<5.000000e-01> : tensor<4xf64>
// CHECK-NEXT:    %[[R:.*]] = arith.mulf %[[S]], %[[H]]
// CHECK-NEXT:    return %[[R]]
func.func @half_sum(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %half = arith.constant dense<0.5> : tensor<4xf64>
  %0 = arith.mulf %x, %half : tensor<4xf64>
  %1 = arith.mulf %y, %half : tensor<4xf64>
  %2 = arith.addf %0, %1 : tensor<4xf64>
  return %2 : tensor<4xf64>
}

// An approx wrapper reads as the expression it computes. The third candidate
// in the file does not parse, and is passed over.
// CHECK-LABEL: func.func @expm1_over
// CHECK:         %[[M:.*]] = math.expm1 %arg0
// CHECK-NEXT:    %[[R:.*]] = arith.divf %[[M]], %arg0
// CHECK-NEXT:    return %[[R]]
func.func @expm1_over(%x: tensor<4xf64>) -> tensor<4xf64> {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  %2 = arith.divf %1, %x : tensor<4xf64>
  return %2 : tensor<4xf64>
}

// A branch on the input: Herbie's regimes.
// CHECK-LABEL: func.func @branch
// CHECK:         %[[R:.*]] = arith.select
// CHECK-NEXT:    return %[[R]]
func.func @branch(%x: tensor<4xf64>) -> tensor<4xf64> {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  return %1 : tensor<4xf64>
}

// Herbie answered with no expression, and a core it never saw: both functions
// are left as they are.
// CHECK-LABEL: func.func @nothing
// CHECK-NEXT:    %[[S:.*]] = math.sin %arg0
// CHECK-NEXT:    %[[C:.*]] = math.cos %[[S]]
// CHECK-NEXT:    return %[[C]]
// CHECK-LABEL: func.func @missing
// CHECK-NEXT:    %[[T:.*]] = math.tan %arg0
// CHECK-NEXT:    %[[A:.*]] = math.atan %[[T]]
// CHECK-NEXT:    return %[[A]]
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
