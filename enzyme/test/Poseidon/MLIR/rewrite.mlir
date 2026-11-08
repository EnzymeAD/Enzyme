// REQUIRES: poseidon-mlir
// RUN: %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="herbie-results=%S/Inputs/rewrite_results.json" \
// RUN:   %s | FileCheck %s
// RUN: %poseidon_mlir_opt --mlir-disable-threading \
// RUN:   --poseidon-rewrite="dump-fpcore herbie-results=%S/Inputs/rewrite_results.json" \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WHY
// RUN: %poseidon_mlir_opt \
// RUN:   --poseidon-rewrite="seed=1 herbie-results=%S/Inputs/rewrite_results.json" \
// RUN:   %s | FileCheck %s --check-prefix=SEED1

// Each chain output is replaced by a candidate Herbie returned, chosen at
// random among the ones the pass can build, and the old ops are erased. The
// splat constants they used are left for -canonicalize to remove.

// CHECK-LABEL: func.func @log_one_plus
// CHECK-NOT:     math.log %
// CHECK:         %[[R:.*]] = math.log1p %arg0 : tensor<8xf64>
// CHECK-NOT:     math.log %
// CHECK:         return %[[R]]
func.func @log_one_plus(%x: tensor<8xf64>) -> tensor<8xf64> {
  %one = arith.constant dense<1.0> : tensor<8xf64>
  %0 = arith.addf %one, %x : tensor<8xf64>
  %1 = math.log %0 : tensor<8xf64>
  return %1 : tensor<8xf64>
}

// Both outputs are replaced; the old exp, used by the return and inside the
// chain, goes once neither needs it.
// CHECK-LABEL: func.func @two_outputs
// CHECK:         %[[E:.*]] = math.exp %arg0
// CHECK-NEXT:    %[[M:.*]] = math.expm1 %arg0
// CHECK-NEXT:    %[[S:.*]] = math.sqrt %[[M]]
// CHECK-NEXT:    return %[[E]], %[[S]]
func.func @two_outputs(%x: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  %2 = math.sqrt %1 : tensor<4xf64>
  return %0, %2 : tensor<4xf64>, tensor<4xf64>
}

// hypot has no MLIR op, so the other candidate is the only choice.
// WHY-LABEL: :name "hypot/0"
// WHY:         (hypot x0 x1) [cannot build: no MLIR op computes 'hypot']
// WHY:         chose (sqrt (fma x0 x0 (* x1 x1)))
// CHECK-LABEL: func.func @hypot
// CHECK:         %[[Y2:.*]] = arith.mulf %arg1, %arg1
// CHECK-NEXT:    %[[F:.*]] = math.fma %arg0, %arg0, %[[Y2]]
// CHECK-NEXT:    %[[R:.*]] = math.sqrt %[[F]]
// CHECK-NEXT:    return %[[R]]
func.func @hypot(%x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %0 = arith.mulf %x, %x : tensor<4xf64>
  %1 = arith.mulf %y, %y : tensor<4xf64>
  %2 = arith.addf %0, %1 : tensor<4xf64>
  %3 = math.sqrt %2 : tensor<4xf64>
  return %3 : tensor<4xf64>
}

// A candidate in another precision than the tensor's is not built.
// WHY-LABEL: :name "precision/0"
// WHY:         [cannot build: computes in f64, not f32]
// WHY:         no candidate can be built; left as is
// CHECK-LABEL: func.func @precision
// CHECK-NEXT:    %[[M:.*]] = arith.mulf %arg0, %arg0 : tensor<4xf32>
// CHECK-NEXT:    %[[A:.*]] = arith.addf %[[M]], %arg0 : tensor<4xf32>
// CHECK-NEXT:    return %[[A]]
func.func @precision(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.mulf %x, %x : tensor<4xf32>
  %1 = arith.addf %0, %x : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// The rewrite of an output goes where the output was computed, so it can only
// use the inputs defined before that: here x1 (the rsqrt) comes later.
// WHY-LABEL: :name "late_input/0"
// WHY:         [cannot build: x1 is defined after the chain's output]
// WHY:         chose (exp x0)
// CHECK-LABEL: func.func @late_input
// CHECK:         %[[E:.*]] = math.exp %arg0
// CHECK-NEXT:    %[[Q:.*]] = math.rsqrt %arg0
// CHECK:         %[[E2:.*]] = math.exp %arg0
// CHECK-NEXT:    %[[A:.*]] = arith.addf %[[E2]], %[[Q]]
// CHECK-NEXT:    return %[[E]], %[[A]]
func.func @late_input(%x: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %0 = math.exp %x : tensor<4xf64>
  %q = math.rsqrt %x : tensor<4xf64>
  %1 = arith.mulf %0, %0 : tensor<4xf64>
  %2 = arith.addf %1, %q : tensor<4xf64>
  return %0, %2 : tensor<4xf64>, tensor<4xf64>
}

// Three candidates that can all be built: the seed decides.
// CHECK-LABEL: func.func @random
// CHECK:         %[[M:.*]] = math.expm1 %arg0
// CHECK-NEXT:    %[[L:.*]] = math.log1p %[[M]]
// CHECK-NEXT:    return %[[L]]
// SEED1-LABEL: func.func @random
// SEED1:         %[[M:.*]] = math.expm1 %arg0
// SEED1-NEXT:    return %[[M]]
func.func @random(%x: tensor<4xf64>) -> tensor<4xf64> {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  return %1 : tensor<4xf64>
}

// A branch becomes a comparison and a select, element by element, and a
// fraction becomes a splat constant rounded to the tensor's precision.
// CHECK-LABEL: func.func @branch
// CHECK:         %[[C:.*]] = arith.constant dense<-1.000000e-01> : tensor<4xf64>
// CHECK-NEXT:    %[[LE:.*]] = arith.cmpf ole, %arg0, %[[C]] : tensor<4xf64>
// CHECK-NEXT:    %[[E:.*]] = math.exp %arg0
// CHECK-NEXT:    %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf64>
// CHECK-NEXT:    %[[SUB:.*]] = arith.subf %[[E]], %[[ONE]]
// CHECK-NEXT:    %[[M:.*]] = math.expm1 %arg0
// CHECK-NEXT:    %[[R:.*]] = arith.select %[[LE]], %[[SUB]], %[[M]] : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:    return %[[R]]
func.func @branch(%x: tensor<4xf64>) -> tensor<4xf64> {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  return %1 : tensor<4xf64>
}

// CHECK-LABEL: func.func @third
// CHECK:         %[[C:.*]] = arith.constant dense<0.333333343> : tensor<4xf32>
// CHECK-NEXT:    %[[R:.*]] = arith.mulf %arg0, %[[C]] : tensor<4xf32>
// CHECK-NEXT:    return %[[R]]
func.func @third(%x: tensor<4xf32>) -> tensor<4xf32> {
  %three = arith.constant dense<3.0> : tensor<4xf32>
  %0 = arith.mulf %x, %three : tensor<4xf32>
  %1 = arith.divf %0, %three : tensor<4xf32>
  %2 = arith.divf %1, %three : tensor<4xf32>
  return %2 : tensor<4xf32>
}
