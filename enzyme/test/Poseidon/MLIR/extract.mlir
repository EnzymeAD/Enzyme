// REQUIRES: poseidon-mlir
// RUN: %poseidon_mlir_opt --mlir-disable-threading \
// RUN:   --poseidon-rewrite=dump-fpcore %s -o /dev/null 2>&1 | FileCheck %s

// Chains are found and written as FPCore; nothing is rewritten yet. Threading
// is off so that the functions print in order.

// Two ops passing a value: one chain, three inputs, one output.
// CHECK-LABEL: @axpy chain 0: 2 ops, 1 output(s)
// CHECK-NEXT:    x0 = %arg0
// CHECK-NEXT:    x1 = %arg1
// CHECK-NEXT:    x2 = %arg2
// CHECK-NEXT:    (FPCore (x0 x1 x2) :name "axpy/0" :precision binary64 (+ (* x0 x1) x2))
func.func @axpy(%a: tensor<4xf64>, %x: tensor<4xf64>, %y: tensor<4xf64>) -> tensor<4xf64> {
  %0 = arith.mulf %a, %x : tensor<4xf64>
  %1 = arith.addf %0, %y : tensor<4xf64>
  return %1 : tensor<4xf64>
}

// A splat constant is a number in the expression, not an input.
// CHECK-LABEL: @log_one_plus chain 0: 2 ops, 1 output(s)
// CHECK-NEXT:    x0 = %arg0
// CHECK-NEXT:    (FPCore (x0) :name "log_one_plus/0" :precision binary64 (log (+ 1 x0)))
func.func @log_one_plus(%x: tensor<8xf64>) -> tensor<8xf64> {
  %one = arith.constant dense<1.0> : tensor<8xf64>
  %0 = arith.addf %one, %x : tensor<8xf64>
  %1 = math.log %0 : tensor<8xf64>
  return %1 : tensor<8xf64>
}

// f32 tensors are searched in binary32. A value used twice is written twice.
// CHECK-LABEL: @square_minus chain 0: 2 ops, 1 output(s)
// CHECK-NEXT:    x0 = %arg0
// CHECK-NEXT:    x1 = %arg1
// CHECK-NEXT:    (FPCore (x0 x1) :name "square_minus/0" :precision binary32 (* (- x0 x1) (- x0 x1)))
func.func @square_minus(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = arith.subf %x, %y : tensor<2x3xf32>
  %1 = arith.mulf %0, %0 : tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// A value the chain uses and also returns is an output of its own.
// CHECK-LABEL: @two_outputs chain 0: 3 ops, 2 output(s)
// CHECK-NEXT:    x0 = %arg0
// CHECK-NEXT:    (FPCore (x0) :name "two_outputs/0" :precision binary64 (exp x0))
// CHECK-NEXT:    (FPCore (x0) :name "two_outputs/1" :precision binary64 (sqrt (- (exp x0) 1)))
func.func @two_outputs(%x: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %one = arith.constant dense<1.0> : tensor<4xf64>
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.subf %0, %one : tensor<4xf64>
  %2 = math.sqrt %1 : tensor<4xf64>
  return %0, %2 : tensor<4xf64>, tensor<4xf64>
}

// An op the pass does not know (math.rsqrt) splits two chains, and its result
// is an input of the second one.
// CHECK-LABEL: @split chain 0: 2 ops, 1 output(s)
// CHECK-NEXT:    x0 = %arg0
// CHECK-NEXT:    (FPCore (x0) :name "split/0" :precision binary64 (* (+ x0 x0) x0))
// CHECK-LABEL: @split chain 1: 2 ops, 1 output(s)
// CHECK-NEXT:    x0 = %{{.*}}
// CHECK-NEXT:    x1 = %arg0
// CHECK-NEXT:    (FPCore (x0 x1) :name "split/1" :precision binary64 (/ (sin x0) x1))
func.func @split(%x: tensor<4xf64>) -> tensor<4xf64> {
  %0 = arith.addf %x, %x : tensor<4xf64>
  %1 = arith.mulf %0, %x : tensor<4xf64>
  %2 = math.rsqrt %1 : tensor<4xf64>
  %3 = math.sin %2 : tensor<4xf64>
  %4 = arith.divf %3, %x : tensor<4xf64>
  return %4 : tensor<4xf64>
}

// Not chains: one op alone, scalars, and integer tensors.
// CHECK-NOT: @not_chains
func.func @not_chains(%x: tensor<4xf64>, %s: f64, %i: tensor<4xi32>) -> (tensor<4xf64>, f64, tensor<4xi32>) {
  %0 = math.exp %x : tensor<4xf64>
  %1 = arith.mulf %s, %s : f64
  %2 = arith.addf %1, %s : f64
  %3 = arith.addi %i, %i : tensor<4xi32>
  %4 = arith.muli %3, %i : tensor<4xi32>
  return %0, %2, %4 : tensor<4xf64>, f64, tensor<4xi32>
}
