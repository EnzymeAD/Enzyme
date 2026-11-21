// RUN: %eopt --enzyme-batch --split-input-file %s | FileCheck %s

// Simple single-block batch_region with multi-dim batch_shape.
func.func @simple_batch_region(%x: tensor<10x2x3xf64>) -> tensor<10x2x3xf64> {
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    %cst = arith.constant dense<2.1> : tensor<3xf64>
    %y = arith.addf %arg0, %cst : tensor<3xf64>
    enzyme.yield %y : tensor<3xf64>
  } <{batch_shape = array<i64: 10, 2>}> : (tensor<10x2x3xf64>) -> tensor<10x2x3xf64>
  return %r : tensor<10x2x3xf64>
}

// CHECK-LABEL: func.func @simple_batch_region(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x2x3xf64>) -> tensor<10x2x3xf64> {
// CHECK-NEXT:    %[[CST:.*]] = arith.constant dense<2.100000e+00> : tensor<10x2x3xf64>
// CHECK-NEXT:    %[[ADD:.*]] = arith.addf %[[ARG0]], %[[CST]] : tensor<10x2x3xf64>
// CHECK-NEXT:    return %[[ADD]] : tensor<10x2x3xf64>
// CHECK-NEXT:  }

// -----

// Captured free variables (both ranked tensor and scalar) are broadcast.
func.func @captured_free_vars(%x: tensor<4x3xf64>, %scale: tensor<3xf64>, %bias: f64) -> (tensor<4x3xf64>, tensor<4xf64>) {
  %r:2 = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    %scaled = arith.mulf %arg0, %scale : tensor<3xf64>
    %b2 = arith.addf %bias, %bias : f64
    enzyme.yield %scaled, %b2 : tensor<3xf64>, f64
  } <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> (tensor<4x3xf64>, tensor<4xf64>)
  return %r#0, %r#1 : tensor<4x3xf64>, tensor<4xf64>
}

// CHECK-LABEL: func.func @captured_free_vars(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x3xf64>, %[[ARG1:.*]]: tensor<3xf64>, %[[ARG2:.*]]: f64) -> (tensor<4x3xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %[[BSCALE:.*]] = "enzyme.broadcast"(%[[ARG1]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %[[BBIAS:.*]] = "enzyme.broadcast"(%[[ARG2]]) <{shape = array<i64: 4>}> : (f64) -> tensor<4xf64>
// CHECK-NEXT:    %[[MUL:.*]] = arith.mulf %[[ARG0]], %[[BSCALE]] : tensor<4x3xf64>
// CHECK-NEXT:    %[[ADD:.*]] = arith.addf %[[BBIAS]], %[[BBIAS]] : tensor<4xf64>
// CHECK-NEXT:    return %[[MUL]], %[[ADD]] : tensor<4x3xf64>, tensor<4xf64>
// CHECK-NEXT:  }

// -----

// Scalar inputs and function calls inside batch_region.
func.func private @helper(%a: tensor<3xf64>) -> tensor<3xf64> {
  %y = arith.mulf %a, %a : tensor<3xf64>
  return %y : tensor<3xf64>
}

func.func @call_inside_region(%x: tensor<4x3xf64>) -> tensor<4x3xf64> {
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    %y = func.call @helper(%arg0) : (tensor<3xf64>) -> tensor<3xf64>
    enzyme.yield %y : tensor<3xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
  return %r : tensor<4x3xf64>
}

// CHECK-LABEL: func.func @call_inside_region(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x3xf64>) -> tensor<4x3xf64> {
// CHECK-NEXT:    %[[CALL:.*]] = call @batched_helper(%[[ARG0]]) : (tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    return %[[CALL]] : tensor<4x3xf64>
// CHECK-NEXT:  }
// CHECK-LABEL: func.func private @batched_helper(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x3xf64>) -> tensor<4x3xf64> {
// CHECK-NEXT:    %[[MUL:.*]] = arith.mulf %[[ARG0]], %[[ARG0]] : tensor<4x3xf64>
// CHECK-NEXT:    return %[[MUL]] : tensor<4x3xf64>
// CHECK-NEXT:  }

// -----

// Nested batch_region inside batch_region.
func.func @nested_batch_region(%x: tensor<4x5x3xf64>) -> tensor<4x5x3xf64> {
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<5x3xf64>):
    %inner = enzyme.batch_region(%arg0) {
    ^bb0(%arg1: tensor<3xf64>):
      %y = arith.mulf %arg1, %arg1 : tensor<3xf64>
      enzyme.yield %y : tensor<3xf64>
    } <{batch_shape = array<i64: 5>}> : (tensor<5x3xf64>) -> tensor<5x3xf64>
    enzyme.yield %inner : tensor<5x3xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x5x3xf64>) -> tensor<4x5x3xf64>
  return %r : tensor<4x5x3xf64>
}

// CHECK-LABEL: func.func @nested_batch_region(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x5x3xf64>) -> tensor<4x5x3xf64> {
// CHECK-NEXT:    %[[MUL:.*]] = arith.mulf %[[ARG0]], %[[ARG0]] : tensor<4x5x3xf64>
// CHECK-NEXT:    return %[[MUL]] : tensor<4x5x3xf64>
// CHECK-NEXT:  }

// -----

// Multi-block batch_region with control flow.
func.func @multiblock_batch_region(%x: tensor<4x3xf64>, %bias: tensor<3xf64>) -> tensor<4x3xf64> {
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    %sum = arith.addf %arg0, %bias : tensor<3xf64>
    cf.br ^bb1(%sum : tensor<3xf64>)
  ^bb1(%arg1: tensor<3xf64>):
    %sq = arith.mulf %arg1, %arg1 : tensor<3xf64>
    enzyme.yield %sq : tensor<3xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
  return %r : tensor<4x3xf64>
}

// CHECK-LABEL: func.func @multiblock_batch_region(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x3xf64>, %[[ARG1:.*]]: tensor<3xf64>) -> tensor<4x3xf64> {
// CHECK-NEXT:    %[[BBIAS:.*]] = "enzyme.broadcast"(%[[ARG1]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    %[[CALL:.*]] = call @batched_multiblock_batch_region_region(%[[ARG0]], %[[BBIAS]]) : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:    return %[[CALL]] : tensor<4x3xf64>
// CHECK-NEXT:  }
// CHECK-LABEL: func.func private @batched_multiblock_batch_region_region(
// CHECK-SAME:    %[[A0:.*]]: tensor<4x3xf64>, %[[A1:.*]]: tensor<4x3xf64>) -> tensor<4x3xf64> {
// CHECK-NEXT:    %[[ADD:.*]] = arith.addf %[[A0]], %[[A1]] : tensor<4x3xf64>
// CHECK-NEXT:    cf.br ^bb1(%[[ADD]] : tensor<4x3xf64>)
// CHECK-NEXT:  ^bb1(%[[A2:.*]]: tensor<4x3xf64>):
// CHECK-NEXT:    %[[MUL:.*]] = arith.mulf %[[A2]], %[[A2]] : tensor<4x3xf64>
// CHECK-NEXT:    return %[[MUL]] : tensor<4x3xf64>
// CHECK-NEXT:  }

// -----

// Unit batch dimension (batch_shape = [1]) with free variable.
func.func @unit_batch_dim(%x: tensor<1x3xf64>, %scale: tensor<3xf64>) -> tensor<1x3xf64> {
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    %y = arith.mulf %arg0, %scale : tensor<3xf64>
    enzyme.yield %y : tensor<3xf64>
  } <{batch_shape = array<i64: 1>}> : (tensor<1x3xf64>) -> tensor<1x3xf64>
  return %r : tensor<1x3xf64>
}

// CHECK-LABEL: func.func @unit_batch_dim(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x3xf64>, %[[ARG1:.*]]: tensor<3xf64>) -> tensor<1x3xf64> {
// CHECK-NEXT:    %[[BSCALE:.*]] = "enzyme.broadcast"(%[[ARG1]]) <{shape = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:    %[[MUL:.*]] = arith.mulf %[[ARG0]], %[[BSCALE]] : tensor<1x3xf64>
// CHECK-NEXT:    return %[[MUL]] : tensor<1x3xf64>
// CHECK-NEXT:  }

// -----

// enzyme.autodiff inside enzyme.batch_region batches the differentiated callee.
func.func private @square(%arg0: tensor<12xf64>) -> tensor<12xf64> {
  %0 = arith.mulf %arg0, %arg0 : tensor<12xf64>
  return %0 : tensor<12xf64>
}

func.func @autodiff_inside_batch_region(%x: tensor<4x12xf64>, %dy: tensor<4x12xf64>) -> tensor<4x12xf64> {
  %r = enzyme.batch_region(%x, %dy) {
  ^bb0(%arg0: tensor<12xf64>, %arg1: tensor<12xf64>):
    %0 = enzyme.autodiff @square(%arg0, %arg1) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<12xf64>, tensor<12xf64>) -> tensor<12xf64>
    enzyme.yield %0 : tensor<12xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
  return %r : tensor<4x12xf64>
}

// CHECK-LABEL: func.func @autodiff_inside_batch_region(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x12xf64>, %[[ARG1:.*]]: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:    %[[AD:.*]] = enzyme.autodiff @batched_square(%[[ARG0]], %[[ARG1]]) {{.*}}activity = [#enzyme.activity<enzyme_active>]{{.*}} : (tensor<4x12xf64>, tensor<4x12xf64>) -> tensor<4x12xf64>
// CHECK-NEXT:    return %[[AD]] : tensor<4x12xf64>
// CHECK-NEXT:  }
// CHECK-LABEL: func.func private @batched_square(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x12xf64>) -> tensor<4x12xf64> {
// CHECK-NEXT:    %[[MUL:.*]] = arith.mulf %[[ARG0]], %[[ARG0]] : tensor<4x12xf64>
// CHECK-NEXT:    return %[[MUL]] : tensor<4x12xf64>
// CHECK-NEXT:  }
