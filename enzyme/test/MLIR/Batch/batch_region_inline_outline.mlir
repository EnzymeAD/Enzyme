// RUN: %eopt --inline-enzyme-regions --split-input-file %s | FileCheck --check-prefix=INLINE %s
// RUN: %eopt --inline-enzyme-regions --outline-enzyme-regions --split-input-file %s | FileCheck --check-prefix=OUTLINE %s
// RUN: %eopt --inline-enzyme-regions --outline-enzyme-regions --enzyme-batch --split-input-file %s | FileCheck --check-prefix=BATCH %s

func.func @square(%x: tensor<3xf64>) -> tensor<3xf64> {
  %y = arith.mulf %x, %x : tensor<3xf64>
  return %y : tensor<3xf64>
}

func.func @batch_inline_outline(%x: tensor<4x3xf64>) -> tensor<4x3xf64> {
  %r = enzyme.batch @square(%x) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
  return %r : tensor<4x3xf64>
}

// INLINE-LABEL: func.func @batch_inline_outline(%arg0: tensor<4x3xf64>) -> tensor<4x3xf64> {
// INLINE-NEXT:    %0 = enzyme.batch_region(%arg0) {
// INLINE-NEXT:    ^bb0(%arg1: tensor<3xf64>):
// INLINE-NEXT:      %1 = arith.mulf %arg1, %arg1 : tensor<3xf64>
// INLINE-NEXT:      enzyme.yield %1 : tensor<3xf64>
// INLINE-NEXT:    } {{.*}}batch_shape = array<i64: 4>{{.*}} : (tensor<4x3xf64>) -> tensor<4x3xf64>
// INLINE-NEXT:    return %0 : tensor<4x3xf64>
// INLINE-NEXT:  }

// OUTLINE-LABEL: func.func @batch_inline_outline(%arg0: tensor<4x3xf64>) -> tensor<4x3xf64> {
// OUTLINE-NEXT:    %0 = enzyme.batch @batch_inline_outline_to_batch0(%arg0) {{.*}}batch_shape = array<i64: 4>{{.*}} : (tensor<4x3xf64>) -> tensor<4x3xf64>
// OUTLINE-NEXT:    return %0 : tensor<4x3xf64>
// OUTLINE-NEXT:  }
// OUTLINE-LABEL: func.func private @batch_inline_outline_to_batch0(%arg0: tensor<3xf64>) -> tensor<3xf64> {
// OUTLINE-NEXT:    %0 = arith.mulf %arg0, %arg0 : tensor<3xf64>
// OUTLINE-NEXT:    return %0 : tensor<3xf64>
// OUTLINE-NEXT:  }

// BATCH-LABEL: func.func @batch_inline_outline(%arg0: tensor<4x3xf64>) -> tensor<4x3xf64> {
// BATCH-NEXT:    %0 = call @batched_batch_inline_outline_to_batch0(%arg0) : (tensor<4x3xf64>) -> tensor<4x3xf64>
// BATCH-NEXT:    return %0 : tensor<4x3xf64>
// BATCH-NEXT:  }
// BATCH-LABEL: func.func private @batched_batch_inline_outline_to_batch0(%arg0: tensor<4x3xf64>) -> tensor<4x3xf64> {
// BATCH-NEXT:    %0 = arith.mulf %arg0, %arg0 : tensor<4x3xf64>
// BATCH-NEXT:    return %0 : tensor<4x3xf64>
// BATCH-NEXT:  }

// -----

func.func @batch_outline_free_var(%x: tensor<4x3xf64>, %scale: tensor<3xf64>) -> tensor<4x3xf64> {
  %r = enzyme.batch_region(%x) {
  ^bb0(%arg0: tensor<3xf64>):
    %y = arith.mulf %arg0, %scale : tensor<3xf64>
    enzyme.yield %y : tensor<3xf64>
  } <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
  return %r : tensor<4x3xf64>
}

// OUTLINE-LABEL: func.func @batch_outline_free_var(%arg0: tensor<4x3xf64>, %arg1: tensor<3xf64>) -> tensor<4x3xf64> {
// OUTLINE-NEXT:    %0 = "enzyme.broadcast"(%arg1) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// OUTLINE-NEXT:    %1 = enzyme.batch @batch_outline_free_var_to_batch0(%arg0, %0) {{.*}}batch_shape = array<i64: 4>{{.*}} : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x3xf64>
// OUTLINE-NEXT:    return %1 : tensor<4x3xf64>
// OUTLINE-NEXT:  }
// OUTLINE-LABEL: func.func private @batch_outline_free_var_to_batch0(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// OUTLINE-NEXT:    %0 = arith.mulf %arg0, %arg1 : tensor<3xf64>
// OUTLINE-NEXT:    return %0 : tensor<3xf64>
// OUTLINE-NEXT:  }

// BATCH-LABEL: func.func @batch_outline_free_var(%arg0: tensor<4x3xf64>, %arg1: tensor<3xf64>) -> tensor<4x3xf64> {
// BATCH-NEXT:    %0 = "enzyme.broadcast"(%arg1) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// BATCH-NEXT:    %1 = call @batched_batch_outline_free_var_to_batch0(%arg0, %0) : (tensor<4x3xf64>, tensor<4x3xf64>) -> tensor<4x3xf64>
// BATCH-NEXT:    return %1 : tensor<4x3xf64>
// BATCH-NEXT:  }
// BATCH-LABEL: func.func private @batched_batch_outline_free_var_to_batch0(%arg0: tensor<4x3xf64>, %arg1: tensor<4x3xf64>) -> tensor<4x3xf64> {
// BATCH-NEXT:    %0 = arith.mulf %arg0, %arg1 : tensor<4x3xf64>
// BATCH-NEXT:    return %0 : tensor<4x3xf64>
// BATCH-NEXT:  }
