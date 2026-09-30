// RUN: %eopt %s --enzyme-batch --canonicalize --inline | FileCheck %s

module @reactant_batch_fwd attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"*_broadcast_scalar"(%arg0: tensor<f64> {enzymexla.memory_effects = []}, %arg1: tensor<f64> {enzymexla.memory_effects = []}) -> (tensor<f64>, tensor<f64>, tensor<f64>) attributes {enzymexla.memory_effects = []} {
    %0 = arith.mulf %arg0, %arg1 : tensor<f64>
    return %0, %arg0, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
  }
  func.func private @identity_broadcast_scalar(%arg0: tensor<f64> {enzymexla.memory_effects = []}) -> tensor<f64> attributes {enzymexla.memory_effects = []} {
    return %arg0 : tensor<f64>
  }
  func.func private @sin_broadcast_scalar(%arg0: tensor<f64> {enzymexla.memory_effects = []}) -> (tensor<f64>, tensor<f64>) attributes {enzymexla.memory_effects = []} {
    %0 = math.sin %arg0 : tensor<f64>
    return %0, %arg0 : tensor<f64>, tensor<f64>
  }
  func.func private @"*_broadcast_scalar_1"(%arg0: tensor<f64> {enzymexla.memory_effects = []}, %arg1: tensor<f64> {enzymexla.memory_effects = []}) -> (tensor<f64>, tensor<f64>, tensor<f64>) attributes {enzymexla.memory_effects = []} {
    %0 = arith.mulf %arg0, %arg1 : tensor<f64>
    return %0, %arg0, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
  }
  func.func private @"Const{typeof(batch_extract_rhs)}_autodiff"(%arg0: tensor<10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> (tensor<10xf64>, tensor<10xf64>, tensor<10xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0:3 = enzyme.batch @"*_broadcast_scalar"(%arg0, %arg0) {batch_shape = array<i64: 10>} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>, tensor<10xf64>)
    %1 = enzyme.batch @identity_broadcast_scalar(%arg0) {batch_shape = array<i64: 10>} : (tensor<10xf64>) -> tensor<10xf64>
    %2:2 = enzyme.batch @sin_broadcast_scalar(%1) {batch_shape = array<i64: 10>} : (tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    %3:3 = enzyme.batch @"*_broadcast_scalar_1"(%2#0, %1) {batch_shape = array<i64: 10>} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>, tensor<10xf64>)
    return %0#0, %3#0, %1 : tensor<10xf64>, tensor<10xf64>, tensor<10xf64>
  }
  func.func private @"unbatched_#batch_fwd##6"(%arg0: tensor<10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: tensor<10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> (tensor<10xf64>, tensor<10xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0:2 = enzyme.fwddiff @"Const{typeof(batch_extract_rhs)}_autodiff"(%arg0, %arg1) {activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_constnoneed>]} : (tensor<10xf64>, tensor<10xf64>) -> (tensor<10xf64>, tensor<10xf64>)
    return %0#0, %0#1 : tensor<10xf64>, tensor<10xf64>
  }
  func.func @main(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg1: tensor<10x10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 3 : i32}) -> (tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %2:2 = enzyme.batch @"unbatched_#batch_fwd##6"(%arg0, %arg1) {batch_shape = array<i64: 10>} : (tensor<10x10xf64>, tensor<10x10xf64>) -> (tensor<10x10xf64>, tensor<10x10xf64>)
    return %2#0, %2#1, %arg0, %arg1 : tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg1: tensor<10x10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 3 : i32}) -> (tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:    %0:2 = enzyme.fwddiff @"batched_Const{typeof(batch_extract_rhs)}_autodiff"(%arg0, %arg1) {activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_constnoneed>]} : (tensor<10x10xf64>, tensor<10x10xf64>) -> (tensor<10x10xf64>, tensor<10x10xf64>)
// CHECK-NEXT:    return %0#0, %0#1, %arg0, %arg1 : tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>
// CHECK-NEXT:  }
// CHECK:  func.func private @"batched_Const{typeof(batch_extract_rhs)}_autodiff"(%arg0: tensor<10x10xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> (tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg0 : tensor<10x10xf64>
// CHECK-NEXT:    %1 = math.sin %arg0 : tensor<10x10xf64>
// CHECK-NEXT:    %2 = arith.mulf %1, %arg0 : tensor<10x10xf64>
// CHECK-NEXT:    return %0, %2, %arg0 : tensor<10x10xf64>, tensor<10x10xf64>, tensor<10x10xf64>
// CHECK-NEXT:  }
