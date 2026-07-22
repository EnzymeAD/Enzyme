// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s

func.func private @test_fillzero(%x: f32) -> f32 {
  %alloc = memref.alloc() : memref<f32>
  memref.store %x, %alloc[] : memref<f32>
  %ld = memref.load %alloc[] : memref<f32>
  return %ld : f32
}

func.func @dtest_fillzero(%x: f32, %dr: f32) -> f32 {
  %dx = enzyme.autodiff @test_fillzero(%x, %dr) {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f32, f32) -> f32
  return %dx : f32
}

// CHECK-LABEL:   func.func private @diffetest_fillzero(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<f32>
// CHECK:           enzyme.fill_zero %[[ALLOC_0]] : memref<f32>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[ALLOC_0]][] : memref<f32>
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG1]] : f32
// CHECK:           memref.store %[[ADDF_0]], %[[ALLOC_0]][] : memref<f32>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]][] : memref<f32>
// CHECK:           memref.store %[[CONSTANT_0]], %[[ALLOC_0]][] : memref<f32>
// CHECK:           return %[[LOAD_1]] : f32
// CHECK:         }
