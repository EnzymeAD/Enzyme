// RUN: %eopt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {
  func.func @main(%i: i32, %x: f32) -> (i32 {sdy.sharding = 1}, f32 {sdy.sharding = 2}) {
    %1 = arith.mulf %x, %x : f32
    %c2 = arith.constant 2 : i32
    %i2 = arith.muli %c2, %i : i32
    return %i2, %1 : i32, f32
  }

  func.func @main_ad(%arg0: i32, %arg1: f32, %arg2: f32) -> (f32, f32) {
    %r:2 = enzyme.autodiff @main(%arg0, %arg1, %arg2) {
      activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_constnoneed>, #enzyme<activity enzyme_active>]
    } : (i32, f32, f32) -> (f32, f32)
    return %r#0, %r#1 : f32, f32
  }
}

// CHECK:  func.func private @diffemain(%arg0: i32, %arg1: f32, %arg2: f32) -> (f32 {sdy.sharding = 2 : i64}, f32) {
// CHECK:    %0 = arith.mulf %arg1, %arg1 : f32
// CHECK:    %1 = arith.mulf %arg2, %arg1 : f32
// CHECK:    %2 = arith.mulf %arg2, %arg1 : f32
// CHECK:    %3 = arith.addf %1, %2 : f32
// CHECK:    return %0, %3 : f32, f32
// CHECK:  }
