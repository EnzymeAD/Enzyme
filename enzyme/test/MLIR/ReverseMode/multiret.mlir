// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s 

module {
  func.func @square(%x: f64, %y : i32, %z : f32) -> (f64, i32, f32) {
    %x2 = arith.mulf %x, %x : f64
    %y2 = arith.muli %y, %y : i32
    %z2 = arith.mulf %z, %z : f32
    return %x2, %y2, %z2 : f64, i32, f32
  }

  func.func @dsquare(%x: f64, %y : i32, %z : f32, %dx: f64, %dz : f32) -> (f64, f32) {
    %r:2 = enzyme.autodiff @square(%x, %y, %z, %dx, %dz) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_constnoneed>, #enzyme<activity enzyme_activenoneed>] } : (f64, i32, f32, f64, f32) -> (f64, f32)
    return %r#0, %r#1 : f64, f32
  }

  func.func @dsquare2(%x: f64, %y : i32, %z : f32, %dx: f64, %dz : f32) -> (f64, i32, f64, f32) {
    %r:4 = enzyme.autodiff @square(%x, %y, %z, %dx, %dz) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_activenoneed>] } : (f64, i32, f32, f64, f32) -> (f64, i32, f64, f32)
    return %r#0, %r#1, %r#2, %r#3 : f64, i32, f64, f32
  }
}

// CHECK:  func.func @dsquare(%arg0: f64, %arg1: i32, %arg2: f32, %arg3: f64, %arg4: f32) -> (f64, f32) {
// CHECK-NEXT:    %0:2 = call @diffesquare(%arg0, %arg1, %arg2, %arg3, %arg4) : (f64, i32, f32, f64, f32) -> (f64, f32)
// CHECK-NEXT:    return %0#0, %0#1 : f64, f32
// CHECK-NEXT:  }

// CHECK:  func.func @dsquare2(%arg0: f64, %arg1: i32, %arg2: f32, %arg3: f64, %arg4: f32) -> (f64, i32, f64, f32) {
// CHECK-NEXT:    %0:4 = call @diffesquare_0(%arg0, %arg1, %arg2, %arg3, %arg4) : (f64, i32, f32, f64, f32) -> (f64, i32, f64, f32)
// CHECK-NEXT:    return %0#0, %0#1, %0#2, %0#3 : f64, i32, f64, f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffesquare(%arg0: f64, %arg1: i32, %arg2: f32, %arg3: f64, %arg4: f32) -> (f64, f32) {
// CHECK-NEXT:    %0 = arith.mulf %arg4, %arg2 : f32
// CHECK-NEXT:    %1 = arith.addf %0, %0 : f32
// CHECK-NEXT:    %2 = arith.mulf %arg3, %arg0 : f64
// CHECK-NEXT:    %3 = arith.addf %2, %2 : f64
// CHECK-NEXT:    return %3, %1 : f64, f32
// CHECK-NEXT:  }


// CHECK:  func.func private @diffesquare_0(%arg0: f64, %arg1: i32, %arg2: f32, %arg3: f64, %arg4: f32) -> (f64, i32, f64, f32) {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:    %1 = arith.muli %arg1, %arg1 : i32
// CHECK-NEXT:    %2 = arith.mulf %arg4, %arg2 : f32
// CHECK-NEXT:    %3 = arith.addf %2, %2 : f32
// CHECK-NEXT:    %4 = arith.mulf %arg3, %arg0 : f64
// CHECK-NEXT:    %5 = arith.addf %4, %4 : f64
// CHECK-NEXT:    return %0, %1, %5, %3 : f64, i32, f64, f32
// CHECK-NEXT:  }
