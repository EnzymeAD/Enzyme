// RUN: %eopt %s --enzyme --canonicalize --lower-enzyme-custom-rules-to-func --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {
  func.func @mul(%a: f32, %b: f32) -> f32 {
    %0 = arith.mulf %a, %b : f32
    %1 = math.exp %0 : f32
    %2 = arith.addf %b, %1 : f32
    return %2 : f32
  }

  // Split mode
  func.func @main(%a: f32, %b: f32) -> (f32, f32, f32) {
    %r, %tape = enzyme.autodiff_split_mode.primal @mul(%a, %b) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (f32, f32) -> (f32, !enzyme.Tape)

    // ---

    %dres = arith.constant 1.0 : f32
    %da, %db = enzyme.autodiff_split_mode.reverse @mul(%dres, %tape) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (f32, !enzyme.Tape) -> (f32, f32)

    return %r, %da, %db : f32, f32, f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> (f32, f32, f32) {
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %0:4 = call @mul_reverse_rule_primal(%arg0, %arg1) : (f32, f32) -> (f32, f32, f32, f32)
// CHECK-NEXT:    %1:2 = call @mul_reverse_rule_reverse(%cst, %0#1, %0#2, %0#3) : (f32, f32, f32, f32) -> (f32, f32)
// CHECK-NEXT:    return %0#0, %1#0, %1#1 : f32, f32, f32
// CHECK-NEXT:  }

// CHECK:  func.func private @mul_reverse_rule_primal(%arg0: f32, %arg1: f32) -> (f32, f32, f32, f32) {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg1 : f32
// CHECK-NEXT:    %1 = math.exp %0 : f32
// CHECK-NEXT:    %2 = arith.addf %arg1, %1 : f32
// CHECK-NEXT:    return %2, %arg0, %arg1, %0 : f32, f32, f32, f32
// CHECK-NEXT:  }

// CHECK:  func.func private @mul_reverse_rule_reverse(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (f32, f32) {
// CHECK-NEXT:    %0 = math.exp %arg3 : f32
// CHECK-NEXT:    %1 = arith.mulf %arg0, %0 : f32
// CHECK-NEXT:    %2 = arith.mulf %1, %arg2 : f32
// CHECK-NEXT:    %3 = arith.mulf %1, %arg1 : f32
// CHECK-NEXT:    %4 = arith.addf %arg0, %3 : f32
// CHECK-NEXT:    return %2, %4 : f32, f32
// CHECK-NEXT:  }
