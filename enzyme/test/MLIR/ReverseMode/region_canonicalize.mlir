// RUN: %eopt --split-input-file --canonicalize %s | FileCheck %s

func.func @test1(%vx : f64, %dout : f64) -> f64 {
  %cst2 = arith.constant 2.0 : f64
  %x2 = arith.mulf %cst2, %vx : f64
  %out = enzyme.autodiff_region(%vx, %dout){
    ^bb0(%x : f64):
      %y = arith.mulf %x2, %x2 : f64
      enzyme.yield %y : f64 
  } attributes { activity = [#enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_active>] }: (f64,f64) -> f64
  return %out : f64
}

// CHECK: func.func @test1(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.mulf %arg0, %cst : f64
// CHECK-NEXT:   %1 = enzyme.autodiff_region(%arg1) {
// CHECK-NEXT:     %2 = arith.mulf %0, %0 : f64
// CHECK-NEXT:     enzyme.yield %2 : f64
// CHECK-NEXT:   } attributes {activity = [], ret_activity = [#enzyme<activity enzyme_active>]} : (f64) -> f64
// CHECK-NEXT:   return %1 : f64
// CHECK-NEXT: }


