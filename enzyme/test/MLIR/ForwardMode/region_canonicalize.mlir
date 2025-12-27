// RUN: %eopt --split-input-file --canonicalize %s | FileCheck %s


func.func @test1(%x : f64, %dx : f64, %y : f64, %z :f64, %dz : f64) -> f64 {
  %cst2 = arith.constant 2.0 : f64
  %x2 = arith.mulf %cst2, %x : f64
  %dout = enzyme.fwddiff_region(%x, %dx, %y, %z, %dz){
  ^bb0(%xx : f64, %yy : f64, %zz : f64):
      %t1 = arith.mulf %x2, %x2 : f64
      %t2 = arith.addf %t1, %zz : f64
      enzyme.yield %t2 : f64 
  } attributes { activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] }: (f64,f64,f64,f64,f64) -> f64
  return %dout : f64
}

// CHECK: func.func @test1(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64, %arg4: f64) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.mulf %arg0, %cst : f64
// CHECK-NEXT:   %1 = enzyme.fwddiff_region(%arg3, %arg4) {
// CHECK-NEXT:   ^bb0(%arg5: f64):
// CHECK-NEXT:     %2 = arith.mulf %0, %0 : f64
// CHECK-NEXT:     %3 = arith.addf %2, %arg5 : f64
// CHECK-NEXT:     enzyme.yield %3 : f64
// CHECK-NEXT:   } attributes {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (f64, f64) -> f64
// CHECK-NEXT:   return %1 : f64
// CHECK-NEXT: }

// ----- 

// Even if x2 = 2*x, it will be considered enzyme_const, 
// and %x will be eliminated here

func.func @test2(%x : f64, %dx : f64) -> f64 {
  %cst2 = arith.constant 2.0 : f64
  %x2 = arith.mulf %cst2, %x : f64
  %dout = enzyme.fwddiff_region(%x, %dx){
  ^bb0(%xx : f64):
      %y = arith.mulf %x2, %x2 : f64
      enzyme.yield %y : f64 
  } attributes { activity = [#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] }: (f64,f64) -> f64
  return %dout : f64
}

// CHECK:  func.func @test2(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg0, %cst : f64
// CHECK-NEXT:    %1 = enzyme.fwddiff_region() {
// CHECK-NEXT:      %2 = arith.mulf %0, %0 : f64
// CHECK-NEXT:      enzyme.yield %2 : f64
// CHECK-NEXT:    } attributes {activity = [], ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : () -> f64
// CHECK-NEXT:    return %1 : f64
// CHECK-NEXT:  }
