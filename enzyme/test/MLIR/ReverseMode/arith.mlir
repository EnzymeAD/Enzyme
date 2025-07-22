// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @select(%c: i1, %a: f64, %b: f64) -> f64 {
  %res = arith.select %c, %a, %b : f64
  return %res : f64
}

func.func @dselect(%c: i1, %a: f64, %b: f64, %dr: f64) -> (f64, f64) {
  %0:2 = enzyme.autodiff @select(%c, %a, %b, %dr)
    {
      activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (i1, f64, f64, f64) -> (f64, f64)
  return %0#0, %0#1 : f64, f64
}

// CHECK: func.func private @diffeselect(%[[c:.+]]: i1, %[[a:.+]]: f64, %[[b:.+]]: f64, %[[dr:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[da:.+]] = arith.select %[[c]], %[[dr]], %[[zero]] : f64
// CHECK-NEXT:    %[[db:.+]] = arith.select %[[c]], %[[zero]], %[[dr]] : f64
// CHECK-NEXT:    return %[[da]], %[[db]] : f64, f64
// CHECK-NEXT:  }
