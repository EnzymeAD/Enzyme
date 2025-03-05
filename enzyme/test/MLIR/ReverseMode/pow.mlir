// RUN: %eopt --enzyme -canonicalize --remove-unnecessary-enzyme-ops -enzyme-simplify-math -canonicalize %s | FileCheck %s

module {
  func.func @ppow(%x: f64) -> f64 {
    %cst = arith.constant 1.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 10 : index
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%r_it = %cst) -> f64 {
      %r_next = arith.mulf %r_it, %x : f64
      scf.yield %r_next : f64
    }
    return %r : f64
  }

  func.func @dppow(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @ppow(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:  func.func private @diffeppow(%[[x:.+]]: f64, %[[dr:.+]]: f64) -> f64 {
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.0
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[cache:.+]] = tensor.empty() : tensor<10xf64>
// CHECK-NEXT:    %{{.+}} = scf.for %[[iv:.+]] = %c0 to %c10 step %c1 iter_args(%[[r_it:.+]] = %[[one]], %[[cache_iter:.+]] = %[[cache]]) -> (f64, tensor<10xf64>) {
// CHECK-NEXT:      %[[cache_new:.+]] = tensor.insert %[[r_it]] into %[[cache_iter]][%[[iv]]] : tensor<10xf64>
// CHECK-NEXT:      %[[fwd:.+]] = arith.mulf %[[r_it]], %[[x]] : f64
// CHECK-NEXT:      scf.yield %[[fwd]], %[[cache_new]] : f64, tensor<10xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.+}} = scf.for %[[div:.+]] = %c0 to %c10 step %c1 iter_args(%[[dr_it:.+]] = %[[dr]], %[[rev_idx:.+]] = %c9, %[[dx0:.+]] = %[[zero]]) -> (f64, index, f64) {
// CHECK-NEXT:      %[[r_cached:.+]] = tensor.extract %1#1[%[[rev_idx]]] : tensor<10xf64>
// CHECK-NEXT:      %[[dr_next:.+]] = arith.mulf %[[dr_it]], %[[x]] : f64
// CHECK-NEXT:      %[[dx_next:.+]] = arith.mulf %[[dr_it]], %[[r_cached]] : f64
// CHECK-NEXT:      %[[dx1:.+]] = arith.addf %[[dx0]], %[[dx_next]]
// CHECK-NEXT:      %[[new_rev_idx:.+]] = arith.subi %[[rev_idx]], %c1 : index
// CHECK-NEXT:      scf.yield %[[dr_next]], %[[new_rev_idx]], %[[dx1]] : f64, index, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %2#2 : f64
