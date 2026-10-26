// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @div(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.div %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @ddiv(%x: complex<f64>, %y: complex<f64>, %dr: complex<f64>) -> (complex<f64>, complex<f64>) {
  %0:2 = enzyme.autodiff @div(%x, %y, %dr) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>, complex<f64>) -> (complex<f64>, complex<f64>)
  return %0#0, %0#1 : complex<f64>, complex<f64>
}

// CHECK: func.func private @diffediv(%[[x:.+]]: complex<f64>, %[[y:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> (complex<f64>, complex<f64>) {
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[gx:.+]] = complex.div %[[cdr]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[rx:.+]] = complex.conj %[[gx]] : complex<f64>
// CHECK-NEXT:    %[[q:.+]] = complex.div %[[cdr]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[xy:.+]] = complex.div %[[x]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[m:.+]] = complex.mul %[[q]], %[[xy]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[gy:.+]] = complex.neg %[[m]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[ry:.+]] = complex.conj %[[gy]] : complex<f64>
// CHECK-NEXT:    return %[[rx]], %[[ry]] : complex<f64>, complex<f64>
// CHECK-NEXT:  }

// -----

func.func @div(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.div %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @ddiv(%x: complex<f64>, %y: complex<f64>, %dr: complex<f64>) -> (complex<f64>, complex<f64>) {
  %0:2 = enzyme.autodiff @div(%x, %y, %dr) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>], strong_zero=true }> : (complex<f64>, complex<f64>, complex<f64>) -> (complex<f64>, complex<f64>)
  return %0#0, %0#1 : complex<f64>, complex<f64>
}

// CHECK: func.func private @diffediv(%[[x:.+]]: complex<f64>, %[[y:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> (complex<f64>, complex<f64>) {
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[cond1:.+]] = complex.eq %[[cdr]], %[[zero]] : complex<f64>
// CHECK-NEXT:    %[[q1:.+]] = complex.div %[[cdr]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[gx:.+]] = arith.select %[[cond1]], %[[zero]], %[[q1]] : complex<f64>
// CHECK-NEXT:    %[[rx:.+]] = complex.conj %[[gx]] : complex<f64>
// CHECK-NEXT:    %[[cond2:.+]] = complex.eq %[[cdr]], %[[zero]] : complex<f64>
// CHECK-NEXT:    %[[q2:.+]] = complex.div %[[cdr]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[s2:.+]] = arith.select %[[cond2]], %[[zero]], %[[q2]] : complex<f64>
// CHECK-NEXT:    %[[xy:.+]] = complex.div %[[x]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[m:.+]] = complex.mul %[[s2]], %[[xy]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[gy:.+]] = complex.neg %[[m]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[ry:.+]] = complex.conj %[[gy]] : complex<f64>
// CHECK-NEXT:    return %[[rx]], %[[ry]] : complex<f64>, complex<f64>
// CHECK-NEXT:  }
