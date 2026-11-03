// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @sin(%x: complex<f64>) -> complex<f64> {
  %res = complex.sin %x : complex<f64>
  return %res : complex<f64>
}

func.func @dsin(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @sin(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @diffesin(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[cos:.+]] = complex.cos %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = complex.mul %[[cdr]], %[[cos]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @cos(%x: complex<f64>) -> complex<f64> {
  %res = complex.cos %x : complex<f64>
  return %res : complex<f64>
}

func.func @dcos(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @cos(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @diffecos(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[sin:.+]] = complex.sin %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[negsin:.+]] = complex.neg %[[sin]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = complex.mul %[[cdr]], %[[negsin]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @tan(%x: complex<f64>) -> complex<f64> {
  %res = complex.tan %x : complex<f64>
  return %res : complex<f64>
}

func.func @dtan(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @tan(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @diffetan(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[cos:.+]] = complex.cos %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[cos2:.+]] = complex.mul %[[cos]], %[[cos]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = complex.div %[[cdr]], %[[cos2]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @tanh(%x: complex<f64>) -> complex<f64> {
  %res = complex.tanh %x : complex<f64>
  return %res : complex<f64>
}

func.func @dtanh(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @tanh(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @diffetanh(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[one:.+]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[tanh:.+]] = complex.tanh %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[tanh2:.+]] = complex.mul %[[tanh]], %[[tanh]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[sub:.+]] = complex.sub %[[one]], %[[tanh2]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = complex.mul %[[cdr]], %[[sub]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }
