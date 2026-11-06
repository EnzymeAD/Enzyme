// RUN: %eopt --split-input-file --enzyme %s | FileCheck %s

func.func @sin(%x: complex<f64>) -> complex<f64> {
  %res = complex.sin %x : complex<f64>
  return %res : complex<f64>
}

func.func @dsin(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @sin(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffesin(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[cos:.+]] = complex.cos %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.mul %[[dx]], %[[cos]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.sin %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @cos(%x: complex<f64>) -> complex<f64> {
  %res = complex.cos %x : complex<f64>
  return %res : complex<f64>
}

func.func @dcos(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @cos(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffecos(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[sin:.+]] = complex.sin %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[negsin:.+]] = complex.neg %[[sin]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.mul %[[dx]], %[[negsin]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.cos %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @tan(%x: complex<f64>) -> complex<f64> {
  %res = complex.tan %x : complex<f64>
  return %res : complex<f64>
}

func.func @dtan(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @tan(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffetan(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[cos:.+]] = complex.cos %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[cos2:.+]] = complex.mul %[[cos]], %[[cos]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.div %[[dx]], %[[cos2]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.tan %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @tanh(%x: complex<f64>) -> complex<f64> {
  %res = complex.tanh %x : complex<f64>
  return %res : complex<f64>
}

func.func @dtanh(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @tanh(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffetanh(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[one:.+]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[tanh:.+]] = complex.tanh %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[tanh2:.+]] = complex.mul %[[tanh]], %[[tanh]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[sub:.+]] = complex.sub %[[one]], %[[tanh2]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.mul %[[dx]], %[[sub]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.tanh %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @exp(%x: complex<f64>) -> complex<f64> {
  %res = complex.exp %x : complex<f64>
  return %res : complex<f64>
}

func.func @dexp(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @exp(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffeexp(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[exp:.+]] = complex.exp %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.mul %[[dx]], %[[exp]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.exp %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @expm1(%x: complex<f64>) -> complex<f64> {
  %res = complex.expm1 %x : complex<f64>
  return %res : complex<f64>
}

func.func @dexpm1(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @expm1(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffeexpm1(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[exp:.+]] = complex.exp %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.mul %[[dx]], %[[exp]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.expm1 %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }
