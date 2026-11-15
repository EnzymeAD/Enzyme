// RUN: %eopt --split-input-file --enzyme %s | FileCheck %s

func.func @div(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.div %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @ddiv(%x: complex<f64>, %dx: complex<f64>, %y: complex<f64>, %dy: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @div(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>, complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffediv(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>, %[[y:.+]]: complex<f64>, %[[dy:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[dxy:.+]] = complex.mul %[[dx]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[dyx:.+]] = complex.mul %[[dy]], %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[num:.+]] = complex.sub %[[dxy]], %[[dyx]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[yy:.+]] = complex.mul %[[y]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.div %[[num]], %[[yy]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.div %[[x]], %[[y]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @div(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.div %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @ddiv(%x: complex<f64>, %dx: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @div(%x, %dx, %y) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffediv(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>, %[[y:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[dxy:.+]] = complex.mul %[[dx]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[num:.+]] = complex.sub %[[dxy]], %[[zero]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[yy:.+]] = complex.mul %[[y]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.div %[[num]], %[[yy]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.div %[[x]], %[[y]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @div(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.div %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @ddiv(%x: complex<f64>, %dx: complex<f64>, %y: complex<f64>, %dy: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @div(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], strong_zero=true }> : (complex<f64>, complex<f64>, complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffediv(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>, %[[y:.+]]: complex<f64>, %[[dy:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[dxy:.+]] = complex.mul %[[dx]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[dyx:.+]] = complex.mul %[[dy]], %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[num:.+]] = complex.sub %[[dxy]], %[[dyx]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[yy:.+]] = complex.mul %[[y]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[cond:.+]] = complex.eq %[[num]], %[[zero]] : complex<f64>
// CHECK-NEXT:    %[[quot:.+]] = complex.div %[[num]], %[[yy]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = arith.select %[[cond]], %[[zero]], %[[quot]] : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.div %[[x]], %[[y]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }
