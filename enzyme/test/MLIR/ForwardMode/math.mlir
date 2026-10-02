// RUN: %eopt --split-input-file --enzyme %s | FileCheck %s

func.func @sinh(%x: f64) -> f64 {
  %res = math.sinh %x : f64
  return %res : f64
}

func.func @dsinh(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @sinh(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffesinh(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[cosh:.+]] = math.cosh %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.mulf %[[dx]], %[[cosh]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.sinh %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @cosh(%x: f64) -> f64 {
  %res = math.cosh %x : f64
  return %res : f64
}

func.func @dcosh(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @cosh(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffecosh(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[sinh:.+]] = math.sinh %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.mulf %[[dx]], %[[sinh]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.cosh %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @tanh(%x: f64) -> f64 {
  %res = math.tanh %x : f64
  return %res : f64
}

func.func @dtanh(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @tanh(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffetanh(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[cosh:.+]] = math.cosh %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[cosh2:.+]] = arith.mulf %[[cosh]], %[[cosh]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[dx]], %[[cosh2]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.tanh %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @tan(%x: f64) -> f64 {
  %res = math.tan %x : f64
  return %res : f64
}

func.func @dtan(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @tan(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffetan(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[cos:.+]] = math.cos %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[cos2:.+]] = arith.mulf %[[cos]], %[[cos]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[dx]], %[[cos2]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.tan %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @asin(%x: f64) -> f64 {
  %res = math.asin %x : f64
  return %res : f64
}

func.func @dasin(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @asin(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeasin(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[omx2:.+]] = arith.subf %[[one]], %[[xsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sqrt:.+]] = math.sqrt %[[omx2]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[dx]], %[[sqrt]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.asin %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @acos(%x: f64) -> f64 {
  %res = math.acos %x : f64
  return %res : f64
}

func.func @dacos(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @acos(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeacos(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[negdx:.+]] = arith.negf %[[dx]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[omx2:.+]] = arith.subf %[[one]], %[[xsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sqrt:.+]] = math.sqrt %[[omx2]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[negdx]], %[[sqrt]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.acos %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }
