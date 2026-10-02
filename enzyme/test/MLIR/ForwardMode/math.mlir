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

// -----

func.func @asinh(%x: f64) -> f64 {
  %res = math.asinh %x : f64
  return %res : f64
}

func.func @dasinh(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @asinh(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeasinh(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[den:.+]] = arith.addf %[[one]], %[[xsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sqrt:.+]] = math.sqrt %[[den]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[dx]], %[[sqrt]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.asinh %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @acosh(%x: f64) -> f64 {
  %res = math.acosh %x : f64
  return %res : f64
}

func.func @dacosh(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @acosh(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeacosh(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[den:.+]] = arith.subf %[[xsqr]], %[[one]] fastmath<fast> : f64
// CHECK-NEXT:    %[[sqrt:.+]] = math.sqrt %[[den]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[dx]], %[[sqrt]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.acosh %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @atanh(%x: f64) -> f64 {
  %res = math.atanh %x : f64
  return %res : f64
}

func.func @datanh(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @atanh(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeatanh(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[den:.+]] = arith.subf %[[one]], %[[xsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[dx]], %[[den]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.atanh %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @expm1(%x: f64) -> f64 {
  %res = math.expm1 %x : f64
  return %res : f64
}

func.func @dexpm1(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @expm1(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeexpm1(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[exp:.+]] = math.exp %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.mulf %[[dx]], %[[exp]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.expm1 %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @log10(%x: f64) -> f64 {
  %res = math.log10 %x : f64
  return %res : f64
}

func.func @dlog10(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @log10(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffelog10(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[ln10:.+]] = arith.constant 2.302585092994{{[0-9]*}} : f64
// CHECK-NEXT:    %[[den:.+]] = arith.mulf %[[x]], %[[ln10]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[dx]], %[[den]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.log10 %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @rsqrt(%x: f64) -> f64 {
  %res = math.rsqrt %x : f64
  return %res : f64
}

func.func @drsqrt(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @rsqrt(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffersqrt(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[rsqrt:.+]] = math.rsqrt %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[neg:.+]] = arith.negf %[[rsqrt]] fastmath<fast> : f64
// CHECK-NEXT:    %[[two:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %[[twox:.+]] = arith.mulf %[[two]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[deriv:.+]] = arith.divf %[[neg]], %[[twox]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.mulf %[[dx]], %[[deriv]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.rsqrt %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @cbrt(%x: f64) -> f64 {
  %res = math.cbrt %x : f64
  return %res : f64
}

func.func @dcbrt(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @cbrt(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffecbrt(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[cbrt:.+]] = math.cbrt %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[num:.+]] = arith.mulf %[[dx]], %[[cbrt]] fastmath<fast> : f64
// CHECK-NEXT:    %[[three:.+]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:    %[[threex:.+]] = arith.mulf %[[three]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[num]], %[[threex]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.cbrt %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @erf(%x: f64) -> f64 {
  %res = math.erf %x : f64
  return %res : f64
}

func.func @derf(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @erf(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeerf(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[c:.+]] = arith.constant 1.128379167095{{[0-9]*}} : f64
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[negxsqr:.+]] = arith.negf %[[xsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[exp:.+]] = math.exp %[[negxsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[fac:.+]] = arith.mulf %[[c]], %[[exp]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.mulf %[[dx]], %[[fac]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.erf %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @erfc(%x: f64) -> f64 {
  %res = math.erfc %x : f64
  return %res : f64
}

func.func @derfc(%x: f64, %dx: f64) -> f64 {
  %0 = enzyme.fwddiff @erfc(%x, %dx) { activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeerfc(%[[x:.+]]: f64, %[[dx:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[c:.+]] = arith.constant -1.128379167095{{[0-9]*}} : f64
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[negxsqr:.+]] = arith.negf %[[xsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[exp:.+]] = math.exp %[[negxsqr]] fastmath<fast> : f64
// CHECK-NEXT:    %[[fac:.+]] = arith.mulf %[[c]], %[[exp]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.mulf %[[dx]], %[[fac]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = math.erfc %[[x]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }
