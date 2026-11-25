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

// -----

func.func @sqrt(%x: complex<f64>) -> complex<f64> {
  %res = complex.sqrt %x : complex<f64>
  return %res : complex<f64>
}

func.func @dsqrt(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @sqrt(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffesqrt(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[iszero:.+]] = complex.eq %[[x]], %[[zero]] : complex<f64>
// CHECK-NEXT:    %[[two:.+]] = complex.constant [2.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[sqrt:.+]] = complex.sqrt %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[den:.+]] = complex.mul %[[two]], %[[sqrt]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[div:.+]] = complex.div %[[dx]], %[[den]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = arith.select %[[iszero]], %[[zero]], %[[div]] : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.sqrt %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @rsqrt(%x: complex<f64>) -> complex<f64> {
  %res = complex.rsqrt %x : complex<f64>
  return %res : complex<f64>
}

func.func @drsqrt(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @rsqrt(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffersqrt(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[rsqrt:.+]] = complex.rsqrt %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[neg:.+]] = complex.neg %[[rsqrt]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[two:.+]] = complex.constant [2.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[den:.+]] = complex.mul %[[two]], %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[div:.+]] = complex.div %[[neg]], %[[den]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.mul %[[dx]], %[[div]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.rsqrt %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @pow(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.pow %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @dpow(%x: complex<f64>, %dx: complex<f64>, %y: complex<f64>, %dy: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @pow(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>, complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffepow(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>, %[[y:.+]]: complex<f64>, %[[dy:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[one:.+]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[ym1:.+]] = complex.sub %[[y]], %[[one]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[powm1:.+]] = complex.pow %[[x]], %[[ym1]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[px:.+]] = complex.mul %[[y]], %[[powm1]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[tx:.+]] = complex.mul %[[dx]], %[[px]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[iszero:.+]] = complex.eq %[[x]], %[[zero]] : complex<f64>
// CHECK-NEXT:    %[[pow:.+]] = complex.pow %[[x]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[log:.+]] = complex.log %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[powlog:.+]] = complex.mul %[[pow]], %[[log]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[py:.+]] = arith.select %[[iszero]], %[[zero]], %[[powlog]] : complex<f64>
// CHECK-NEXT:    %[[ty:.+]] = complex.mul %[[dy]], %[[py]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.add %[[tx]], %[[ty]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.pow %[[x]], %[[y]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @powi(%x: complex<f32>, %n: i32) -> complex<f32> {
  %res = complex.powi %x, %n : complex<f32>, i32
  return %res : complex<f32>
}

func.func @dpowi(%x: complex<f32>, %dx: complex<f32>, %n: i32) -> complex<f32> {
  %0 = enzyme.fwddiff @powi(%x, %dx, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f32>, complex<f32>, i32) -> complex<f32>
  return %0 : complex<f32>
}

// CHECK: func.func private @fwddiffepowi(%[[x:.+]]: complex<f32>, %[[dx:.+]]: complex<f32>, %[[n:.+]]: i32) -> complex<f32> {
// CHECK-NEXT:    %[[izero:.+]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[iszero:.+]] = arith.cmpi eq, %[[n]], %[[izero]] : i32
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
// CHECK-NEXT:    %[[nf:.+]] = arith.sitofp %[[n]] : i32 to f32
// CHECK-NEXT:    %[[fzero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[nc:.+]] = complex.create %[[nf]], %[[fzero]] : complex<f32>
// CHECK-NEXT:    %[[ione:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[nm1:.+]] = arith.subi %[[n]], %[[ione]] : i32
// CHECK-NEXT:    %[[powm1:.+]] = complex.powi %[[x]], %[[nm1]] fastmath<fast> : complex<f32>, i32
// CHECK-NEXT:    %[[px:.+]] = complex.mul %[[nc]], %[[powm1]] fastmath<fast> : complex<f32>
// CHECK-NEXT:    %[[tx:.+]] = complex.mul %[[dx]], %[[px]] fastmath<fast> : complex<f32>
// CHECK-NEXT:    %[[res:.+]] = arith.select %[[iszero]], %[[zero]], %[[tx]] : complex<f32>
// CHECK-NEXT:    %{{.+}} = complex.powi %[[x]], %[[n]] : complex<f32>, i32
// CHECK-NEXT:    return %[[res]] : complex<f32>
// CHECK-NEXT:  }

// -----

func.func @log(%x: complex<f64>) -> complex<f64> {
  %res = complex.log %x : complex<f64>
  return %res : complex<f64>
}

func.func @dlog(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @log(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffelog(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[res:.+]] = complex.div %[[dx]], %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.log %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>

// -----

func.func @log1p(%x: complex<f64>) -> complex<f64> {
  %res = complex.log1p %x : complex<f64>
  return %res : complex<f64>
}

func.func @dlog1p(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @log1p(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffelog1p(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[one:.+]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[xp1:.+]] = complex.add %[[x]], %[[one]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.div %[[dx]], %[[xp1]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.log1p %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>

// -----

func.func @abs(%x: complex<f64>) -> f64 {
  %res = complex.abs %x : complex<f64>
  return %res : f64
}

func.func @dabs(%x: complex<f64>, %dx: complex<f64>) -> f64 {
  %0 = enzyme.fwddiff @abs(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> f64
  return %0 : f64
}

// CHECK: func.func private @fwddiffeabs(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> f64 {
// CHECK-NEXT:    %[[dxre:.+]] = complex.re %[[dx]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[s:.+]] = complex.sign %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[sre:.+]] = complex.re %[[s]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[a:.+]] = arith.mulf %[[dxre]], %[[sre]] fastmath<fast> : f64
// CHECK-NEXT:    %[[dxim:.+]] = complex.im %[[dx]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[sim:.+]] = complex.im %[[s]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[b:.+]] = arith.mulf %[[dxim]], %[[sim]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.addf %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = complex.abs %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @neg(%x: complex<f64>) -> complex<f64> {
  %res = complex.neg %x : complex<f64>
  return %res : complex<f64>
}

func.func @dneg(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @neg(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}


// CHECK: func.func private @fwddiffeneg(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[res:.+]] = complex.neg %[[dx]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.neg %[[x]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>

// -----

func.func @sub(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.sub %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @dsub(%x: complex<f64>, %dx: complex<f64>, %y: complex<f64>, %dy: complex<f64>) -> complex<f64> {
  %0 = enzyme.fwddiff @sub(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (complex<f64>, complex<f64>, complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @fwddiffesub(%[[x:.+]]: complex<f64>, %[[dx:.+]]: complex<f64>, %[[y:.+]]: complex<f64>, %[[dy:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[negdy:.+]] = complex.neg %[[dy]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.add %[[dx]], %[[negdy]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %{{.+}} = complex.sub %[[x]], %[[y]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
