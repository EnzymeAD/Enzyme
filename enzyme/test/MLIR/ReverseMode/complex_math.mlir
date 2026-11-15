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

// -----

func.func @exp(%x: complex<f64>) -> complex<f64> {
  %res = complex.exp %x : complex<f64>
  return %res : complex<f64>
}

func.func @dexp(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @exp(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @diffeexp(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[exp:.+]] = complex.exp %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = complex.mul %[[cdr]], %[[exp]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @expm1(%x: complex<f64>) -> complex<f64> {
  %res = complex.expm1 %x : complex<f64>
  return %res : complex<f64>
}

func.func @dexpm1(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @expm1(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @diffeexpm1(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[exp:.+]] = complex.exp %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = complex.mul %[[cdr]], %[[exp]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @sqrt(%x: complex<f64>) -> complex<f64> {
  %res = complex.sqrt %x : complex<f64>
  return %res : complex<f64>
}

func.func @dsqrt(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @sqrt(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @diffesqrt(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[two:.+]] = complex.constant [2.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[iszero:.+]] = complex.eq %[[x]], %[[zero]] : complex<f64>
// CHECK-NEXT:    %[[sqrt:.+]] = complex.sqrt %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[den:.+]] = complex.mul %[[sqrt]], %[[two]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[div:.+]] = complex.div %[[cdr]], %[[den]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = arith.select %[[iszero]], %[[zero]], %[[div]] : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @rsqrt(%x: complex<f64>) -> complex<f64> {
  %res = complex.rsqrt %x : complex<f64>
  return %res : complex<f64>
}

func.func @drsqrt(%x: complex<f64>, %dr: complex<f64>) -> complex<f64> {
  %0 = enzyme.autodiff @rsqrt(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>) -> complex<f64>
  return %0 : complex<f64>
}

// CHECK: func.func private @differsqrt(%[[x:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> complex<f64> {
// CHECK-NEXT:    %[[two:.+]] = complex.constant [2.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[rsqrt:.+]] = complex.rsqrt %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[neg:.+]] = complex.neg %[[rsqrt]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[den:.+]] = complex.mul %[[x]], %[[two]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[div:.+]] = complex.div %[[neg]], %[[den]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[g:.+]] = complex.mul %[[cdr]], %[[div]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f64>
// CHECK-NEXT:    return %[[res]] : complex<f64>
// CHECK-NEXT:  }

// -----

func.func @pow(%x: complex<f64>, %y: complex<f64>) -> complex<f64> {
  %res = complex.pow %x, %y : complex<f64>
  return %res : complex<f64>
}

func.func @dpow(%x: complex<f64>, %y: complex<f64>, %dr: complex<f64>) -> (complex<f64>, complex<f64>) {
  %0, %1 = enzyme.autodiff @pow(%x, %y, %dr) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f64>, complex<f64>, complex<f64>) -> (complex<f64>, complex<f64>)
  return %0, %1 : complex<f64>, complex<f64>
}

// CHECK: func.func private @diffepow(%[[x:.+]]: complex<f64>, %[[y:.+]]: complex<f64>, %[[dr:.+]]: complex<f64>) -> (complex<f64>, complex<f64>) {
// CHECK-NEXT:    %[[one:.+]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f64>
// CHECK-NEXT:    %[[ym1:.+]] = complex.sub %[[y]], %[[one]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[powm1:.+]] = complex.pow %[[x]], %[[ym1]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[px:.+]] = complex.mul %[[y]], %[[powm1]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[gx:.+]] = complex.mul %[[cdr]], %[[px]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[resx:.+]] = complex.conj %[[gx]] : complex<f64>
// CHECK-NEXT:    %[[iszero:.+]] = complex.eq %[[x]], %[[zero]] : complex<f64>
// CHECK-NEXT:    %[[pow:.+]] = complex.pow %[[x]], %[[y]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[log:.+]] = complex.log %[[x]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[powlog:.+]] = complex.mul %[[pow]], %[[log]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[py:.+]] = arith.select %[[iszero]], %[[zero]], %[[powlog]] : complex<f64>
// CHECK-NEXT:    %[[gy:.+]] = complex.mul %[[cdr]], %[[py]] fastmath<fast> : complex<f64>
// CHECK-NEXT:    %[[resy:.+]] = complex.conj %[[gy]] : complex<f64>
// CHECK-NEXT:    return %[[resx]], %[[resy]] : complex<f64>, complex<f64>
// CHECK-NEXT:  }

// -----

func.func @powi(%x: complex<f32>, %n: i32) -> complex<f32> {
  %res = complex.powi %x, %n : complex<f32>, i32
  return %res : complex<f32>
}

func.func @dpowi(%x: complex<f32>, %n: i32, %dr: complex<f32>) -> complex<f32> {
  %0 = enzyme.autodiff @powi(%x, %n, %dr) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (complex<f32>, i32, complex<f32>) -> complex<f32>
  return %0 : complex<f32>
}

// CHECK: func.func private @diffepowi(%[[x:.+]]: complex<f32>, %[[n:.+]]: i32, %[[dr:.+]]: complex<f32>) -> complex<f32> {
// CHECK-NEXT:    %[[ione:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[fzero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[izero:.+]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[zero:.+]] = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
// CHECK-NEXT:    %[[cdr:.+]] = complex.conj %[[dr]] : complex<f32>
// CHECK-NEXT:    %[[iszero:.+]] = arith.cmpi eq, %[[n]], %[[izero]] : i32
// CHECK-NEXT:    %[[nf:.+]] = arith.sitofp %[[n]] : i32 to f32
// CHECK-NEXT:    %[[nc:.+]] = complex.create %[[nf]], %[[fzero]] : complex<f32>
// CHECK-NEXT:    %[[nm1:.+]] = arith.subi %[[n]], %[[ione]] : i32
// CHECK-NEXT:    %[[powm1:.+]] = complex.powi %[[x]], %[[nm1]] fastmath<fast> : complex<f32>, i32
// CHECK-NEXT:    %[[px:.+]] = complex.mul %[[nc]], %[[powm1]] fastmath<fast> : complex<f32>
// CHECK-NEXT:    %[[gx:.+]] = complex.mul %[[cdr]], %[[px]] fastmath<fast> : complex<f32>
// CHECK-NEXT:    %[[g:.+]] = arith.select %[[iszero]], %[[zero]], %[[gx]] : complex<f32>
// CHECK-NEXT:    %[[res:.+]] = complex.conj %[[g]] : complex<f32>
// CHECK-NEXT:    return %[[res]] : complex<f32>
// CHECK-NEXT:  }
