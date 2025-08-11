// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @sq(%x: f64) -> f64 {
  %res = math.sqrt %x : f64
  return %res : f64
}

func.func @dsq(%x: f64, %dr: f64) -> f64 {
  %0 = enzyme.autodiff @sq(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @diffesq(%[[x:.+]]: f64, %[[dr:.+]]: f64) -> f64 {
// CHECK-NEXT:   %[[two:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[cond:.+]] = arith.cmpf oeq, %[[x]], %[[zero]] : f64
// CHECK-NEXT:   %[[sq:.+]] = math.sqrt %[[x]] : f64
// CHECK-NEXT:   %[[twosq:.+]] = arith.mulf %[[sq]], %[[two]] : f64
// CHECK-NEXT:   %[[rtwosq:.+]] = arith.divf %[[dr]], %[[twosq]] : f64
// CHECK-NEXT:   %[[res:.+]] = arith.select %[[cond]], %[[zero]], %[[rtwosq]] : f64
// CHECK-NEXT:   return %[[res]] : f64
// CHECK-NEXT: }

// -----

func.func @atan(%x: f64) -> f64 {
  %res = math.atan %x : f64
  return %res : f64
}

func.func @datan(%x: f64, %dr: f64) -> f64 {
  %0 = enzyme.autodiff @atan(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @diffeatan(%[[x:.+]]: f64, %[[dr:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[one:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %[[xsqr:.+]] = arith.mulf %[[x]], %[[x]] : f64
// CHECK-NEXT:    %[[xsqrp1:.+]] = arith.addf %[[xsqr]], %[[one]] : f64
// CHECK-NEXT:    %[[recip:.+]] = arith.divf %[[one]], %[[xsqrp1]] : f64
// CHECK-NEXT:    %[[res:.+]] = arith.mulf %[[dr]], %[[recip]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @absf(%x: f64) -> f64 {
  %res = math.absf %x : f64
  return %res : f64
}

func.func @dabsf(%x: f64, %dr: f64) -> f64 {
  %0 = enzyme.autodiff @absf(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
  return %0 : f64
}

// CHECK: func.func private @diffeabsf(%[[x]]: f64, %[[dr]]: f64) -> f64 {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[ge_z:.+]] = arith.cmpf oge, %[[x]], %[[zero]] : f64
// CHECK-NEXT:    %[[negdr:.+]] = arith.negf %[[dr]] : f64
// CHECK-NEXT:    %[[res:.+]] = arith.select %[[ge_z]], %[[dr]], %[[negdr]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }
