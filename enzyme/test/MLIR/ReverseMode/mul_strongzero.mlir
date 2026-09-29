// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s

module {
  func.func @square(%x: f64) -> f64 {
    %next = arith.mulf %x, %x : f64
    return %next : f64
  }

  func.func @dsquare(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @square(%x, %dr) { activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>], strong_zero=true } : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:  func.func private @diffesquare(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.cmpf oeq, %arg1, %cst fastmath<fast> : f64
// CHECK-NEXT:    %1 = arith.mulf %arg1, %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %2 = arith.select %0, %cst, %1 : f64
// CHECK-NEXT:    %3 = arith.addf %2, %2 fastmath<fast> : f64
// CHECK-NEXT:    return %3 : f64
// CHECK-NEXT:  }

// -----

// Derivatives with different strong-zero settings must not share a cache entry.
module {
  func.func @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @main(%x: f64, %dy: f64) -> (f64, f64) {
    %plain = enzyme.autodiff @square(%x, %dy) {
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64) -> f64
    %strong = enzyme.autodiff @square(%x, %dy) {
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      strong_zero = true
    } : (f64, f64) -> f64
    return %plain, %strong : f64, f64
  }
}

// CHECK-LABEL: func.func @main
// CHECK: call @diffesquare(
// CHECK: call @diffesquare_0(
// CHECK-LABEL: func.func private @diffesquare(
// CHECK-NOT: arith.select
// CHECK: return
// CHECK-LABEL: func.func private @diffesquare_0(
// CHECK: arith.cmpf oeq
// CHECK: arith.select
// CHECK: return

// -----

// Derived custom rules must also keep the strong-zero setting of each caller.
module {
  func.func @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @outer1(%x: f64) -> f64 {
    %y = func.call @square(%x) : (f64) -> f64
    return %y : f64
  }
  func.func @outer2(%x: f64) -> f64 {
    %y = func.call @square(%x) : (f64) -> f64
    return %y : f64
  }
  func.func @main(%x: f64, %dy: f64) -> (f64, f64) {
    %plain = enzyme.autodiff @outer1(%x, %dy) {
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64) -> f64
    %strong = enzyme.autodiff @outer2(%x, %dy) {
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      strong_zero = true
    } : (f64, f64) -> f64
    return %plain, %strong : f64, f64
  }
}

// CHECK-LABEL: func.func private @square_reverse_rule_reverse(
// CHECK-NOT: arith.select
// CHECK: return
// CHECK-LABEL: func.func private @square_reverse_rule_0_reverse(
// CHECK: arith.cmpf oeq
// CHECK: arith.select
// CHECK: return
