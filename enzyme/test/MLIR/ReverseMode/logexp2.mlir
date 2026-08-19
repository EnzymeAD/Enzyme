// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

module {
  func.func @log2(%x: f64) -> f64 {
    %y = math.log2 %x : f64
    return %y : f64
  }

  func.func @dlog2(%x: f64, %dy: f64) -> f64 {
    %dx = enzyme.autodiff @log2(%x, %dy) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %dx : f64
  }

  func.func @exp2(%x: f64) -> f64 {
    %y = math.exp2 %x : f64
    return %y : f64
  }

  func.func @dexp2(%x: f64, %dy: f64) -> f64 {
    %dx = enzyme.autodiff @exp2(%x, %dy) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>]
    } : (f64, f64) -> f64
    return %dx : f64
  }
}

// CHECK-LABEL: func.func private @diffelog2(
// CHECK:         %[[LN2:.+]] = arith.constant 0.693147{{.*}} : f64
// CHECK:         %[[DENOM:.+]] = arith.mulf %arg0, %[[LN2]]{{.*}} : f64
// CHECK:         %[[DY:.+]] = arith.divf %arg1, %[[DENOM]]{{.*}} : f64
// CHECK:         return %[[DY]] : f64

// CHECK-LABEL: func.func private @diffeexp2(
// CHECK:         %[[LN2:.+]] = arith.constant 0.693147{{.*}} : f64
// CHECK:         %[[EXP2:.+]] = math.exp2 %arg0{{.*}} : f64
// CHECK:         %[[FACTOR:.+]] = arith.mulf %[[EXP2]], %[[LN2]]{{.*}} : f64
// CHECK:         %[[DY:.+]] = arith.mulf %arg1, %[[FACTOR]]{{.*}} : f64
// CHECK:         return %[[DY]] : f64
