// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @log2(%x: f64) -> f64 {
    %y = math.log2 %x : f64
    return %y : f64
  }

  func.func @dlog2(%x: f64, %dx: f64) -> f64 {
    %dy = enzyme.fwddiff @log2(%x, %dx) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (f64, f64) -> f64
    return %dy : f64
  }

  func.func @exp2(%x: f64) -> f64 {
    %y = math.exp2 %x : f64
    return %y : f64
  }

  func.func @dexp2(%x: f64, %dx: f64) -> f64 {
    %dy = enzyme.fwddiff @exp2(%x, %dx) {
      activity = [#enzyme<activity enzyme_dup>],
      ret_activity = [#enzyme<activity enzyme_dupnoneed>]
    } : (f64, f64) -> f64
    return %dy : f64
  }
}

// CHECK-LABEL: func.func private @fwddiffelog2(
// CHECK:         %[[TWO:.+]] = arith.constant 2.0000{{.*}} : f64
// CHECK:         %[[LN2:.+]] = math.log %[[TWO]]{{.*}} : f64
// CHECK:         %[[DENOM:.+]] = arith.mulf %arg0, %[[LN2]]{{.*}} : f64
// CHECK:         %[[DY:.+]] = arith.divf %arg1, %[[DENOM]]{{.*}} : f64
// CHECK:         return %[[DY]] : f64

// CHECK-LABEL: func.func private @fwddiffeexp2(
// CHECK:         %[[EXP2:.+]] = math.exp2 %arg0{{.*}} : f64
// CHECK:         %[[TWO:.+]] = arith.constant 2.0000{{.*}} : f64
// CHECK:         %[[LN2:.+]] = math.log %[[TWO]]{{.*}} : f64
// CHECK:         %[[FACTOR:.+]] = arith.mulf %[[EXP2]], %[[LN2]]{{.*}} : f64
// CHECK:         %[[DY:.+]] = arith.mulf %arg1, %[[FACTOR]]{{.*}} : f64
// CHECK:         return %[[DY]] : f64
