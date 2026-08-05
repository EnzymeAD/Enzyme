// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s

module {
  func.func @fma(%x: f64, %y: f64, %z: f64) -> f64 {
    %res = math.fma %x, %y, %z : f64
    return %res : f64
  }

  func.func @dfma(%x: f64, %dx: f64, %y: f64, %dy: f64, %z: f64, %dz: f64) -> f64 {
    %r = enzyme.fwddiff @fma(%x, %dx, %y, %dy, %z, %dz) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64, f64, f64, f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:  func.func private @fwddiffefma(%[[x:.+]]: f64, %[[dx:.+]]: f64, %[[y:.+]]: f64, %[[dy:.+]]: f64, %[[z:.+]]: f64, %[[dz:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[a:.+]] = arith.mulf %[[dx]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[b:.+]] = arith.mulf %[[dy]], %[[x]] fastmath<fast> : f64
// CHECK-NEXT:    %[[s:.+]] = arith.addf %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[r:.+]] = arith.addf %[[s]], %[[dz]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[r]] : f64
// CHECK-NEXT:  }
