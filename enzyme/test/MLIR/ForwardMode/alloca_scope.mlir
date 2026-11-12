// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @alloca_scope_fwd(%x: f64) -> f64 {
    %res = memref.alloca_scope -> (f64) {
      %sq = arith.mulf %x, %x : f64
      memref.alloca_scope.return %sq : f64
    }
    return %res : f64
  }

  func.func @dalloca_scope_fwd(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @alloca_scope_fwd(%x, %dx) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>]
    }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @fwddiffealloca_scope_fwd(
// CHECK-SAME: %[[X:.+]]: f64, %[[DX:.+]]: f64) -> f64 {
// CHECK-NEXT:   %[[SCOPE:.+]]:2 = memref.alloca_scope -> (f64, f64) {
// CHECK-NEXT:     %[[M0:.+]] = arith.mulf %[[DX]], %[[X]] fastmath<fast> : f64
// CHECK-NEXT:     %[[M1:.+]] = arith.mulf %[[DX]], %[[X]] fastmath<fast> : f64
// CHECK-NEXT:     %[[ADD:.+]] = arith.addf %[[M0]], %[[M1]] fastmath<fast> : f64
// CHECK-NEXT:     %[[SQ:.+]] = arith.mulf %[[X]], %[[X]] : f64
// CHECK-NEXT:     memref.alloca_scope.return %[[SQ]], %[[ADD]] : f64, f64
// CHECK:        return %[[SCOPE]]#1 : f64
