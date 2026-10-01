// RUN: %eopt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_dup,enzyme_dup retTys= mode=ForwardMode" --canonicalize | FileCheck %s

// Forward mode mirrors the primal on the shadows: the tangent of the added
// value joins the shadow location through the same atomic add.

module {
  func.func @f(%a: f64, %m: memref<?xf64>) {
    affine.for %i = 0 to 4 {
      %v = arith.mulf %a, %a : f64
      %old = "enzyme.affine_atomic_rmw"(%v, %m) <{ordering = 2 : i32, alignment = 8 : i64, fastmath = #arith.fastmath<fast>, kind = 0 : i64, map = affine_map<() -> (0)>}> : (f64, memref<?xf64>) -> f64
    }
    return
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<() -> (0)>
// CHECK-LABEL: func.func @f(
// CHECK-SAME: %[[a:.+]]: f64, %[[da:.+]]: f64, %[[m:[^ :]+]]: memref<?xf64>, %[[dm:[^ :]+]]: memref<?xf64>)
// CHECK: affine.for
// CHECK: %[[t0:.+]] = arith.mulf %[[da]], %[[a]] fastmath<fast> : f64
// CHECK: %[[t1:.+]] = arith.mulf %[[da]], %[[a]] fastmath<fast> : f64
// CHECK: %[[dv:.+]] = arith.addf %[[t0]], %[[t1]] fastmath<fast> : f64
// CHECK: %[[v:.+]] = arith.mulf %[[a]], %[[a]] : f64
// CHECK-DAG: enzyme.affine_atomic_rmw addf %[[dv]], %[[dm]], (#[[$MAP]]) [] monotonic fastmath<fast> {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
// CHECK-DAG: enzyme.affine_atomic_rmw addf %[[v]], %[[m]], (#[[$MAP]]) [] monotonic fastmath<fast> {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
