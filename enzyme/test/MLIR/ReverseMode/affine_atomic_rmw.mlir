// RUN: %eopt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_active,enzyme_dup retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// The primal op atomically adds into the location and yields the pre-add
// value; here the result is unused, so backward the location's adjoint
// simply flows to the added value: d_v += dm[i], read off the shadow at the
// same affine position. The load is plain -- the reverse sweep mirrors the
// forward structure, so whatever ordered the primal's atomics against this
// thread's reads orders the shadow's atomics against this load.

module {
  func.func @f(%a: f64, %m: memref<?xf64>) {
    affine.parallel (%i) = (0) to (4) {
      %v = arith.mulf %a, %a : f64
      %old = "enzyme.affine_atomic_rmw"(%v, %m) <{ordering = 2 : i32, alignment = 8 : i64, fastmath = #arith.fastmath<fast>, kind = 0 : i64, map = affine_map<() -> (0)>}> : (f64, memref<?xf64>) -> f64
    }
    return
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<() -> (0)>
// CHECK-LABEL: func.func @f(
// CHECK-SAME: %[[a:.+]]: f64, %[[m:[^ :]+]]: memref<?xf64>, %[[dm:[^ :]+]]: memref<?xf64>) -> f64
// CHECK: %[[sq:.+]] = arith.mulf %[[a]], %[[a]] : f64
// CHECK: affine.parallel
// CHECK: enzyme.affine_atomic_rmw addf %[[sq]], %[[m]], (#[[$MAP]]) [] monotonic fastmath<fast> {alignment = 8 : i64} : (f64, memref<?xf64>) -> f64
// CHECK: %[[red:.+]] = affine.parallel (%{{.+}}) = (0) to (4) reduce ("addf") -> (f64)
// CHECK: %[[g:.+]] = affine.load %[[dm]][0] {alignment = 8 : i64} : memref<?xf64>
// CHECK: %[[t0:.+]] = arith.mulf %[[g]], %[[a]] fastmath<fast> : f64
// CHECK: %[[t1:.+]] = arith.mulf %[[g]], %[[a]] fastmath<fast> : f64
// CHECK: %[[t2:.+]] = arith.addf %[[t0]], %[[t1]] fastmath<fast> : f64
// CHECK: affine.yield %[[t2]] : f64
// CHECK: return %[[red]] : f64
