// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s

// Regression test for the EnzymeMLIR mincut node-splitting fix
// (Enzyme/MLIR/Passes/RemovalUtils.cpp, minCutValues).
//
// A single non-recomputable value `a` (a memref load) is used by four pure
// multiplies o1..o4. Those are combined linearly into two values p = o1 + o2
// and q = o3 + o4, which are each used nonlinearly (squared), so the reverse
// pass genuinely needs `p` and `q`.
//
// Because p and q are pure functions of `a`, the optimal cache stores the
// SINGLE value `a` per iteration and recomputes o1..o4, p, q in the reverse
// pass.
//
// The mincut must therefore recognize that caching `a` -- which is used by four
// operations -- costs one, not four. This is achieved by node-splitting the
// mincut graph: each value V is split into V_in -> V_out joined by a single
// unit-capacity edge, so all of V's flow funnels through it and V costs one to
// cache regardless of fan-out (mirroring the LLVM Enzyme mincut in
// Enzyme/DifferentialUseAnalysis.cpp). Without the split, `a` was charged four
// (one per use) and the mincut instead cached the two downstream values p, q.
//
// Below we check that exactly ONE value (`a`) is cached and that o1..o4/p/q are
// recomputed in the reverse pass.

func.func private @reduce(%x: memref<?xf32>, %ub: index) -> f32 {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %c1 = arith.constant 2.0 : f32
  %c2 = arith.constant 3.0 : f32
  %c3 = arith.constant 5.0 : f32
  %c4 = arith.constant 7.0 : f32
  %sum_0 = arith.constant 0.0 : f32
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%acc = %sum_0) -> (f32) {
    %a = memref.load %x[%iv] : memref<?xf32>
    %o1 = arith.mulf %a, %c1 : f32
    %o2 = arith.mulf %a, %c2 : f32
    %o3 = arith.mulf %a, %c3 : f32
    %o4 = arith.mulf %a, %c4 : f32
    %p = arith.addf %o1, %o2 : f32
    %q = arith.addf %o3, %o4 : f32
    %p2 = arith.mulf %p, %p : f32
    %q2 = arith.mulf %q, %q : f32
    %s = arith.addf %p2, %q2 : f32
    %acc_next = arith.addf %acc, %s : f32
    scf.yield %acc_next : f32
  }
  // Clobber %x after the loop so its loaded values cannot be recovered by
  // re-reading the memref in the reverse pass. This forces the mincut to
  // actually cache values, isolating the miscount we want to test.
  memref.store %sum_0, %x[%lb] : memref<?xf32>
  return %sum : f32
}

func.func @dreduce(%x: memref<?xf32>, %dx: memref<?xf32>, %ub: index, %dseed: f32) {
  enzyme.autodiff @reduce(%x, %dx, %ub, %dseed) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (memref<?xf32>, memref<?xf32>, index, f32) -> ()
  return
}

// Exactly one cache is allocated (for `a`); there is no second alloc.
// CHECK-LABEL:   func.func private @differeduce(
// CHECK:           %[[ALLOC:.*]] = memref.alloc(%arg2) : memref<?xf32>
// CHECK-NOT:       memref.alloc
// CHECK:           scf.for
// The single cached value is `a` itself (the load), stored once per iteration.
// CHECK:             %[[A:.*]] = memref.load %arg0[%arg4] : memref<?xf32>
// CHECK:             enzyme.store %[[A]], %[[ALLOC]]
// CHECK-NOT:         enzyme.store
// CHECK:           }
// The post-loop clobber prevents recovering the load by re-reading %arg0.
// CHECK:           memref.store %{{.*}}, %arg0
// CHECK:           scf.for
// The reverse pass loads `a` once and recomputes o1..o4, p, q from it.
// CHECK:             %[[LA:.*]] = enzyme.load %[[ALLOC]]
// CHECK-NOT:         enzyme.load
// CHECK:             arith.mulf %[[LA]], %cst_2 : f32
// CHECK:             arith.mulf %[[LA]], %cst_1 : f32
// CHECK:             arith.mulf %[[LA]], %cst_0 : f32
// CHECK:             arith.mulf %[[LA]], %cst : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC]]
