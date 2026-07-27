// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s

// This test documents a suboptimality (logic error) in the EnzymeMLIR mincut
// (Enzyme/MLIR/Passes/RemovalUtils.cpp, minCutCache).
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
// The mincut graph, however, uses one node per value with an edge per *use*:
// `a` has four outgoing edges (a->o1..o4), so cutting "at a" costs 4 edges even
// though it represents caching just one value. The downstream cut {p, q} costs
// only 2 edges, so the minimum-edge-cut picks {p, q} and caches TWO values per
// iteration instead of one.
//
// The LLVM Enzyme mincut (Enzyme/DifferentialUseAnalysis.cpp) avoids this by
// node-splitting: each value V becomes V_in->V_out with a single capacity-1
// edge, so a value used N times still costs 1 to cache. The MLIR version has no
// such split, hence the miscount reproduced below (two memref.alloc's).

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

// The optimal result would allocate ONE cache and recompute p,q in reverse.
// The current mincut allocates TWO caches (for p and q) -- this is the bug.

// CHECK-LABEL:   func.func private @differeduce(
// CHECK:           %[[ALLOC:.*]] = memref.alloc(%arg2) : memref<?xf32>
// CHECK:           %[[ALLOC2:.*]] = memref.alloc(%arg2) : memref<?xf32>
// CHECK:           scf.for
// CHECK:             %[[A:.*]] = memref.load %arg0[%arg4] : memref<?xf32>
// CHECK:             %[[O1:.*]] = arith.mulf %[[A]], %cst_2 : f32
// CHECK:             %[[O2:.*]] = arith.mulf %[[A]], %cst_1 : f32
// CHECK:             %[[O3:.*]] = arith.mulf %[[A]], %cst_0 : f32
// CHECK:             %[[O4:.*]] = arith.mulf %[[A]], %cst : f32
// CHECK:             %[[P:.*]] = arith.addf %[[O1]], %[[O2]] : f32
// CHECK:             enzyme.store %[[P]], %[[ALLOC]]
// CHECK:             %[[Q:.*]] = arith.addf %[[O3]], %[[O4]] : f32
// CHECK:             enzyme.store %[[Q]], %[[ALLOC2]]
// CHECK:           }
// The post-loop clobber prevents recovering the loads by re-reading %arg0.
// CHECK:           memref.store %{{.*}}, %arg0
// CHECK:           scf.for
// CHECK:             %[[LP:.*]] = enzyme.load %[[ALLOC]]
// CHECK:             %[[LQ:.*]] = enzyme.load %[[ALLOC2]]
