// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" | FileCheck %s

// Regression test for the EnzymeMLIR mincut "cache the last value in a chain"
// heuristic (Enzyme/MLIR/Passes/RemovalUtils.cpp, pushCachesDownstream).
//
// A non-recomputable value `a` (a memref load) feeds a straight-line chain
//
//   a -> c1 -> c2 -> c3
//
// of pure single-use operations, and only `c3` is used nonlinearly (squared),
// so the reverse pass genuinely needs exactly one value from this chain.
//
// Every cut of that chain has the same capacity -- one f32 -- so max flow alone
// does not say WHICH of a/c1/c2/c3 to cache. Extracting the cut from the set
// reachable from the sources picks the one nearest the roots (`a`), which is
// the worst choice: the reverse pass must then rebuild c1, c2 and c3. Caching
// `c3` instead costs exactly the same memory and recomputes nothing.
//
// pushCachesDownstream slides each cut edge downstream while the operation it
// crosses has a single graph user and a single result, and while the value it
// moves to is no larger. Here that walks a -> c1 -> c2 -> c3 and stops at `c3`,
// whose user is required by the reverse pass.
//
// This mirrors the "push to cache the last value in a computation chain"
// heuristic that the LLVM Enzyme mincut already applies in
// Enzyme/DifferentialUseAnalysis.cpp.

func.func private @chain(%x: memref<?xf32>, %ub: index) -> f32 {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %k1 = arith.constant 2.0 : f32
  %k2 = arith.constant 3.0 : f32
  %k3 = arith.constant 5.0 : f32
  %sum_0 = arith.constant 0.0 : f32
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%acc = %sum_0) -> (f32) {
    %a = memref.load %x[%iv] : memref<?xf32>
    %c1 = arith.mulf %a, %k1 : f32
    %c2 = arith.mulf %c1, %k2 : f32
    %c3 = arith.mulf %c2, %k3 : f32
    // Nonlinear use: the reverse pass needs %c3 itself.
    %s = arith.mulf %c3, %c3 : f32
    %acc_next = arith.addf %acc, %s : f32
    scf.yield %acc_next : f32
  }
  // Clobber %x after the loop so the load cannot be recovered by re-reading the
  // memref in the reverse pass; this forces the mincut to actually cache.
  memref.store %sum_0, %x[%lb] : memref<?xf32>
  return %sum : f32
}

func.func @dchain(%x: memref<?xf32>, %dx: memref<?xf32>, %ub: index, %dseed: f32) {
  enzyme.autodiff @chain(%x, %dx, %ub, %dseed) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (memref<?xf32>, memref<?xf32>, index, f32) -> ()
  return
}

// Still exactly one cache, of the same f32 element type -- the slide is free.
// CHECK-LABEL:   func.func private @diffechain(
// CHECK:           %[[ALLOC:.*]] = memref.alloc(%arg2) : memref<?xf32>
// CHECK-NOT:       memref.alloc
// CHECK:           scf.for
// The forward pass computes the whole chain and caches its LAST value, %c3 --
// not the load, and not either intermediate.
// CHECK:             %[[A:.*]] = memref.load %arg0[%arg4] : memref<?xf32>
// CHECK:             %[[C1:.*]] = arith.mulf %[[A]], %{{.*}} : f32
// CHECK:             %[[C2:.*]] = arith.mulf %[[C1]], %{{.*}} : f32
// CHECK:             %[[C3:.*]] = arith.mulf %[[C2]], %{{.*}} : f32
// CHECK:             enzyme.store %[[C3]], %[[ALLOC]]
// CHECK-NOT:         enzyme.store
// CHECK:           }
// CHECK:           memref.store %{{.*}}, %arg0
// CHECK:           scf.for
// The reverse pass reads %c3 back and feeds it straight into the adjoint of
// `%s = %c3 * %c3`. Before the slide the cache held `a` and these two lines
// were instead `mulf %[[LC3]], %k1` / `mulf %.., %k2` rebuilding the chain.
// CHECK:             %[[LC3:.*]] = enzyme.load %[[ALLOC]]
// CHECK-NOT:         enzyme.load
// CHECK-NEXT:        %[[G0:.*]] = arith.mulf %{{.*}}, %[[LC3]] : f32
// CHECK-NEXT:        %[[G1:.*]] = arith.mulf %{{.*}}, %[[LC3]] : f32
// CHECK-NEXT:        %[[G2:.*]] = arith.addf %[[G0]], %[[G1]] : f32
// The three multiplies that follow are the chain rule for c1/c2/c3, each
// against a constant -- not a recomputation of the forward chain.
// CHECK-NEXT:        arith.mulf %[[G2]], %{{.*}} : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[ALLOC]]
