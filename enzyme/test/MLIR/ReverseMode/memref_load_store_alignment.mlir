// RUN: %eopt %s --enzyme-wrap="infn=loadstore outfn= argTys=enzyme_dup,enzyme_dup retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --flatten-enzyme-caches --canonicalize --enzyme-simplify-math --canonicalize --cse | FileCheck %s

// An over-aligned memref access keeps its alignment in the reverse pass: the
// shadow load/store the load and store adjoints build must carry the same
// alignment as the primal, or the shadow buffer is accessed under-aligned.

func.func @loadstore(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v = memref.load %arg0[%c0] {alignment = 16 : i64} : memref<?xf32>
  %sq = arith.mulf %v, %v : f32
  memref.store %sq, %arg1[%c1] {alignment = 16 : i64} : memref<?xf32>
  return
}

// CHECK-LABEL: func.func @loadstore(
// The store adjoint loads and zeroes the store's shadow slot.
// CHECK:         memref.load %arg3[%c1] {alignment = 16 : i64} : memref<?xf32>
// CHECK:         memref.store %cst, %arg3[%c1] {alignment = 16 : i64} : memref<?xf32>
// The load adjoint accumulates into the load's shadow slot.
// CHECK:         memref.load %arg1[%c0] {alignment = 16 : i64} : memref<?xf32>
// CHECK:         memref.store %{{.+}}, %arg1[%c0] {alignment = 16 : i64} : memref<?xf32>
