// RUN: %eopt %s --enzyme-wrap="infn=square_ip outfn= argTys=enzyme_dup,enzyme_const retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --flatten-enzyme-caches --canonicalize --enzyme-simplify-math --canonicalize --cse | FileCheck %s

// An over-aligned affine access carries its alignment (the discardable
// "alignment" attribute the raising path records) into the reverse pass: the
// shadow load/store the load and store adjoints build must keep it, or the
// shadow buffer is accessed under-aligned. The atomic path already forwarded
// it; the non-atomic path now does too.

func.func @square_ip(%arg0: memref<?xf32>, %ub: index) {
  affine.for %iv = 0 to %ub {
    %v = affine.load %arg0[2 * %iv] {alignment = 16 : i64} : memref<?xf32>
    %sq = arith.mulf %v, %v : f32
    affine.store %sq, %arg0[2 * %iv + 1] {alignment = 16 : i64} : memref<?xf32>
    affine.yield
  }
  return
}

// CHECK-LABEL: func.func @square_ip(
// The store adjoint loads and zeroes the shadow slot.
// CHECK:         memref.load %arg1[%{{.+}}] {alignment = 16 : i64} : memref<?xf32>
// CHECK:         memref.store %cst, %arg1[%{{.+}}] {alignment = 16 : i64} : memref<?xf32>
// The load adjoint accumulates into the shadow slot.
// CHECK:         memref.load %arg1[%{{.+}}] {alignment = 16 : i64} : memref<?xf32>
// CHECK:         memref.store %{{.+}}, %arg1[%{{.+}}] {alignment = 16 : i64} : memref<?xf32>
