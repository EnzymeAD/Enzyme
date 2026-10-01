// RUN: %eopt %s --enzyme-wrap="infn=stm outfn= argTys=enzyme_active,enzyme_dupnoneed retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s --check-prefix=MEMREF
// RUN: %eopt %s --enzyme-wrap="infn=stl outfn= argTys=enzyme_active,enzyme_dupnoneed retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s --check-prefix=LLVM
// RUN: %eopt %s --enzyme-wrap="infn=sta outfn= argTys=enzyme_active,enzyme_dupnoneed retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s --check-prefix=AFFINE

// enzyme_dupnoneed in reverse mode: the augmented forward pass need not
// store the primal value either -- nothing in the function reads it back,
// and the adjoint of the store reads only the shadow.

func.func @stm(%x: f64, %out: memref<f64>) {
  %sq = arith.mulf %x, %x : f64
  memref.store %sq, %out[] : memref<f64>
  return
}

// MEMREF:  func.func @stm(%arg0: f64, %arg1: memref<f64>, %arg2: memref<f64>) -> f64 {
// MEMREF-NEXT:    %[[CST:.+]] = arith.constant 0.000000e+00 : f64
// MEMREF-NEXT:    %[[G:.+]] = memref.load %arg2[] : memref<f64>
// MEMREF-NEXT:    memref.store %[[CST]], %arg2[] : memref<f64>
// MEMREF-NEXT:    %[[M:.+]] = arith.mulf %[[G]], %arg0 fastmath<fast> : f64
// MEMREF-NEXT:    %[[A:.+]] = arith.addf %[[M]], %[[M]] fastmath<fast> : f64
// MEMREF-NEXT:    return %[[A]] : f64

func.func @stl(%x: f64, %out: !llvm.ptr) {
  %sq = arith.mulf %x, %x : f64
  llvm.store %sq, %out : f64, !llvm.ptr
  return
}

// LLVM:  func.func @stl(%arg0: f64, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> f64 {
// LLVM-NEXT:    %[[CST:.+]] = arith.constant 0.000000e+00 : f64
// LLVM-NEXT:    %[[G:.+]] = llvm.load %arg2 : !llvm.ptr -> f64
// LLVM-NEXT:    llvm.store %[[CST]], %arg2 : f64, !llvm.ptr
// LLVM-NEXT:    %[[M:.+]] = arith.mulf %[[G]], %arg0 fastmath<fast> : f64
// LLVM-NEXT:    %[[A:.+]] = arith.addf %[[M]], %[[M]] fastmath<fast> : f64
// LLVM-NEXT:    return %[[A]] : f64

func.func @sta(%x: f64, %out: memref<4xf64>) {
  affine.for %i = 0 to 4 {
    %sq = arith.mulf %x, %x : f64
    affine.store %sq, %out[%i] : memref<4xf64>
  }
  return
}

// AFFINE:  func.func @sta(%arg0: f64, %arg1: memref<4xf64>, %arg2: memref<4xf64>) -> f64 {
// AFFINE:        affine.for %{{.*}} = 0 to 4 iter_args(%{{.*}}) -> (f64) {
// AFFINE-NOT:      affine.store
// AFFINE:          memref.load %arg2[%{{.*}}] : memref<4xf64>
// AFFINE-NEXT:     memref.store %{{.*}}, %arg2[%{{.*}}] : memref<4xf64>
// AFFINE-NOT:      affine.store
// AFFINE:          affine.yield
