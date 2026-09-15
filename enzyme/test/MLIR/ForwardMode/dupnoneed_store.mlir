// RUN: %eopt %s --enzyme-wrap="infn=stm outfn= argTys=enzyme_dup,enzyme_dupnoneed retTys= mode=ForwardMode" | FileCheck %s --check-prefix=MEMREF
// RUN: %eopt %s --enzyme-wrap="infn=stl outfn= argTys=enzyme_dup,enzyme_dupnoneed retTys= mode=ForwardMode" | FileCheck %s --check-prefix=LLVM
// RUN: %eopt %s --enzyme-wrap="infn=sta outfn= argTys=enzyme_dup,enzyme_dupnoneed retTys= mode=ForwardMode" | FileCheck %s --check-prefix=AFFINE
// RUN: %eopt %s --enzyme-wrap="infn=msz outfn= argTys=enzyme_dup,enzyme_dupnoneed retTys= mode=ForwardMode" | FileCheck %s --check-prefix=MEMSET
// RUN: %eopt %s --enzyme-wrap="infn=cst outfn= argTys=enzyme_dup,enzyme_dupnoneed retTys= mode=ForwardMode" | FileCheck %s --check-prefix=CAST
// RUN: %eopt %s --enzyme-wrap="infn=rdb outfn= argTys=enzyme_dup,enzyme_dupnoneed retTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=READBACK
// RUN: %eopt %s --enzyme-wrap="infn=stm outfn= argTys=enzyme_dup,enzyme_dup retTys= mode=ForwardMode" | FileCheck %s --check-prefix=DUP

// enzyme_dupnoneed on a pointer is the caller saying it will not use the
// primal contents. A store into such memory, never read back inside the
// function, need not happen: only the tangent store is emitted.

func.func @stm(%x: f64, %out: memref<f64>) {
  %sq = arith.mulf %x, %x : f64
  memref.store %sq, %out[] : memref<f64>
  return
}

// MEMREF:  func.func @stm(%arg0: f64, %arg1: f64, %arg2: memref<f64>, %arg3: memref<f64>) {
// MEMREF-NOT:  memref.store %{{.*}}, %arg2
// MEMREF:      memref.store %{{.*}}, %arg3[] : memref<f64>
// MEMREF-NOT:  memref.store
// MEMREF:      return

// A plain enzyme_dup argument makes no such promise: both stores stay.

// DUP:  func.func @stm(%arg0: f64, %arg1: f64, %arg2: memref<f64>, %arg3: memref<f64>) {
// DUP-DAG:  memref.store %{{.*}}, %arg2[] : memref<f64>
// DUP-DAG:  memref.store %{{.*}}, %arg3[] : memref<f64>
// DUP:      return

func.func @stl(%x: f64, %out: !llvm.ptr) {
  %sq = arith.mulf %x, %x : f64
  llvm.store %sq, %out : f64, !llvm.ptr
  return
}

// LLVM:  func.func @stl(%arg0: f64, %arg1: f64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
// LLVM-NOT:  llvm.store %{{.*}}, %arg2
// LLVM:      llvm.store %{{.*}}, %arg3 : f64, !llvm.ptr
// LLVM-NOT:  llvm.store
// LLVM:      return

func.func @sta(%x: f64, %out: memref<4xf64>) {
  affine.for %i = 0 to 4 {
    %sq = arith.mulf %x, %x : f64
    affine.store %sq, %out[%i] : memref<4xf64>
  }
  return
}

// AFFINE:  func.func @sta(%arg0: f64, %arg1: f64, %arg2: memref<4xf64>, %arg3: memref<4xf64>) {
// AFFINE-NOT:  affine.store %{{.*}}, %arg2
// AFFINE:      affine.store %{{.*}}, %arg3[%arg4] : memref<4xf64>
// AFFINE-NOT:  affine.store
// AFFINE:      return

// A memset writes primal contents too; under dupnoneed only the shadow
// clear survives.

func.func @msz(%x: f64, %out: !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : i8) : i8
  %sz = llvm.mlir.constant(8 : i64) : i64
  "llvm.intr.memset"(%out, %c0, %sz) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %sq = arith.mulf %x, %x : f64
  llvm.store %sq, %out : f64, !llvm.ptr
  return
}

// MEMSET:  func.func @msz(%arg0: f64, %arg1: f64, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
// MEMSET:      "llvm.intr.memset"(%arg3, %{{.*}}, %{{.*}}) <{isVolatile = false}>
// MEMSET-NOT:  llvm.intr.memset
// MEMSET-NOT:  llvm.store %{{.*}}, %arg2
// MEMSET:      llvm.store %{{.*}}, %arg3 : f64, !llvm.ptr
// MEMSET:      return

// The declaration is about the underlying object: it survives views and
// casts of the pointer.

func.func @cst(%x: f64, %out: memref<f64>) {
  %v = memref.cast %out : memref<f64> to memref<f64>
  %sq = arith.mulf %x, %x : f64
  memref.store %sq, %v[] : memref<f64>
  return
}

// CAST:  func.func @cst(%arg0: f64, %arg1: f64, %arg2: memref<f64>, %arg3: memref<f64>) {
// CAST-NOT:  memref.store %{{.*}}, %{{.*}}0
// CAST:      %[[SHADOW:.+]] = memref.cast %arg3
// CAST:      memref.store %{{.*}}, %[[SHADOW]][] : memref<f64>
// CAST-NOT:  memref.store
// CAST:      return

// The declaration covers the function's own reads too: dupnoneed is the
// caller saying nothing needs the primal contents of that memory, so the
// store goes even though a load follows it.

func.func @rdb(%x: f64, %out: memref<f64>) -> f64 {
  %sq = arith.mulf %x, %x : f64
  memref.store %sq, %out[] : memref<f64>
  %r = memref.load %out[] : memref<f64>
  return %r : f64
}

// READBACK:  func.func @rdb(%arg0: f64, %arg1: f64, %arg2: memref<f64>, %arg3: memref<f64>)
// READBACK-NOT:  memref.store %{{.*}}, %arg2
// READBACK:      memref.store %{{.*}}, %arg3[] : memref<f64>
// READBACK-NOT:  memref.store
// READBACK:      return
