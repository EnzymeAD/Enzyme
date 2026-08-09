// RUN: %eopt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_dup,enzyme_dupnoneed retTys= mode=ForwardMode" | FileCheck %s

// A pointer whose base the caller declared enzyme_dupnoneed keeps that
// declaration through a call: the callee is where the stores live, and it
// can only skip their primal halves if it is told. The cast in between
// checks the declaration is looked up on the underlying object.

func.func private @leaf(%x: f64, %out: memref<f64>) {
  %sq = arith.mulf %x, %x : f64
  memref.store %sq, %out[] : memref<f64>
  return
}

func.func @f(%x: f64, %out: memref<f64>) {
  %c = memref.cast %out : memref<f64> to memref<f64>
  call @leaf(%x, %c) : (f64, memref<f64>) -> ()
  return
}

// CHECK:  func.func @f(%arg0: f64, %arg1: f64, %arg2: memref<f64>, %arg3: memref<f64>) {
// CHECK:      call @fwddiffeleaf(%arg0, %arg1, %{{.*}}, %{{.*}}) : (f64, f64, memref<f64>, memref<f64>) -> ()

// CHECK:  func.func private @fwddiffeleaf(%arg0: f64, %arg1: f64, %arg2: memref<f64>, %arg3: memref<f64>) {
// CHECK-NOT:  memref.store %{{.*}}, %arg2
// CHECK:      memref.store %{{.*}}, %arg3[] : memref<f64>
// CHECK-NOT:  memref.store
// CHECK:      return
