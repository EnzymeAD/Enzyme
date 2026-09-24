// RUN: %eopt --split-input-file --enzyme --canonicalize %s | FileCheck %s

// Verify that loading a mutable type (!llvm.ptr) from an active pointer does
// not push a dangling address cache into the augmented forward pass.

llvm.func @load_ptr(%pp: !llvm.ptr) -> f64 {
  %p = llvm.load %pp : !llvm.ptr -> !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> f64
  %s = arith.mulf %v, %v : f64
  llvm.return %s : f64
}

func.func @dload_ptr(%pp: !llvm.ptr, %dpp: !llvm.ptr, %dr: f64) {
  enzyme.autodiff @load_ptr(%pp, %dpp, %dr) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (!llvm.ptr, !llvm.ptr, f64) -> ()
  return
}

// CHECK-LABEL: llvm.func @diffeload_ptr
// Before the fix, load of !llvm.ptr pushed an address cache that was never popped.
// With the fix, no enzyme.init / enzyme.push cache is created for the ptr load address.
// CHECK-NOT: enzyme.init
// CHECK-NOT: enzyme.push
// CHECK: llvm.return

// -----

// Verify that storing a VectorType works in reverse mode with AutoDiffTypeInterface.

func.func @store_vector(%m: memref<10xvector<4xf32>>, %v: vector<4xf32>) {
  %c0 = arith.constant 0 : index
  memref.store %v, %m[%c0] : memref<10xvector<4xf32>>
  return
}

func.func @dstore_vector(%m: memref<10xvector<4xf32>>, %dm: memref<10xvector<4xf32>>, %v: vector<4xf32>) -> vector<4xf32> {
  %r = enzyme.autodiff @store_vector(%m, %dm, %v) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_active>], ret_activity=[] } : (memref<10xvector<4xf32>>, memref<10xvector<4xf32>>, vector<4xf32>) -> vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL: func.func private @diffestore_vector
// CHECK: memref.load
// CHECK: return
