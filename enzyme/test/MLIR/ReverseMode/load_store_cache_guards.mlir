// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s
// RUN: %eopt --split-input-file --enzyme %s | FileCheck %s --check-prefix=RAW

// Verify that loading a mutable type (!llvm.ptr) from an active pointer does
// not push a dangling address cache into the augmented forward pass.

llvm.func @load_ptr(%pp: !llvm.ptr) -> f64 {
  %p = llvm.load %pp : !llvm.ptr -> !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> f64
  %s = arith.mulf %v, %v : f64
  llvm.return %s : f64
}

func.func @dload_ptr(%pp: !llvm.ptr, %dpp: !llvm.ptr, %dr: f64) {
  enzyme.autodiff @load_ptr(%pp, %dpp, %dr) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!llvm.ptr, !llvm.ptr, f64) -> ()
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
  %r = enzyme.autodiff @store_vector(%m, %dm, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[] }> : (memref<10xvector<4xf32>>, memref<10xvector<4xf32>>, vector<4xf32>) -> vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL: func.func private @diffestore_vector
// CHECK: memref.load
// CHECK: return

// RAW-LABEL: func.func private @diffestore_vector(
// RAW: %[[MEMREF:.*]] = "enzyme.pop"{{.*}} -> memref<10xvector<4xf32>>
// RAW: %[[INDEX:.*]] = "enzyme.pop"{{.*}} -> index
// RAW: %[[LOADED:.*]] = memref.load %[[MEMREF]][%[[INDEX]]] : memref<10xvector<4xf32>>
// RAW: %[[CURRENT:.*]] = "enzyme.get"{{.*}} -> vector<4xf32>
// RAW: %[[ADDED:.*]] = arith.addf %[[CURRENT]], %[[LOADED]]{{.*}} : vector<4xf32>
// RAW: "enzyme.set"{{.*}}, %[[ADDED]])
// RAW: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// RAW: memref.store %[[ZERO]], %[[MEMREF]][%[[INDEX]]] : memref<10xvector<4xf32>>
// RAW: %[[RETURNED:.*]] = "enzyme.get"{{.*}} -> vector<4xf32>
// RAW: return %[[RETURNED]] : vector<4xf32>

// -----

// Verify that storing a VectorType in LLVM dialect works in reverse mode.

llvm.func @store_vector_llvm(%p: !llvm.ptr, %v: vector<4xf32>) {
  llvm.store %v, %p : vector<4xf32>, !llvm.ptr
  llvm.return
}

func.func @dstore_vector_llvm(%p: !llvm.ptr, %dp: !llvm.ptr, %v: vector<4xf32>) -> vector<4xf32> {
  %r = enzyme.autodiff @store_vector_llvm(%p, %dp, %v) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[] }> : (!llvm.ptr, !llvm.ptr, vector<4xf32>) -> vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL: llvm.func @diffestore_vector_llvm
// CHECK: llvm.load
// CHECK: llvm.return

// RAW-LABEL: llvm.func @diffestore_vector_llvm(
// RAW: %[[POINTER:.*]] = "enzyme.pop"{{.*}} -> !llvm.ptr
// RAW: %[[LLVM_LOADED:.*]] = llvm.load %[[POINTER]] : !llvm.ptr -> vector<4xf32>
// RAW: %[[LLVM_CURRENT:.*]] = "enzyme.get"{{.*}} -> vector<4xf32>
// RAW: %[[LLVM_ADDED:.*]] = arith.addf %[[LLVM_CURRENT]], %[[LLVM_LOADED]]{{.*}} : vector<4xf32>
// RAW: "enzyme.set"{{.*}}, %[[LLVM_ADDED]])
// RAW: %[[LLVM_ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// RAW: llvm.store %[[LLVM_ZERO]], %[[POINTER]] : vector<4xf32>, !llvm.ptr
// RAW: %[[LLVM_RETURNED:.*]] = "enzyme.get"{{.*}} -> vector<4xf32>
// RAW: llvm.return %[[LLVM_RETURNED]] : vector<4xf32>
