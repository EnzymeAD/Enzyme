// RUN: %eopt %s --enzyme-wrap="infn=f outfn= argTys=enzyme_dup,enzyme_const retTys= mode=ForwardMode" --canonicalize | FileCheck %s

// Storing an inactive pointer into differentiated memory: a mutable value
// nothing differentiates is its own shadow, so the shadow slot holds the
// primal pointer and structural fields read the same through the shadow
// object.

module {
  func.func @f(%m: memref<?x!llvm.ptr>, %p: !llvm.ptr) {
    affine.store %p, %m[0] : memref<?x!llvm.ptr>
    return
  }
}

// CHECK-LABEL: func.func @f(
// CHECK-SAME: %[[m:[^ :]+]]: memref<?x!llvm.ptr>, %[[dm:[^ :]+]]: memref<?x!llvm.ptr>, %[[p:.+]]: !llvm.ptr)
// CHECK-DAG: affine.store %[[p]], %[[dm]][0] : memref<?x!llvm.ptr>
// CHECK-DAG: affine.store %[[p]], %[[m]][0] : memref<?x!llvm.ptr>
