// RUN: %eopt %s --enzyme-wrap="infn=memref_zero outfn= argTys=enzyme_dup retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=MEMREF
// RUN: %eopt %s --enzyme-wrap="infn=affine_zero_loop outfn= argTys=enzyme_dup,enzyme_const retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=AFFINE
// RUN: %eopt %s --enzyme-wrap="infn=llvm_zero_then_load outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=LLVM

// Storing an inactive value (here a constant) into active memory transfers no
// derivative, but it overwrites the location and so kills the derivative of
// what was there: the reverse pass must still zero the shadow of the
// overwritten location, even though no adjoint flows into the stored value.

func.func @memref_zero(%arg0: memref<f32>) {
  %zero = arith.constant 0.0 : f32
  memref.store %zero, %arg0[] : memref<f32>
  return
}

func.func @affine_zero_loop(%arg0: memref<?xf32>, %ub: index) {
  %zero = arith.constant 0.0 : f32
  affine.for %iv = 0 to %ub {
    affine.store %zero, %arg0[2 * %iv] : memref<?xf32>
  }
  return
}

// The result is always 0, so the derivative with respect to the initial
// contents of %arg0 is 0: the load adjoint accumulates into the shadow, then
// the store adjoint must reset it.
func.func @llvm_zero_then_load(%arg0: !llvm.ptr) -> f32 {
  %zero = llvm.mlir.constant(0.0 : f32) : f32
  llvm.store %zero, %arg0 : f32, !llvm.ptr
  %v = llvm.load %arg0 : !llvm.ptr -> f32
  return %v : f32
}

// MEMREF-LABEL: func.func @memref_zero(
// MEMREF-SAME:      %[[X:.+]]: memref<f32>, %[[DX:.+]]: memref<f32>)
// MEMREF-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// MEMREF-NEXT:    memref.store %[[ZERO]], %[[X]][] : memref<f32>
// MEMREF-NEXT:    memref.store %[[ZERO]], %[[DX]][] : memref<f32>
// MEMREF-NEXT:    return

// AFFINE-LABEL: func.func @affine_zero_loop(
// AFFINE-SAME:      %[[X:.+]]: memref<?xf32>, %[[DX:.+]]: memref<?xf32>, %[[UB:.+]]: index)
// AFFINE-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// AFFINE-NEXT:    affine.for %[[IV:.+]] = 0 to %[[UB]] {
// AFFINE-NEXT:      affine.store %[[ZERO]], %[[X]][%[[IV]] * 2] : memref<?xf32>
// AFFINE-NEXT:    }
// AFFINE-NEXT:    affine.for %[[RIV:.+]] = 0 to %[[UB]] {
// AFFINE-NEXT:      %[[IDX:.+]] = affine.apply #{{.+}}(%[[RIV]])[%[[UB]]]
// AFFINE-NEXT:      memref.store %[[ZERO]], %[[DX]][%[[IDX]]] : memref<?xf32>
// AFFINE-NEXT:    }
// AFFINE-NEXT:    return

// LLVM-LABEL: func.func @llvm_zero_then_load(
// LLVM-SAME:      %[[X:.+]]: !llvm.ptr, %[[DX:.+]]: !llvm.ptr, %[[DRET:.+]]: f32)
// LLVM:           %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// LLVM:           %[[D:.+]] = llvm.load %[[DX]] : !llvm.ptr -> f32
// LLVM-NEXT:      %[[SUM:.+]] = arith.addf %[[D]], %[[DRET]]
// LLVM-NEXT:      llvm.store %[[SUM]], %[[DX]] : f32, !llvm.ptr
// LLVM-NEXT:      llvm.store %[[ZERO]], %[[DX]] : f32, !llvm.ptr
// LLVM-NEXT:      return
