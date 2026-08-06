// RUN: %eopt --enzyme %s | FileCheck %s

// A comdat says which of the identical copies of a symbol the linker should
// keep, and it keeps or discards the whole group at once. The derivative is
// cloned from the primal and so arrived holding the primal's comdat, which put
// it in that group -- but only the translation units that differentiate the
// primal put a derivative in it. A unit that merely calls the primal offers a
// group of one under the same key, and if that is the copy the linker keeps,
// the derivative goes with the copy it discarded:
//
//   undefined reference to `fwddiffe_ZNK4mfem6future14tensor_ndarray...'
//
// even though the object that made the call also defined the symbol. Five of
// MFEM's unit-test objects offered the derivative-less group for the same key
// (EnzymeAD/Enzyme-JAX#2778).

module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @f any
  }

  llvm.func @f(%p: !llvm.ptr) comdat(@__llvm_global_comdat::@f) {
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    llvm.return
  }

  func.func @df(%p: !llvm.ptr, %dp: !llvm.ptr) {
    enzyme.fwddiff @f(%p, %dp) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// The derivative gets a group of its own, keyed on its own name, so it still
// dedupes against the other units that built the same derivative and nothing
// else can take it away. The kind of deduplication is the primal's.

// CHECK:       llvm.comdat @__llvm_global_comdat {
// CHECK-DAG:     llvm.comdat_selector @f any
// CHECK-DAG:     llvm.comdat_selector @fwddiffef any
// CHECK:       }
// CHECK:       llvm.func @f(%{{.+}}: !llvm.ptr) comdat(@__llvm_global_comdat::@f)
// CHECK:       llvm.func @fwddiffef(%{{.+}}: !llvm.ptr, %{{.+}}: !llvm.ptr) comdat(@__llvm_global_comdat::@fwddiffef)
