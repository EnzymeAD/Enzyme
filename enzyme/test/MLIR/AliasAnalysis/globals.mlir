// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

// The address of a global is a concrete alias class. Before, `llvm.mlir.addressof`
// had no transfer function at all, so its lattice stayed undefined and any query
// against it hit the "incomplete alias analysis" assertion.

// Two distinct globals are distinct objects: taking the address of one can never
// produce a pointer into the other, whatever their linkage.

// CHECK: "a" and "b": NoAlias
llvm.mlir.global external @ext1(0 : i64) : i64
llvm.mlir.global external @ext2(0 : i64) : i64
func.func @two_external_globals() {
  %a = llvm.mlir.addressof @ext1 {tag = "a"} : !llvm.ptr
  %b = llvm.mlir.addressof @ext2 {tag = "b"} : !llvm.ptr
  return
}

// -----

// CHECK: "a" and "b": NoAlias
llvm.mlir.global internal @g1(0 : i64) : i64
llvm.mlir.global internal @g2(0 : i64) : i64
func.func @two_internal_globals() {
  %a = llvm.mlir.addressof @g1 {tag = "a"} : !llvm.ptr
  %b = llvm.mlir.addressof @g2 {tag = "b"} : !llvm.ptr
  return
}

// -----

// Mixed linkage is no different: still two separate objects.

// CHECK: "a" and "b": NoAlias
llvm.mlir.global external @pub(0 : i64) : i64
llvm.mlir.global internal @priv(0 : i64) : i64
func.func @mixed_linkage() {
  %a = llvm.mlir.addressof @pub {tag = "a"} : !llvm.ptr
  %b = llvm.mlir.addressof @priv {tag = "b"} : !llvm.ptr
  return
}

// -----

// Conversely, two addresses of the *same* global are the same object. The class
// is keyed on the symbol rather than on the `llvm.mlir.addressof` result, which
// matters because the LLVM importer emits one `addressof` per use.

// CHECK: "a" and "b": MayAlias
llvm.mlir.global external @same(0 : i64) : i64
func.func @same_global_twice() {
  %a = llvm.mlir.addressof @same {tag = "a"} : !llvm.ptr
  %b = llvm.mlir.addressof @same {tag = "b"} : !llvm.ptr
  return
}

// -----

// A global keeps its own class across functions too.

// CHECK: "a" and "b": MayAlias
llvm.mlir.global external @shared(0 : i64) : i64
func.func @user1() {
  %a = llvm.mlir.addressof @shared {tag = "a"} : !llvm.ptr
  return
}
func.func @user2() {
  %b = llvm.mlir.addressof @shared {tag = "b"} : !llvm.ptr
  return
}

// -----

// An `llvm.mlir.addressof` of a function is not data storage; it stays in the
// shared entry class when the function may be defined elsewhere.

// CHECK: "a" and "b": MayAlias
llvm.func external @decl()
func.func @address_of_external_func() {
  %a = llvm.mlir.addressof @decl {tag = "a"} : !llvm.ptr
  %b = llvm.mlir.addressof @decl {tag = "b"} : !llvm.ptr
  return
}

// -----

// A global does not alias an unannotated pointer argument: argument classes and
// global classes are disjoint. Note that this asserts no incoming pointer points
// into a global, which a caller passing `&g` would violate. The analysis answered
// NoAlias here before this file existed too, but only because the addressof side
// was undefined rather than because anything had decided it.

// CHECK: "arg" and "glob": NoAlias
llvm.mlir.global external @escapes(0 : i64) : i64
func.func @global_vs_argument(%arg0: !llvm.ptr {enzyme.tag = "arg"}) {
  %glob = llvm.mlir.addressof @escapes {tag = "glob"} : !llvm.ptr
  return
}
