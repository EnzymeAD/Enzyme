// RUN: %eopt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llvm.mlir.global external @glob()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: pointer {0: f64, 8: integer, 16: anything}}>
llvm.mlir.global external @glob() {enzyme.type = #enzyme.type_tree<{-1: pointer {0: f64, 8: integer, 16: anything}}>} : !llvm.array<24 x i8>

// Subtrees are sorted by offset.
// CHECK-LABEL: llvm.mlir.global external @unsorted()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: pointer {0: f32, 4: integer}}>
llvm.mlir.global external @unsorted() {enzyme.type = #enzyme.type_tree<{-1: pointer {4: integer, 0: f32}}>} : !llvm.array<8 x i8>

llvm.func @g(!llvm.ptr, !llvm.ptr)

// CHECK-LABEL: llvm.func @f
// CHECK-SAME: enzyme.ret_types = [#enzyme.type_tree<bf16>]
// CHECK: llvm.call @g
// CHECK-SAME: enzyme.arg_types = [#enzyme.type_tree<>, unit]
llvm.func @f(%p: !llvm.ptr) -> bf16 attributes {enzyme.ret_types = [#enzyme.type_tree<bf16>]} {
  llvm.call @g(%p, %p) {enzyme.arg_types = [#enzyme.type_tree<>, unit]} : (!llvm.ptr, !llvm.ptr) -> ()
  %v = llvm.load %p : !llvm.ptr -> bf16
  llvm.return %v : bf16
}

// -----

// expected-error @below {{type tree offset 8 appears more than once}}
llvm.mlir.global external @duplicate() {enzyme.type = #enzyme.type_tree<{-1: pointer {8: integer, 8: f32}}>} : !llvm.array<16 x i8>

// -----

// expected-error @below {{a type tree node has a floating-point type exactly if it is a float}}
llvm.mlir.global external @not_float() {enzyme.type = #enzyme.type_tree<{-1: i32}>} : !llvm.array<4 x i8>

// -----

// expected-error @below {{type tree offset -2 is below -1}}
llvm.mlir.global external @bad_offset() {enzyme.type = #enzyme.type_tree<{-2: pointer}>} : !llvm.array<4 x i8>

// -----

// expected-error @below {{type tree is empty at offset -1}}
llvm.mlir.global external @empty_subtree() {enzyme.type = #enzyme.type_tree<{-1: {}}>} : !llvm.array<4 x i8>

// -----

// expected-error @below {{enzyme.type must be a type tree}}
llvm.mlir.global external @not_tree() {enzyme.type = 1 : i64} : !llvm.array<4 x i8>

// -----

// expected-error @below {{enzyme.type must be a type tree}}
llvm.func @arg_not_tree(%p: !llvm.ptr {enzyme.type = 1 : i64}) {
  llvm.return
}

// -----

// expected-error @below {{enzyme.arg_types only applies to calls}}
llvm.mlir.global external @not_call() {enzyme.arg_types = []} : !llvm.array<4 x i8>

// -----

llvm.func @g(!llvm.ptr, !llvm.ptr)

llvm.func @arg_count(%p: !llvm.ptr) {
  // expected-error @below {{enzyme.arg_types has 1 entries for 2 arguments}}
  llvm.call @g(%p, %p) {enzyme.arg_types = [unit]} : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

llvm.func @g(!llvm.ptr)

llvm.func @arg_not_trees(%p: !llvm.ptr) {
  // expected-error @below {{enzyme.arg_types must be an array of type trees or unit}}
  llvm.call @g(%p) {enzyme.arg_types = [1 : i64]} : (!llvm.ptr) -> ()
  llvm.return
}

// -----

// expected-error @below {{enzyme.ret_types only applies to functions}}
llvm.mlir.global external @not_function() {enzyme.ret_types = []} : !llvm.array<4 x i8>

// -----

// expected-error @below {{enzyme.ret_types has 1 entries for 0 results}}
llvm.func @ret_count() attributes {enzyme.ret_types = [unit]} {
  llvm.return
}
