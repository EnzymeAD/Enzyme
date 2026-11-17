// RUN: %eopt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llvm.mlir.global external @glob()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<[-1]: pointer, [-1, 0]: f64, [-1, 8]: integer, [-1, 16]: anything>
llvm.mlir.global external @glob() {enzyme.type = #enzyme.type_tree<[-1]: pointer, [-1, 0]: f64, [-1, 8]: integer, [-1, 16]: anything>} : !llvm.array<24 x i8>

// Entries are sorted by path.
// CHECK-LABEL: llvm.mlir.global external @unsorted()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<[-1]: pointer, [-1, 0]: f32, [-1, 4]: integer>
llvm.mlir.global external @unsorted() {enzyme.type = #enzyme.type_tree<[-1, 4]: integer, [-1]: pointer, [-1, 0]: f32>} : !llvm.array<8 x i8>

llvm.func @g(!llvm.ptr, !llvm.ptr)

// CHECK-LABEL: llvm.func @f
// CHECK-SAME: enzyme.ret_types = [#enzyme.type_tree<[]: bf16>]
// CHECK: llvm.call @g
// CHECK-SAME: enzyme.arg_types = [#enzyme.type_tree<>, unit]
llvm.func @f(%p: !llvm.ptr) -> bf16 attributes {enzyme.ret_types = [#enzyme.type_tree<[]: bf16>]} {
  llvm.call @g(%p, %p) {enzyme.arg_types = [#enzyme.type_tree<>, unit]} : (!llvm.ptr, !llvm.ptr) -> ()
  %v = llvm.load %p : !llvm.ptr -> bf16
  llvm.return %v : bf16
}

// -----

// expected-error @below {{type tree path [-1, 8] appears more than once}}
llvm.mlir.global external @duplicate() {enzyme.type = #enzyme.type_tree<[-1, 8]: integer, [-1]: pointer, [-1, 8]: f32>} : !llvm.array<16 x i8>

// -----

// expected-error @below {{a type tree entry has a floating-point type exactly if it is a float}}
llvm.mlir.global external @not_float() {enzyme.type = #enzyme.type_tree<[-1]: i32>} : !llvm.array<4 x i8>

// -----

// expected-error @below {{type tree offset -2 is below -1}}
llvm.mlir.global external @bad_offset() {enzyme.type = #enzyme.type_tree<[-2]: pointer>} : !llvm.array<4 x i8>
