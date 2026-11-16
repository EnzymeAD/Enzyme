// RUN: %eopt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func private @f
// CHECK-SAME: enzyme.arg_types = [#enzyme.type_tree<>, unit, #enzyme.type_tree<[]: bf16>]
// CHECK-SAME: enzyme.type = #enzyme.type_tree<[-1]: pointer, [-1, 0]: f64, [-1, 8]: integer, [-1, 16]: anything>
// CHECK-SAME: test.type_tree_entry = #enzyme.type_tree_entry<[-1, -1]: f32>
// An entry of a type tree also round-trips as an attribute of its own (under
// an arbitrary name).
func.func private @f() attributes {
  enzyme.type = #enzyme.type_tree<[-1]: pointer, [-1, 0]: f64, [-1, 8]: integer, [-1, 16]: anything>,
  enzyme.arg_types = [#enzyme.type_tree<>, unit, #enzyme.type_tree<[]: bf16>],
  test.type_tree_entry = #enzyme.type_tree_entry<[-1, -1]: f32>
}

// -----

// expected-error @below {{type tree paths must be unique and ordered}}
func.func private @unordered() attributes {enzyme.type = #enzyme.type_tree<[-1, 8]: integer, [-1]: pointer>}

// -----

// expected-error @below {{a type tree entry has a floating-point type exactly if it is a float}}
func.func private @not_float() attributes {enzyme.type = #enzyme.type_tree<[-1]: i32>}

// -----

// expected-error @below {{type tree offset -2 is below -1}}
func.func private @bad_offset() attributes {enzyme.type = #enzyme.type_tree<[-2]: pointer>}
