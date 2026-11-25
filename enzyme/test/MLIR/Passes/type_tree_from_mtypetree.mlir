// RUN: %eopt %s -test-type-tree-attr | FileCheck %s

// CHECK-LABEL: llvm.func @scalar()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: f32}>
llvm.func @scalar() attributes {test.type_tree = "{[-1]:Float@f32}"}

// CHECK-LABEL: llvm.func @pointer()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: pointer {0: f64, 8: integer, 16: anything}}>
llvm.func @pointer() attributes {test.type_tree = "{[-1]:Pointer, [-1,0]:Float@f64, [-1,8]:Integer, [-1,16]:Anything}"}

// CHECK-LABEL: llvm.func @nested()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: pointer {0: pointer {-1: bf16}, 8: f64}}>
llvm.func @nested() attributes {test.type_tree = "{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@bf16, [-1,8]:Float@f64}"}

// The root has a type of its own.
// CHECK-LABEL: llvm.func @root()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<pointer {0: integer}>
llvm.func @root() attributes {test.type_tree = "{[]:Pointer, [0]:Integer}"}

// Entries of unknown type are left out, and so is a tree without any other.
// CHECK-LABEL: llvm.func @unknown()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{8: integer}>
llvm.func @unknown() attributes {test.type_tree = "{[0]:Unknown, [8]:Integer}"}

// CHECK-LABEL: llvm.func @empty()
// CHECK-NOT: enzyme.type
// CHECK-NOT: test.type_tree
llvm.func @empty() attributes {test.type_tree = "{}"}
