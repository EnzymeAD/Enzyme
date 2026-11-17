// REQUIRES: enzymemlir-translate
// RUN: %etranslate %s -enzyme-to-llvmir | FileCheck %s

// CHECK: @glob = external global [16 x i8], !enzyme_type ![[GLOB:[0-9]+]]
llvm.mlir.global external @glob() {enzyme.type = #enzyme.type_tree<{-1: pointer {0: f64, 8: integer}}>} : !llvm.array<16 x i8>

llvm.func @g(!llvm.ptr, !llvm.ptr)

// CHECK: define "enzyme_type"="{[-1]:Float@double}" double @f(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float}" %[[P:[0-9]+]])
// CHECK: %[[X:[0-9]+]] = alloca i32, i64 1, align 4, !enzyme_type ![[LOCAL:[0-9]+]]
// CHECK: alloca i32, i64 2, align 4{{$}}
// CHECK: load float, ptr %[[P]], align 4, !enzyme_type ![[VAL:[0-9]+]]
// CHECK: call void @g(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float}" %[[P]], ptr %[[X]])
llvm.func @f(%p: !llvm.ptr {enzyme.type = #enzyme.type_tree<{-1: pointer {0: f32}}>}) -> f64 attributes {
    enzyme.ret_types = [#enzyme.type_tree<{-1: f64}>]} {
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %x = llvm.alloca %c1 x i32 {alignment = 4 : i64, enzyme.type = #enzyme.type_tree<{-1: pointer {0: integer}}>} : (i64) -> !llvm.ptr
  %y = llvm.alloca %c2 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
  %v = llvm.load %p {enzyme.type = #enzyme.type_tree<{-1: f32}>} : !llvm.ptr -> f32
  llvm.call @g(%p, %x) {enzyme.arg_types = [#enzyme.type_tree<{-1: pointer {-1: f32}}>, unit]} : (!llvm.ptr, !llvm.ptr) -> ()
  %r = llvm.fpext %v : f32 to f64
  llvm.return %r : f64
}

// CHECK-DAG: ![[GLOB]] = !{!"Unknown", i32 -1, ![[GLOBP:[0-9]+]]}
// CHECK-DAG: ![[GLOBP]] = !{!"Pointer", i32 0, ![[DOUBLE:[0-9]+]], i32 8, ![[INT:[0-9]+]]}
// CHECK-DAG: ![[DOUBLE]] = !{!"Float@double"}
// CHECK-DAG: ![[INT]] = !{!"Integer"}
// CHECK-DAG: ![[LOCAL]] = !{!"Unknown", i32 -1, ![[LOCALP:[0-9]+]]}
// CHECK-DAG: ![[LOCALP]] = !{!"Pointer", i32 0, ![[INT]]}
// CHECK-DAG: ![[VAL]] = !{!"Unknown", i32 -1, ![[FLOAT:[0-9]+]]}
// CHECK-DAG: ![[FLOAT]] = !{!"Float@float"}
