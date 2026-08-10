// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// In forward mode the tangent of a memcpy is a memcpy of the shadows: float
// bytes carry their tangents, pointer bytes carry their shadow pointers, and
// one copy serves both.

module {
  llvm.func @f(%p: !llvm.ptr, %q: !llvm.ptr) {
    %n = llvm.mlir.constant(8 : i64) : i64
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    "llvm.intr.memcpy"(%q, %p, %n) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }

  func.func @df(%p: !llvm.ptr, %dp: !llvm.ptr, %q: !llvm.ptr, %dq: !llvm.ptr) {
    enzyme.fwddiff @f(%p, %dp, %q, %dq) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @fwddiffef(%[[p:.+]]: !llvm.ptr, %[[dp:.+]]: !llvm.ptr, %[[q:.+]]: !llvm.ptr, %[[dq:.+]]: !llvm.ptr)
// CHECK-DAG:     "llvm.intr.memcpy"(%[[dq]], %[[dp]], %{{.+}})
// CHECK-DAG:     "llvm.intr.memcpy"(%[[q]], %[[p]], %{{.+}})

// -----

// A source nothing differentiates has no shadow to copy from; the
// destination is an alloca of floats, whose tangent over the copied range
// is zero.

module {
  llvm.func @g(%p: !llvm.ptr, %c: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(8 : i64) : i64
    %a = llvm.alloca %c1 x f64 : (i32) -> !llvm.ptr
    "llvm.intr.memcpy"(%a, %c, %n) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %v = llvm.load %a : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    llvm.return
  }

  func.func @dg(%p: !llvm.ptr, %dp: !llvm.ptr, %c: !llvm.ptr) {
    enzyme.fwddiff @g(%p, %dp, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @fwddiffeg(%[[p2:.+]]: !llvm.ptr, %[[dp2:.+]]: !llvm.ptr, %[[c2:.+]]: !llvm.ptr)
// CHECK-DAG:     "llvm.intr.memset"(%[[sa:.+]], %{{.+}}, %{{.+}})
// CHECK-DAG:     "llvm.intr.memcpy"(%[[a:.+]], %[[c2]], %{{.+}})

// -----

// Memory nothing differentiates has no shadow to write, so there is only the
// one memcpy.

module {
  llvm.func @h(%p: !llvm.ptr, %c: !llvm.ptr) {
    %n = llvm.mlir.constant(8 : i64) : i64
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    "llvm.intr.memcpy"(%c, %p, %n) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return
  }

  func.func @dh(%p: !llvm.ptr, %dp: !llvm.ptr, %c: !llvm.ptr) {
    enzyme.fwddiff @h(%p, %dp, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @fwddiffeh(%[[p3:.+]]: !llvm.ptr, %[[dp3:.+]]: !llvm.ptr, %[[c3:.+]]: !llvm.ptr)
// CHECK:         "llvm.intr.memcpy"(%[[c3]], %[[p3]], %{{.+}})
// CHECK-NOT:     llvm.intr.memcpy
