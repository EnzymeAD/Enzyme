// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// After a memset the memory holds a fixed byte pattern, which depends on no
// input, so its tangent is zero everywhere the memset reached. Forward mode
// says that by clearing the shadow over the same range.
//
// llvm.intr.memset is declared InactiveOp, which attaches an ActivityOpInterface
// and nothing else, so forward mode had no rule for it and stopped at "could not
// compute the adjoint for this operation". Being inactive is about what an op
// makes active, not about whether the derivative needs it said.

module {
  llvm.func @f(%p: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i8) : i8
    %n = llvm.mlir.constant(8 : i64) : i64
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    "llvm.intr.memset"(%p, %c0, %n) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.return
  }

  func.func @df(%p: !llvm.ptr, %dp: !llvm.ptr) {
    enzyme.fwddiff @f(%p, %dp) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @fwddiffef(%[[p:.+]]: !llvm.ptr, %[[dp:.+]]: !llvm.ptr)
// CHECK-DAG:     %[[zero:.+]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-DAG:     %[[len:.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK:         llvm.store %{{.+}}, %[[dp]] : f64, !llvm.ptr
// CHECK:         llvm.store %{{.+}}, %[[p]] : f64, !llvm.ptr
// CHECK-DAG:     "llvm.intr.memset"(%[[dp]], %[[zero]], %[[len]])
// CHECK-DAG:     "llvm.intr.memset"(%[[p]], %[[zero]], %[[len]])

// -----

// The byte written says nothing about the tangent: whatever pattern the primal
// gets, the shadow gets zeros.

module {
  llvm.func @g(%p: !llvm.ptr) {
    %c7 = llvm.mlir.constant(7 : i8) : i8
    %n = llvm.mlir.constant(8 : i64) : i64
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    "llvm.intr.memset"(%p, %c7, %n) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.return
  }

  func.func @dg(%p: !llvm.ptr, %dp: !llvm.ptr) {
    enzyme.fwddiff @g(%p, %dp) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @fwddiffeg(%[[p2:.+]]: !llvm.ptr, %[[dp2:.+]]: !llvm.ptr)
// CHECK-DAG:     %[[seven:.+]] = llvm.mlir.constant(7 : i8) : i8
// CHECK-DAG:     %[[zero2:.+]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-DAG:     "llvm.intr.memset"(%[[dp2]], %[[zero2]], %{{.+}})
// CHECK-DAG:     "llvm.intr.memset"(%[[p2]], %[[seven]], %{{.+}})

// -----

// Memory nothing differentiates has no shadow to clear, so there is only the
// one memset.

module {
  llvm.func @h(%p: !llvm.ptr, %c: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i8) : i8
    %n = llvm.mlir.constant(8 : i64) : i64
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    "llvm.intr.memset"(%c, %c0, %n) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.return
  }

  func.func @dh(%p: !llvm.ptr, %dp: !llvm.ptr, %c: !llvm.ptr) {
    enzyme.fwddiff @h(%p, %dp, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @fwddiffeh(%[[p3:.+]]: !llvm.ptr, %[[dp3:.+]]: !llvm.ptr, %[[c:.+]]: !llvm.ptr)
// CHECK:         "llvm.intr.memset"(%[[c]], %{{.+}}, %{{.+}})
// CHECK-NOT:     llvm.intr.memset
