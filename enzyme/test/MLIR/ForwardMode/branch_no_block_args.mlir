// RUN: %eopt --enzyme --remove-unnecessary-enzyme-ops %s | FileCheck %s

// branchingForwardHandler copies the operands ahead of the first one any
// successor forwards -- for cf.cond_br that is the condition -- and it found
// that boundary by looking for a successor that forwards something. Neither
// block here takes an argument, so there was nothing to find and the boundary
// stayed at 0: the condition was dropped and the tangent came out as
//
//   "cf.cond_br"()[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 0, 0, 0>}>
//
// which fails the verifier with "expected 1 or more operands, but found 0".

module {
  func.func private @f(%p: !llvm.ptr, %n: i64) {
    %z = arith.constant 0 : i64
    %c = arith.cmpi eq, %n, %z : i64
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    llvm.intr.trap
    llvm.unreachable
  ^bb2:
    return
  }

  func.func @df(%p: !llvm.ptr, %dp: !llvm.ptr, %n: i64) {
    enzyme.fwddiff @f(%p, %dp, %n) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, i64) -> ()
    return
  }
}

// CHECK: func.func private @fwddiffef(%[[p:.+]]: !llvm.ptr, %[[dp:.+]]: !llvm.ptr, %[[n:.+]]: i64)
// CHECK:         %[[c:.+]] = arith.cmpi eq, %[[n]], %{{.+}} : i64
// CHECK:         cf.cond_br %[[c]], ^[[bb1:.+]], ^[[bb2:.+]]
// CHECK:       ^[[bb1]]:
// CHECK:         llvm.intr.trap
// CHECK:       ^[[bb2]]:
// CHECK:         return
