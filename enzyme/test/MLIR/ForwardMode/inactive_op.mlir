// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// A lifetime marker says when the memory it names is live. The shadow is memory
// too and lives exactly as long, so the tangent is the same marker said again of
// the shadow.
//
// llvm.intr.lifetime.start is declared InactiveOp, which attaches an
// ActivityOpInterface and nothing else, so forward mode had no rule for it and
// stopped at "could not compute the adjoint for this operation". Being inactive
// is about what an op makes active, not about whether the derivative needs it
// said -- a barrier is inactive too and still has to be there.

module {
  llvm.func @f(%p: !llvm.ptr) {
    "llvm.intr.lifetime.start"(%p) : (!llvm.ptr) -> ()
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    "llvm.intr.lifetime.end"(%p) : (!llvm.ptr) -> ()
    llvm.return
  }

  func.func @df(%p: !llvm.ptr, %dp: !llvm.ptr) {
    enzyme.fwddiff @f(%p, %dp) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// Both the primal and its shadow are marked, and the tangent of v*v is 2*v*dv.

// CHECK: llvm.func @fwddiffef(%[[p:.+]]: !llvm.ptr, %[[dp:.+]]: !llvm.ptr)
// CHECK-DAG:     llvm.intr.lifetime.start %[[dp]] : !llvm.ptr
// CHECK-DAG:     llvm.intr.lifetime.start %[[p]] : !llvm.ptr
// CHECK:         %[[dv:.+]] = llvm.load %[[dp]] : !llvm.ptr -> f64
// CHECK:         %[[v:.+]] = llvm.load %[[p]] : !llvm.ptr -> f64
// CHECK:         %[[a:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:         %[[b:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK:         %[[ds:.+]] = arith.addf %[[a]], %[[b]] fastmath<fast> : f64
// CHECK:         llvm.store %[[ds]], %[[dp]] : f64, !llvm.ptr
// CHECK-DAG:     llvm.intr.lifetime.end %[[dp]] : !llvm.ptr
// CHECK-DAG:     llvm.intr.lifetime.end %[[p]] : !llvm.ptr

// -----

// Memory nothing differentiates has no shadow whose lifetime this could be, so
// there is only the one marker.

module {
  llvm.func @g(%p: !llvm.ptr, %c: !llvm.ptr) {
    "llvm.intr.lifetime.start"(%c) : (!llvm.ptr) -> ()
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.store %s, %p : f64, !llvm.ptr
    "llvm.intr.lifetime.end"(%c) : (!llvm.ptr) -> ()
    llvm.return
  }

  func.func @dg(%p: !llvm.ptr, %dp: !llvm.ptr, %c: !llvm.ptr) {
    enzyme.fwddiff @g(%p, %dp, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// CHECK: llvm.func @fwddiffeg(%[[p2:.+]]: !llvm.ptr, %[[dp2:.+]]: !llvm.ptr, %[[c:.+]]: !llvm.ptr)
// CHECK:         llvm.intr.lifetime.start %[[c]] : !llvm.ptr
// CHECK-NOT:     llvm.intr.lifetime.start
// CHECK:         llvm.intr.lifetime.end %[[c]] : !llvm.ptr
