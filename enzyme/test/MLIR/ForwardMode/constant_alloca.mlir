// RUN: %eopt --pass-pipeline="builtin.module(enzyme{dataflow},canonicalize,remove-unnecessary-enzyme-ops)" %s | FileCheck %s

// The alloca here only ever holds what was read out of the constant argument,
// so it is inactive and the store into it has nothing to differentiate. The
// dataflow analyzer says so; the older analyzer, which knows the alloca only by
// its type, does not. visitChild's skip asked the older one directly while
// deciding the rest of the same condition with the dataflow one, so the store
// was not skipped and reached memoryIdentityForwardHandler with a pointer that
// has no shadow to be made of it:
//
//   Unsupported constant arg to memory identity forward handler(opidx=0, ...)

module {
  llvm.func @e(%p: !llvm.ptr, %c: !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i64) : i64
    %a = llvm.alloca %one x f64 : (i64) -> !llvm.ptr
    %v = llvm.load %c : !llvm.ptr -> f64
    llvm.store %v, %a : f64, !llvm.ptr
    %r = llvm.load %a : !llvm.ptr -> f64
    llvm.store %r, %c : f64, !llvm.ptr
    %w = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %w, %w : f64
    llvm.store %s, %p : f64, !llvm.ptr
    llvm.return
  }

  func.func @de(%p: !llvm.ptr, %dp: !llvm.ptr, %c: !llvm.ptr) {
    enzyme.fwddiff @e(%p, %dp, %c) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}

// The alloca traffic is carried through untouched -- one alloca, no shadow of
// it -- and only the active argument gets a tangent.

// CHECK: llvm.func @fwddiffee(%[[p:.+]]: !llvm.ptr, %[[dp:.+]]: !llvm.ptr, %[[c:.+]]: !llvm.ptr)
// CHECK:         %[[one:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[a:.+]] = llvm.alloca %[[one]] x f64 : (i64) -> !llvm.ptr
// CHECK-NOT:     llvm.alloca
// CHECK:         %[[v:.+]] = llvm.load %[[c]] : !llvm.ptr -> f64
// CHECK:         llvm.store %[[v]], %[[a]] : f64, !llvm.ptr
// CHECK:         %[[r:.+]] = llvm.load %[[a]] : !llvm.ptr -> f64
// CHECK:         llvm.store %[[r]], %[[c]] : f64, !llvm.ptr
// CHECK:         %[[dw:.+]] = llvm.load %[[dp]] : !llvm.ptr -> f64
// CHECK:         %[[w:.+]] = llvm.load %[[p]] : !llvm.ptr -> f64
// CHECK:         %[[m1:.+]] = arith.mulf %[[dw]], %[[w]] fastmath<fast> : f64
// CHECK:         %[[m2:.+]] = arith.mulf %[[dw]], %[[w]] fastmath<fast> : f64
// CHECK:         %[[ds:.+]] = arith.addf %[[m1]], %[[m2]] fastmath<fast> : f64
// CHECK:         %[[s:.+]] = arith.mulf %[[w]], %[[w]] : f64
// CHECK:         llvm.store %[[ds]], %[[dp]] : f64, !llvm.ptr
// CHECK:         llvm.store %[[s]], %[[p]] : f64, !llvm.ptr
