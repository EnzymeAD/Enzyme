// RUN: %eopt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// Storing an active pointer into memory has no float adjoint; its derivative
// story is structural: the shadow slot must hold the shadow pointer, so the
// later shadow load traverses to the shadow data, where the actual adjoint
// accumulates. Previously nothing wrote the shadow slot and the shadow load
// read back a null data pointer.

module {
  llvm.func @f(%p: !llvm.ptr) -> f64 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.alloca %c1 x !llvm.struct<(ptr)> : (i32) -> !llvm.ptr
    llvm.store %p, %a : !llvm.ptr, !llvm.ptr
    %q = llvm.load %a : !llvm.ptr -> !llvm.ptr
    %v = llvm.load %q : !llvm.ptr -> f64
    %s = arith.mulf %v, %v : f64
    llvm.return %s : f64
  }

  func.func @df(%p: !llvm.ptr, %dp: !llvm.ptr, %dr: f64) {
    enzyme.autodiff @f(%p, %dp, %dr) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (!llvm.ptr, !llvm.ptr, f64) -> ()
    return
  }
}

// CHECK: llvm.func @diffef(%[[p:.+]]: !llvm.ptr, %[[dp:.+]]: !llvm.ptr, %[[dr:.+]]: f64)
// The shadow pointer takes the same path through the shadow struct as the
// primal pointer does through the primal struct.
// CHECK: llvm.store %[[dp]], %[[sa:.+]] : !llvm.ptr, !llvm.ptr
// CHECK: llvm.store %[[p]], %[[a:.+]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[sq:.+]] = llvm.load %[[sa]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[q:.+]] = llvm.load %[[a]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[v:.+]] = llvm.load %[[q]] : !llvm.ptr -> f64
// CHECK: %[[t0:.+]] = arith.mulf %[[dr]], %[[v]] fastmath<fast> : f64
// CHECK: %[[t1:.+]] = arith.mulf %[[dr]], %[[v]] fastmath<fast> : f64
// CHECK: %[[t2:.+]] = arith.addf %[[t0]], %[[t1]] fastmath<fast> : f64
// CHECK: %[[prev:.+]] = llvm.load %[[sq]] : !llvm.ptr -> f64
// CHECK: %[[acc:.+]] = arith.addf %[[prev]], %[[t2]] fastmath<fast> : f64
// CHECK: llvm.store %[[acc]], %[[sq]] : f64, !llvm.ptr
