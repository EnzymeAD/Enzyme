// RUN: %eopt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// Same structural story as the llvm.store of a pointer: the shadow slot must
// hold the shadow pointer, so the shadow load traverses to the shadow data.

module {
  llvm.func @f(%p: !llvm.ptr) -> f64 {
    %c0 = arith.constant 0 : index
    %a = memref.alloca() : memref<1x!llvm.ptr>
    memref.store %p, %a[%c0] : memref<1x!llvm.ptr>
    %q = memref.load %a[%c0] : memref<1x!llvm.ptr>
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
// CHECK: memref.store %[[dp]], %[[sa:.+]][%{{.+}}] : memref<1x!llvm.ptr>
// CHECK: memref.store %[[p]], %[[a:.+]][%{{.+}}] : memref<1x!llvm.ptr>
// CHECK: %[[sq:.+]] = memref.load %[[sa]][%{{.+}}] : memref<1x!llvm.ptr>
// CHECK: %[[q:.+]] = memref.load %[[a]][%{{.+}}] : memref<1x!llvm.ptr>
// CHECK: %[[v:.+]] = llvm.load %[[q]] : !llvm.ptr -> f64
// CHECK: %[[prev:.+]] = llvm.load %[[sq]] : !llvm.ptr -> f64
// CHECK: %[[acc:.+]] = arith.addf %[[prev]], %{{.+}} fastmath<fast> : f64
// CHECK: llvm.store %[[acc]], %[[sq]] : f64, !llvm.ptr
