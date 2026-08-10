// RUN: %eopt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// Same structural story as the llvm.store of a pointer: the shadow slot must
// hold the shadow pointer at the same affine position.

module {
  llvm.func @f(%p: !llvm.ptr) -> f64 {
    %a = memref.alloca() : memref<1x!llvm.ptr>
    affine.store %p, %a[0] : memref<1x!llvm.ptr>
    %q = affine.load %a[0] : memref<1x!llvm.ptr>
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
// CHECK: affine.store %[[dp]], %[[sa:.+]][0] : memref<1x!llvm.ptr>
// CHECK: affine.store %[[p]], %[[a:.+]][0] : memref<1x!llvm.ptr>
// CHECK: %[[sq:.+]] = affine.load %[[sa]][0] : memref<1x!llvm.ptr>
// CHECK: %[[q:.+]] = affine.load %[[a]][0] : memref<1x!llvm.ptr>
// CHECK: %[[v:.+]] = llvm.load %[[q]] : !llvm.ptr -> f64
// CHECK: %[[prev:.+]] = llvm.load %[[sq]] : !llvm.ptr -> f64
// CHECK: %[[acc:.+]] = arith.addf %[[prev]], %{{.+}} fastmath<fast> : f64
// CHECK: llvm.store %[[acc]], %[[sq]] : f64, !llvm.ptr
