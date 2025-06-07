// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

func.func @load(%x: !llvm.ptr) -> f64 {
  %res = llvm.load %x : !llvm.ptr -> f64
  return %res : f64
}

func.func @dload(%x: !llvm.ptr, %dx: !llvm.ptr, %dres: f64) {
  enzyme.autodiff @load(%x, %dx, %dres) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (!llvm.ptr, !llvm.ptr, f64) -> ()
  return
}

// CHECK:  func.func private @diffeload(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: f64) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.addf %arg2, %cst : f64
// CHECK-NEXT:    %1 = llvm.atomicrmw fadd %arg1, %0 monotonic : !llvm.ptr, f64
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

func.func @store(%x: f64, %ptr: !llvm.ptr) {
  llvm.store %x, %ptr : f64, !llvm.ptr
  return
}

func.func @dstore(%x: f64, %ptr: !llvm.ptr, %dptr: !llvm.ptr) -> f64 {
  %0 = enzyme.autodiff @store(%x, %ptr, %dptr) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (f64, !llvm.ptr, !llvm.ptr) -> f64
  return %0 : f64
}

// CHECK:   func.func private @diffestore(%arg0: f64, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    llvm.store %arg0, %arg1 : f64, !llvm.ptr
// CHECK-NEXT:    %0 = llvm.load %arg2 : !llvm.ptr -> f64
// CHECK-NEXT:    %1 = arith.addf %0, %cst : f64
// CHECK-NEXT:    return %1 : f64
// CHECK-NEXT:  }
