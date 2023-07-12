// RUN: %eopt --enzyme %s | FileCheck %s

module {
  // This moves active data from x to y
  func.func private @dps(%x: !llvm.ptr, %y: !llvm.ptr) -> f64 {
    %0 = llvm.load %x : !llvm.ptr -> f64
    llvm.store %0, %y : f64, !llvm.ptr
    return %0 : f64
  }

  func.func @ddps(%x: !llvm.ptr, %dx: !llvm.ptr, %y: !llvm.ptr, %dy: !llvm.ptr) {
    "enzyme.autodiff"(%x, %dx, %y, %dy) {activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dupnoneed>], fn = @dps} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    return
  }
}
