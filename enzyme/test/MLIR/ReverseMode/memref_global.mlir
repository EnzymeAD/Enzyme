// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s

// A private global that is only written is inactive, so its store does not
// need a shadow.
module {
  memref.global "private" @w : memref<f64> = dense<0.0>
  func.func @f(%x: f64) -> f64 {
    %p = memref.get_global @w : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @df(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @f(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @diffef(
// CHECK-SAME:      %[[X:.*]]: f64, %[[DR:.*]]: f64) -> f64 {
// CHECK:         %[[P:.*]] = memref.get_global @w : memref<f64>
// CHECK:         memref.store %[[X]], %[[P]][] : memref<f64>
// CHECK:         %[[A:.*]] = arith.mulf %[[DR]], %[[X]]
// CHECK:         %[[B:.*]] = arith.mulf %[[DR]], %[[X]]
// CHECK:         %[[DX:.*]] = arith.addf %[[A]], %[[B]]
// CHECK:         return %[[DX]] : f64

// -----

// Passing the global to a call known to be inactive does not read the active
// value back out, so the global stays inactive and the call is kept as is.
module {
  memref.global "private" @w : memref<f64> = dense<0.0>
  func.func private @print_buf(memref<f64>) attributes {enzyme_inactive}
  func.func @g(%x: f64) -> f64 {
    %p = memref.get_global @w : memref<f64>
    memref.store %x, %p[] : memref<f64>
    call @print_buf(%p) : (memref<f64>) -> ()
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @dg(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @g(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @diffeg(
// CHECK-SAME:      %[[X:.*]]: f64, %[[DR:.*]]: f64) -> f64 {
// CHECK:         %[[P:.*]] = memref.get_global @w : memref<f64>
// CHECK:         memref.store %[[X]], %[[P]][] : memref<f64>
// CHECK:         call @print_buf(%[[P]]) : (memref<f64>) -> ()
// CHECK:         %[[A:.*]] = arith.mulf %[[DR]], %[[X]]
// CHECK:         %[[B:.*]] = arith.mulf %[[DR]], %[[X]]
// CHECK:         %[[DX:.*]] = arith.addf %[[A]], %[[B]]
// CHECK:         return %[[DX]] : f64

// -----

// The stored value is read back, so the global is active. Before, it was
// inactive, and the gradient silently lost the path through @w.
module {
  memref.global @w : memref<f64> = dense<0.0>
  func.func @f(%x: f64) -> f64 {
    // expected-error @+1 {{could not compute the adjoint for this operation}}
    %p = memref.get_global @w : memref<f64>
    memref.store %x, %p[] : memref<f64>
    %v = memref.load %p[] : memref<f64>
    %r = arith.mulf %v, %v : f64
    return %r : f64
  }
  func.func @df(%x: f64, %dr: f64) -> f64 {
    %r = enzyme.autodiff @f(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}
