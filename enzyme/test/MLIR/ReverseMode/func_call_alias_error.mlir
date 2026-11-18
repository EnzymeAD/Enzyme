// RUN: %eopt --enzyme --verify-diagnostics --split-input-file %s

// Two distinct memref function arguments may alias, and the callee writes one
// while reading the other. Cloning them independently would break aliasing on
// replay, so reject the call.
module {
  func.func @alias_write(%a: memref<f32>, %b: memref<f32>, %x: f32) -> f32 {
    memref.store %x, %a[] : memref<f32>
    %v = memref.load %b[] : memref<f32>
    return %v : f32
  }
  func.func @caller(%a: memref<f32>, %b: memref<f32>, %x: f32) -> f32 {
    // expected-error @below {{cannot differentiate a call in reverse mode whose operands may alias a written argument: "alias_write"}}
    %r = func.call @alias_write(%a, %b, %x) : (memref<f32>, memref<f32>, f32) -> f32
    return %r : f32
  }
  func.func @d_caller(%a: memref<f32>, %da: memref<f32>, %b: memref<f32>, %db: memref<f32>, %x: f32, %dr: f32) -> f32 {
    %0 = enzyme.autodiff @caller(%a, %da, %b, %db, %x, %dr) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, f32, f32) -> f32
    return %0 : f32
  }
}

// -----

// A raw !llvm.ptr function argument has unknown extent, so it cannot be cloned.
module {
  llvm.func @scale(%p: !llvm.ptr, %f: f64) {
    %v = llvm.load %p : !llvm.ptr -> f64
    %s = arith.mulf %v, %f : f64
    llvm.store %s, %p : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @f(%x: !llvm.ptr, %f: f64) {
    // expected-error @below {{cannot differentiate a call in reverse mode: cannot clone overwritten argument of callee "scale"}}
    llvm.call @scale(%x, %f) : (!llvm.ptr, f64) -> ()
    llvm.return
  }
  func.func @df(%x: !llvm.ptr, %dx: !llvm.ptr, %f: f64) -> f64 {
    %0 = enzyme.autodiff @f(%x, %dx, %f) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>], ret_activity=[] }> : (!llvm.ptr, !llvm.ptr, f64) -> f64
    return %0 : f64
  }
}
