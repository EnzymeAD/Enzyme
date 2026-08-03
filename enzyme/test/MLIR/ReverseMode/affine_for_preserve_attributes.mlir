// RUN: %eopt %s --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// Same for affine.for: widening the loop to carry the cache index must not
// discard what was set on it.

module {
  func.func @reduce(%x: f32) -> (f32) {
    %sum_0 = arith.constant 1.0 : f32

    %sum = affine.for %iv = 0 to 128
        iter_args(%sum_iter = %sum_0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %x : f32
      affine.yield %sum_next : f32
    } {enzyme.marker}

    return %sum : f32
  }
}

// CHECK-LABEL: func.func @reduce
// CHECK: affine.for
// CHECK: } {enzyme.marker}
// CHECK: affine.for
// CHECK: } {enzyme.marker}
