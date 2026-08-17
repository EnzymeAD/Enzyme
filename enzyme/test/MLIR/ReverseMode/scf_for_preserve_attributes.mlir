// RUN: %eopt %s --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// The removal pass widens both loops to carry the cache index by building a new
// loop and taking the old one's region. That is the same loop with extra
// iteration arguments, so whatever was set on it has to survive. enzyme.marker
// stands in for any attribute that is not re-derived from somewhere else --
// enzyme.disable_mincut happens to be backstopped by hasMinCut walking parent
// ops, which is why dropping it went unnoticed.

func.func @reduce(%x: f32, %ub: index) -> (f32) {
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %sum_0 = arith.constant 1.0 : f32
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (f32) {
    %sum_next = arith.mulf %sum_iter, %x : f32
    scf.yield %sum_next : f32
  } {enzyme.cache_use_tensor, enzyme.disable_mincut, enzyme.marker}
  return %sum : f32
}

// Both the widened forward loop and the widened reverse loop keep them.
// CHECK-LABEL: func.func @reduce
// CHECK: scf.for
// CHECK: } {enzyme.cache_use_tensor, enzyme.disable_mincut, enzyme.marker}
// CHECK: scf.for
// CHECK: } {enzyme.cache_use_tensor, enzyme.disable_mincut, enzyme.marker}
