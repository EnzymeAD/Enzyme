// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

// Reverse-mode coverage for the complex.create -> complex.im composition.
// f(t) = im(create(0, t)) = t, a smooth real identity, so df/dt must be +1.
// The existing complex.mlir pins re/im standalone and create in the forward
// position; this consumer path (create feeding im) is not covered. With the
// gradient correct, im's adjoint negation cancels against the Wirtinger conj
// wrap, so no arith.negf survives and the returned cotangent is the seed.
module {
  func.func @main(%t: f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    %z = complex.create %zero, %t : complex<f32>
    %o = complex.im %z : complex<f32>
    return %o : f32
  }
}

// f(t) = im(create(0, t)) = t, so df/dt must be +1. The fix negates the
// complex.im result, undoing the conj the reverse wrap applies to create's
// incoming cotangent; without it the returned cotangent traces to -seed.
// CHECK-LABEL: func.func @main(%arg0: f32, %arg1: f32) -> f32
// CHECK: %[[IM:.+]] = complex.im
// CHECK: arith.negf %[[IM]]
