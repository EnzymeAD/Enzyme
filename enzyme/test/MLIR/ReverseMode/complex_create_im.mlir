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

// The reverse-mode cotangent of t must be the seed itself (df/dt = +1).
// A returned value that traces to -seed means the create-as-consumer path
// applied an extra Wirtinger conj and flipped the gradient sign.
// CHECK-LABEL: func.func @main(%arg0: f32, %arg1: f32) -> f32
// CHECK: return %arg1 : f32
