// RUN: %eopt %s --enzyme-wrap="infn=square outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

// The same value is yielded for two iter_args. Its adjoint in the reverse body
// must be the sum of both incoming carried adjoints, not just the last one.

module {
  func.func @square(%x: f32) -> f32 {
    %r:2 = affine.for %iv = 0 to 4 iter_args(%a = %x, %b = %x) -> (f32, f32) {
      %n = arith.mulf %a, %b : f32
      affine.yield %n, %n : f32, f32
    }
    return %r#0 : f32
  }
}

// CHECK-LABEL: func.func @square(%arg0: f32, %arg1: f32) -> f32 {
// CHECK:         %[[REV:.+]]:2 = affine.for %{{.+}} = 0 to 4 iter_args(%[[DA:.+]] = %arg1, %[[DB:.+]] = %cst) -> (f32, f32) {
// CHECK:           %[[DN:.+]] = arith.addf %[[DA]], %[[DB]] fastmath<fast> : f32
// CHECK-NEXT:      %[[GA:.+]] = arith.mulf %[[DN]], %{{.+}} fastmath<fast> : f32
// CHECK-NEXT:      %[[GB:.+]] = arith.mulf %[[DN]], %{{.+}} fastmath<fast> : f32
// CHECK-NEXT:      affine.yield %[[GA]], %[[GB]] : f32, f32
// CHECK-NEXT:    }
// CHECK:         %[[DX:.+]] = arith.addf %[[REV]]#0, %[[REV]]#1 fastmath<fast> : f32
// CHECK-NEXT:    return %[[DX]] : f32
