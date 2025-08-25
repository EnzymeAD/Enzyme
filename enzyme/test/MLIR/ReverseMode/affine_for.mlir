// RUN: %eopt %s  --enzyme-wrap="infn=reduce outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {
  func.func @reduce(%x: f32) -> (f32) {
    %sum_0 = arith.constant 1.0 : f32

    %sum = affine.for %iv = 0 to 128
        iter_args(%sum_iter = %sum_0) -> (f32) {
      %sum_next = arith.mulf %sum_iter, %x : f32
      affine.yield %sum_next : f32
    }

    return %sum : f32
  }
}

// CHECK:  func.func @reduce(%[[X:.+]]: f32, %[[DR:.+]]: f32) -> f32 {
// CHECK-NEXT:    %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[G:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f32>
// CHECK-NEXT:    "enzyme.set"(%[[G]], %[[ZERO]]) : (!enzyme.Gradient<f32>, f32) -> ()
// CHECK-NEXT:    %[[C1:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f32>
// CHECK-NEXT:    %[[C2:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f32>
// CHECK-NEXT:    %[[PRIMAL:.+]] = affine.for %arg2 = 0 to 128 iter_args(%[[CARRIED:.+]] = %[[ONE]]) -> (f32) {
// CHECK-NEXT:      "enzyme.push"(%[[C2]], %[[CARRIED]]) : (!enzyme.Cache<f32>, f32) -> ()
// CHECK-NEXT:      "enzyme.push"(%[[C1]], %[[X]]) : (!enzyme.Cache<f32>, f32) -> ()
// CHECK-NEXT:      %[[NEW:.+]] = arith.mulf %[[CARRIED]], %[[X]] : f32
// CHECK-NEXT:      affine.yield %[[NEW]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = affine.for %arg2 = 0 to 128 iter_args(%[[DIT:.+]] = %[[DR]]) -> (f32) {
// CHECK-NEXT:      %[[V1:.+]] = "enzyme.pop"(%[[C2]]) : (!enzyme.Cache<f32>) -> f32
// CHECK-NEXT:      %[[V2:.+]] = "enzyme.pop"(%[[C1]]) : (!enzyme.Cache<f32>) -> f32
// CHECK-NEXT:      %[[NEW_DIT:.+]] = arith.mulf %[[DIT]], %[[V2]] : f32
// CHECK-NEXT:      %[[NEW_D0:.+]] = arith.mulf %[[DIT]], %[[V1]] : f32
// CHECK-NEXT:      %[[D0:.+]] = "enzyme.get"(%[[G]]) : (!enzyme.Gradient<f32>) -> f32
// CHECK-NEXT:      %[[D_ACC:.+]] = arith.addf %[[D0]], %[[NEW_D0]] : f32
// CHECK-NEXT:      "enzyme.set"(%[[G]], %[[D_ACC]]) : (!enzyme.Gradient<f32>, f32) -> ()
// CHECK-NEXT:      affine.yield %[[NEW_DIT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[G0:.+]] = "enzyme.get"(%[[G]]) : (!enzyme.Gradient<f32>) -> f32
// CHECK-NEXT:    return %[[G0]] : f32
// CHECK-NEXT:  }
