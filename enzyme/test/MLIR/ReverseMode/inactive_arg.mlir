// RUN: %eopt %s --convert-scf-to-cf --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {
  func.func @main(%arg0: f32) -> f32 {
    %cst = arith.constant 0.0 : f32
    %0 = arith.cmpf ogt, %arg0, %cst : f32
    %1 = scf.if %0 -> f32 {
      scf.yield %arg0 : f32
    } else {
      scf.yield %cst : f32
    }
    return %1 : f32
  }
}

//      CHECK:  func.func @main(%[[X:.+]]: f32, %[[DIFFE:.+]]: f32) -> f32 {
// CHECK-NEXT:    %[[CST0:.+]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[CST1:.+]] = arith.constant -1 : i32
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[CACHE:.+]] = "enzyme.init"() : () -> !enzyme.Cache<i32>
//      CHECK:    %[[COND:.+]] = arith.cmpf ogt, %[[X]], %[[ZERO]] : f32
//      CHECK:    cf.cond_br %[[COND]], ^bb1(%[[CST0]] : i32), ^bb1(%[[CST1]] : i32)
//      CHECK:  ^bb1(%[[BLOCKARG:.+]]: i32): // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:    "enzyme.push"(%[[CACHE]], %[[BLOCKARG]]) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.br ^bb2
//      CHECK:  ^bb2: // pred: ^bb1
//      CHECK:    %[[INCOMING:.+]] = "enzyme.pop"(%[[CACHE]]) : (!enzyme.Cache<i32>) -> i32
// CHECK-NEXT:    %[[COND2:.+]] = arith.cmpi eq, %[[INCOMING]], %[[CST1]] : i32
// CHECK-NEXT:    %[[GRAD:.+]] = arith.select %[[COND2]], %[[DIFFE]], %[[ZERO]] : f32
//      CHECK:    return %[[GRAD]] : f32
// CHECK-NEXT:  }
