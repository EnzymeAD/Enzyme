// RUN:%eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s

module {
  func.func @main(%init1: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index

    %res:2 = scf.while (%iter = %c0, %arg1 = %init1) : (index, f32) -> (index, f32) {
      // "Before" region.
      // In a "while" loop, this region computes the condition.
      %condition = arith.cmpi eq, %iter, %c10 : index

      // Forward the argument (as result or "after" region argument).
      scf.condition(%condition) %iter, %arg1 : index, f32

    } do {
    ^bb0(%iterAfter: index, %arg2: f32):
      // "After" region.
      // In a "while" loop, this region is the loop body.
      %next = arith.mulf %arg2, %arg2 : f32
      %nextIter = arith.addi %iterAfter,  %c1 : index

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %nextIter, %next : index, f32
    }
    return %res#1 : f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[v0:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f32>
// CHECK-NEXT:    %[[v1:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f32>
// CHECK-NEXT:    %[[v2:.+]]:3 = scf.while (%arg2 = %c0, %arg3 = %arg0, %arg4 = %c0) : (index, f32, index) -> (index, f32, index) {
// CHECK-NEXT:      %[[v4:.+]] = arith.cmpi eq, %arg2, %c10 : index
// CHECK-NEXT:      %[[v5:.+]] = arith.addi %arg4, %c1 : index
// CHECK-NEXT:      scf.condition(%[[v4]]) %arg2, %arg3, %[[v5]] : index, f32, index
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%arg2: index, %arg3: f32, %arg4: index):
// CHECK-NEXT:      "enzyme.push"(%[[v1]], %arg3) : (!enzyme.Cache<f32>, f32) -> ()
// CHECK-NEXT:      "enzyme.push"(%[[v0]], %arg3) : (!enzyme.Cache<f32>, f32) -> ()
// CHECK-NEXT:      %[[v4:.+]] = arith.mulf %arg3, %arg3 : f32
// CHECK-NEXT:      %[[v5:.+]] = arith.addi %arg2, %c1 : index
// CHECK-NEXT:      scf.yield %[[v5]], %[[v4]], %arg4 : index, f32, index
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v3:.+]] = scf.for %arg2 = %c0 to %[[v2]]#2 step %c1 iter_args(%arg3 = %arg1) -> (f32) {
// CHECK-NEXT:      %[[v4:.+]] = arith.cmpi eq, %arg2, %c0 : index
// CHECK-NEXT:      %[[v5:.+]] = scf.if %[[v4]] -> (f32) {
// CHECK-NEXT:        scf.yield %arg3 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[v6:.+]] = "enzyme.pop"(%[[v1]]) : (!enzyme.Cache<f32>) -> f32
// CHECK-NEXT:        %[[v7:.+]] = "enzyme.pop"(%[[v0]]) : (!enzyme.Cache<f32>) -> f32
// CHECK-NEXT:        %[[v8:.+]] = arith.mulf %arg3, %[[v7]] : f32
// CHECK-NEXT:        %[[v9:.+]] = arith.mulf %arg3, %[[v6]] : f32
// CHECK-NEXT:        %[[v10:.+]] = arith.addf %[[v8]], %[[v9]] : f32
// CHECK-NEXT:        scf.yield %[[v10]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[v5]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[v3]] : f32
// CHECK-NEXT:  }
