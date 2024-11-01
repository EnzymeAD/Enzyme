// RUN: %eopt --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

module {
    func.func @square(%arg0: f32) -> f32 {
        %0 = arith.mulf %arg0, %arg0 : f32
        return %0 : f32
    }

    func.func @inactive(%arg0: f32, %arg1: i32) -> (f32, i32) {
        %0 = arith.constant 42.0 : f32
        %1 = arith.constant 42 : i32
        return %0, %1 : f32, i32
    }

    func.func @main(%arg0: f32) -> f32 {
        %0 = func.call @square(%arg0) : (f32) -> f32
        %1 = arith.constant 10 : i32
        %2:2 = func.call @inactive(%arg0, %1) : (f32, i32) -> (f32, i32)
        %3 = arith.addf %0, %2#0 : f32
        return %3 : f32
    }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:    %0 = call @square(%arg0) : (f32) -> f32
// CHECK-NEXT:    %1:2 = call @inactive(%arg0, %c10_i32) : (f32, i32) -> (f32, i32)
// CHECK-NEXT:    %2 = call @diffeinactive(%arg0, %c10_i32, %arg1) : (f32, i32, f32) -> f32
// CHECK-NEXT:    %3 = call @diffesquare(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT:    %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:    return %4 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffeinactive(%arg0: f32, %arg1: i32, %arg2: f32) -> f32 {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    return %cst : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffesquare(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = arith.mulf %arg1, %arg0 : f32
// CHECK-NEXT:    %1 = arith.mulf %arg1, %arg0 : f32
// CHECK-NEXT:    %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:    return %2 : f32
// CHECK-NEXT:  }
