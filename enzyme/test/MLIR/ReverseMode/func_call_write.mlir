// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s
// The callee writes through its dup memref argument, which reverse mode
// cannot yet differentiate: the reverse pass would need the argument's
// pre-call contents, and caching of overwritten arguments (LLVM Enzyme's
// overwritten_args) is not implemented on the MLIR side. Tracked in
// https://github.com/EnzymeAD/Enzyme/issues/3109 -- remove the XFAIL when
// it lands.
// XFAIL: *

module {
  func.func @main(%a: memref<f32>, %b: f32) -> f32 {
    %0 = func.call @f(%a, %b) : (memref<f32>, f32) -> f32
    return %0 : f32
  }

  func.func @f(%a: memref<f32>, %b: f32) -> f32 {
    memref.store %b, %a[] : memref<f32>
    %0 = memref.load %a[] : memref<f32>
    return %0 : f32
  }

}

// CHECK:  func.func @main(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: f32, %arg3: f32) -> f32 {
// CHECK-NEXT:    %alloc = memref.alloc() : memref<f32>
// CHECK-NEXT:    memref.copy %arg0, %alloc : memref<f32> to memref<f32>
// CHECK-NEXT:    %0 = call @f(%arg0, %arg2) : (memref<f32>, f32) -> f32
// CHECK-NEXT:    %1 = call @diffef(%alloc, %arg1, %arg2, %arg3) : (memref<f32>, memref<f32>, f32, f32) -> f32
// CHECK-NEXT:    memref.dealloc %alloc : memref<f32>
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffef(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: f32, %arg3: f32) -> f32 {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    memref.store %arg2, %arg0[] : memref<f32>
// CHECK-NEXT:    %0 = memref.load %arg1[] : memref<f32>
// CHECK-NEXT:    %1 = arith.addf %0, %arg3 fastmath<fast> : f32
// CHECK-NEXT:    memref.store %1, %arg1[] : memref<f32>
// CHECK-NEXT:    %2 = memref.load %arg1[] : memref<f32>
// CHECK-NEXT:    memref.store %[[zero]], %arg1[] : memref<f32>
// CHECK-NEXT:    return %2 : f32
// CHECK-NEXT:  }
