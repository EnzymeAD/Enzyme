// RUN: %eopt --enzyme-wrap="infn=square outfn=dsq retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
}

// CHECK:   func.func private @dsq(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:     %[[i0:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i1:.+]] = arith.mulf %[[arg1]], %[[arg0]] : f64
// CHECK-NEXT:     %[[i2:.+]] = arith.addf %[[i0]], %[[i1]] : f64
// CHECK-NEXT:     %[[i3:.+]] = arith.mulf %[[arg0]], %[[arg0]] : f64
// CHECK-NEXT:     return %[[i3]], %[[i2]] : f64, f64
// CHECK-NEXT:   }
