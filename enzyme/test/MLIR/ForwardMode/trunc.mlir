// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @f(%x : f64) -> f32 {
    %y = arith.truncf %x : f64 to f32
    return %y : f32
  }
  func.func @dsq(%x : f64, %dx : f64) -> f32 {
    %r = enzyme.fwddiff @f(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> (f32)
    return %r : f32
  }
}

// CHECK:   func.func private @fwddiffef(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f32 {
// CHECK-NEXT:     %[[dy:.+]] = arith.truncf %[[arg1]] : f64 to f32
// CHECK-NEXT:     %[[y:.+]] = arith.truncf %[[arg0]] : f64 to f32
// CHECK-NEXT:     return %[[dy]] : f32
// CHECK-NEXT:   }
