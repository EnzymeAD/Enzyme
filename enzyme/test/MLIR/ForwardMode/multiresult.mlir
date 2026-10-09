// RUN: %eopt --enzyme --cse %s | FileCheck %s

module {
  func.func @both(%x : f64) -> (f64, f64) {
    %s, %c = math.sincos %x : f64
    return %s, %c : f64, f64
  }
  func.func @dboth(%x : f64, %dx : f64) -> (f64, f64) {
    %r:2 = enzyme.fwddiff @both(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> (f64, f64)
    return %r#0, %r#1 : f64, f64
  }

  func.func @onlysin(%x : f64) -> f64 {
    %s, %c = math.sincos %x : f64
    return %s : f64
  }
  func.func @donlysin(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @onlysin(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }

  func.func @onlycos(%x : f64) -> f64 {
    %s, %c = math.sincos %x : f64
    return %c : f64
  }
  func.func @donlycos(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @onlycos(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// CHECK:   func.func private @fwddiffeboth(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> (f64, f64) {
// CHECK-NEXT:     %[[sin:.+]], %[[cos:.+]] = math.sincos %[[arg0]] fastmath<fast> : f64
// CHECK-NEXT:     %[[dsin:.+]] = arith.mulf %[[arg1]], %[[cos]] fastmath<fast> : f64
// CHECK-NEXT:     %[[m:.+]] = arith.mulf %[[arg1]], %[[sin]] fastmath<fast> : f64
// CHECK-NEXT:     %[[dcos:.+]] = arith.negf %[[m]] fastmath<fast> : f64
// CHECK-NEXT:     return %[[dsin]], %[[dcos]] : f64, f64
// CHECK-NEXT:   }

// CHECK:   func.func private @fwddiffeonlysin(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-NEXT:     %[[cos:.+]] = math.cos %[[arg0]] fastmath<fast> : f64
// CHECK-NEXT:     %[[dsin:.+]] = arith.mulf %[[arg1]], %[[cos]] fastmath<fast> : f64
// CHECK-NEXT:     return %[[dsin]] : f64
// CHECK-NEXT:   }

// CHECK:   func.func private @fwddiffeonlycos(%[[arg0:.+]]: f64, %[[arg1:.+]]: f64) -> f64 {
// CHECK-NEXT:     %[[sin:.+]] = math.sin %[[arg0]] fastmath<fast> : f64
// CHECK-NEXT:     %[[m:.+]] = arith.mulf %[[arg1]], %[[sin]] fastmath<fast> : f64
// CHECK-NEXT:     %[[dcos:.+]] = arith.negf %[[m]] fastmath<fast> : f64
// CHECK-NEXT:     return %[[dcos]] : f64
// CHECK-NEXT:   }
