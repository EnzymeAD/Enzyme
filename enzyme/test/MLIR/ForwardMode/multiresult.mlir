// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --cse %s | FileCheck %s

func.func @both(%x : f64) -> (f64, f64) {
  %s, %c = math.sincos %x : f64
  return %s, %c : f64, f64
}

func.func @dboth(%x : f64, %dx : f64) -> (f64, f64) {
  %r:2 = enzyme.fwddiff @both(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> (f64, f64)
  return %r#0, %r#1 : f64, f64
}

// CHECK-LABEL: func.func private @fwddiffeboth
// CHECK-SAME:      (%[[X:.+]]: f64, %[[DX:.+]]: f64) -> (f64, f64)
// CHECK-NEXT:    %[[SIN:.+]], %[[COS:.+]] = math.sincos %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DSIN:.+]] = arith.mulf %[[DX]], %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    %[[M:.+]] = arith.mulf %[[DX]], %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DCOS:.+]] = arith.negf %[[M]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[DSIN]], %[[DCOS]] : f64, f64

// -----

func.func @onlysin(%x : f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %s : f64
}

func.func @donlysin(%x : f64, %dx : f64) -> f64 {
  %r = enzyme.fwddiff @onlysin(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @fwddiffeonlysin
// CHECK-SAME:      (%[[X:.+]]: f64, %[[DX:.+]]: f64) -> f64
// CHECK-NEXT:    %[[COS:.+]] = math.cos %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DSIN:.+]] = arith.mulf %[[DX]], %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[DSIN]] : f64

// -----

func.func @onlycos(%x : f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %c : f64
}

func.func @donlycos(%x : f64, %dx : f64) -> f64 {
  %r = enzyme.fwddiff @onlycos(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @fwddiffeonlycos
// CHECK-SAME:      (%[[X:.+]]: f64, %[[DX:.+]]: f64) -> f64
// CHECK-NEXT:    %[[SIN:.+]] = math.sin %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[M:.+]] = arith.mulf %[[DX]], %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DCOS:.+]] = arith.negf %[[M]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[DCOS]] : f64
