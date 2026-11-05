// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s

func.func @both(%x : f64) -> (f64, f64) {
  %s, %c = math.sincos %x : f64
  return %s, %c : f64, f64
}

func.func @dboth(%x : f64, %ds : f64, %dc : f64) -> f64 {
  %r = enzyme.autodiff @both(%x, %ds, %dc) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_activenoneed>] }> : (f64, f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @diffeboth
// CHECK-SAME:      (%[[X:.+]]: f64, %[[DS:.+]]: f64, %[[DC:.+]]: f64) -> f64
// CHECK-NEXT:    %[[COS:.+]] = math.cos %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[A:.+]] = arith.mulf %[[DS]], %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    %[[SIN:.+]] = math.sin %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[NSIN:.+]] = arith.negf %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[B:.+]] = arith.mulf %[[DC]], %[[NSIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DX:.+]] = arith.addf %[[A]], %[[B]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[DX]] : f64

// -----

func.func @onlysin(%x : f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %s : f64
}

func.func @donlysin(%x : f64, %ds : f64) -> f64 {
  %r = enzyme.autodiff @onlysin(%x, %ds) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @diffeonlysin
// CHECK-SAME:      (%[[X:.+]]: f64, %[[DS:.+]]: f64) -> f64
// CHECK-NEXT:    %[[COS:.+]] = math.cos %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DX:.+]] = arith.mulf %[[DS]], %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[DX]] : f64

// -----

func.func @onlycos(%x : f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %c : f64
}

func.func @donlycos(%x : f64, %dc : f64) -> f64 {
  %r = enzyme.autodiff @onlycos(%x, %dc) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @diffeonlycos
// CHECK-SAME:      (%[[X:.+]]: f64, %[[DC:.+]]: f64) -> f64
// CHECK-NEXT:    %[[SIN:.+]] = math.sin %[[X]] fastmath<fast> : f64
// CHECK-NEXT:    %[[NSIN:.+]] = arith.negf %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DX:.+]] = arith.mulf %[[DC]], %[[NSIN]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[DX]] : f64
