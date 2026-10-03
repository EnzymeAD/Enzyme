// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --cse %s | FileCheck %s

func.func @both(%x: f64) -> f64 {
  %s, %c = math.sincos %x : f64
  %r = arith.addf %s, %c : f64
  return %r : f64
}

func.func @dboth(%x: f64, %dx: f64) -> f64 {
  %r = enzyme.fwddiff @both(%x, %dx)
    {
      activity=[#enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
    } : (f64, f64) -> f64
  return %r : f64
}

// CHECK-LABEL: func.func private @fwddiffeboth(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %[[SIN:.+]], %[[COS:.+]] = math.sincos %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %[[DS:.+]] = arith.mulf %arg1, %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    %[[M:.+]] = arith.mulf %arg1, %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[DC:.+]] = arith.negf %[[M]] fastmath<fast> : f64
// CHECK-NEXT:    %[[R:.+]] = arith.addf %[[DS]], %[[DC]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[R]] : f64
// CHECK-NEXT:  }

// -----

func.func @onlysin(%x: f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %s : f64
}

func.func @donlysin(%x: f64, %dx: f64) -> f64 {
  %r = enzyme.fwddiff @onlysin(%x, %dx)
    {
      activity=[#enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
    } : (f64, f64) -> f64
  return %r : f64
}

// The cosine is unused, so only ṡ is computed: ṙ = ẋ·cos(x)
// CHECK-LABEL: func.func private @fwddiffeonlysin(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %[[COS:.+]] = math.cos %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %[[R:.+]] = arith.mulf %arg1, %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[R]] : f64
// CHECK-NEXT:  }

// -----

func.func @onlycos(%x: f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %c : f64
}

func.func @donlycos(%x: f64, %dx: f64) -> f64 {
  %r = enzyme.fwddiff @onlycos(%x, %dx)
    {
      activity=[#enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
    } : (f64, f64) -> f64
  return %r : f64
}

// Result 0 (the sine) is unused and result 1 is: ṙ = −ẋ·sin(x)
// CHECK-LABEL: func.func private @fwddiffeonlycos(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %[[SIN:.+]] = math.sin %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %[[M:.+]] = arith.mulf %arg1, %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[R:.+]] = arith.negf %[[M]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[R]] : f64
// CHECK-NEXT:  }
