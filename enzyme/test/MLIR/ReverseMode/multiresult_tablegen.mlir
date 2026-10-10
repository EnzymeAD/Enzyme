// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --cse %s | FileCheck %s

func.func @both(%x: f64) -> f64 {
  %s, %c = math.sincos %x : f64
  %r = arith.addf %s, %c : f64
  return %r : f64
}

func.func @dboth(%x: f64, %dr: f64) -> f64 {
  %0 = enzyme.autodiff @both(%x, %dr)
    {
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64) -> f64
  return %0 : f64
}

// x̄ = dr·cos(x) − dr·sin(x)
// CHECK-LABEL: func.func private @diffeboth(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %[[SIN:.+]], %[[COS:.+]] = math.sincos %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %[[A:.+]] = arith.mulf %arg1, %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    %[[B:.+]] = arith.mulf %arg1, %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[R:.+]] = arith.subf %[[A]], %[[B]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[R]] : f64
// CHECK-NEXT:  }

// -----

func.func @onlysin(%x: f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %s : f64
}

func.func @donlysin(%x: f64, %dr: f64) -> f64 {
  %0 = enzyme.autodiff @onlysin(%x, %dr)
    {
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64) -> f64
  return %0 : f64
}

// The cosine is inactive, so only the sine's term remains: x̄ = dr·cos(x)
// CHECK-LABEL: func.func private @diffeonlysin(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %[[COS:.+]] = math.cos %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %[[R:.+]] = arith.mulf %arg1, %[[COS]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[R]] : f64
// CHECK-NEXT:  }

// -----

func.func @onlycos(%x: f64) -> f64 {
  %s, %c = math.sincos %x : f64
  return %c : f64
}

func.func @donlycos(%x: f64, %dr: f64) -> f64 {
  %0 = enzyme.autodiff @onlycos(%x, %dr)
    {
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_activenoneed>]
    } : (f64, f64) -> f64
  return %0 : f64
}

// Result 0 (the sine) is inactive and result 1 is active: x̄ = −dr·sin(x)
// CHECK-LABEL: func.func private @diffeonlycos(%arg0: f64, %arg1: f64) -> f64 {
// CHECK-NEXT:    %[[SIN:.+]] = math.sin %arg0 fastmath<fast> : f64
// CHECK-NEXT:    %[[M:.+]] = arith.mulf %arg1, %[[SIN]] fastmath<fast> : f64
// CHECK-NEXT:    %[[R:.+]] = arith.negf %[[M]] fastmath<fast> : f64
// CHECK-NEXT:    return %[[R]] : f64
// CHECK-NEXT:  }
