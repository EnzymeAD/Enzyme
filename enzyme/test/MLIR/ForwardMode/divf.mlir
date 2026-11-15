// RUN: %eopt --split-input-file --enzyme %s | FileCheck %s
// RUN: %eopt --split-input-file --enzyme --inline --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s --check-prefix=EVAL

func.func @div(%x: f64, %y: f64) -> f64 {
  %res = arith.divf %x, %y : f64
  return %res : f64
}

func.func @ddiv(%x: f64, %dx: f64, %y: f64, %dy: f64) -> f64 {
  %0 = enzyme.fwddiff @div(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64, f64, f64) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func private @fwddiffediv(
// CHECK-SAME: %[[x:.+]]: f64, %[[dx:.+]]: f64, %[[y:.+]]: f64, %[[dy:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[quot:.+]] = arith.divf %[[x]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[dyq:.+]] = arith.mulf %[[dy]], %[[quot]] fastmath<fast> : f64
// CHECK-NEXT:    %[[num:.+]] = arith.subf %[[dx]], %[[dyq]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[num]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = arith.divf %[[x]], %[[y]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @tensor_div(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
  %res = arith.divf %x, %y : tensor<2xf32>
  return %res : tensor<2xf32>
}

func.func @dtensor_div(%x: tensor<2xf32>, %dx: tensor<2xf32>, %y: tensor<2xf32>, %dy: tensor<2xf32>) -> tensor<2xf32> {
  %0 = enzyme.fwddiff @tensor_div(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func private @fwddiffetensor_div(
// CHECK-SAME: %[[tx:.+]]: tensor<2xf32>, %[[tdx:.+]]: tensor<2xf32>, %[[ty:.+]]: tensor<2xf32>, %[[tdy:.+]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK-NEXT:    %[[tquot:.+]] = arith.divf %[[tx]], %[[ty]] fastmath<fast> : tensor<2xf32>
// CHECK-NEXT:    %[[tdyq:.+]] = arith.mulf %[[tdy]], %[[tquot]] fastmath<fast> : tensor<2xf32>
// CHECK-NEXT:    %[[tnum:.+]] = arith.subf %[[tdx]], %[[tdyq]] fastmath<fast> : tensor<2xf32>
// CHECK-NEXT:    %[[tres:.+]] = arith.divf %[[tnum]], %[[ty]] fastmath<fast> : tensor<2xf32>
// CHECK-NEXT:    %{{.+}} = arith.divf %[[tx]], %[[ty]] : tensor<2xf32>
// CHECK-NEXT:    return %[[tres]] : tensor<2xf32>
// CHECK-NEXT:  }

// -----

func.func @div_const_y(%x: f64, %y: f64) -> f64 {
  %res = arith.divf %x, %y : f64
  return %res : f64
}

func.func @ddiv_const_y(%x: f64, %dx: f64, %y: f64) -> f64 {
  %0 = enzyme.fwddiff @div_const_y(%x, %dx, %y) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64, f64) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func private @fwddiffediv_const_y(
// CHECK-SAME: %[[x:.+]]: f64, %[[dx:.+]]: f64, %[[y:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[num:.+]] = arith.subf %[[dx]], %[[zero]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[num]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = arith.divf %[[x]], %[[y]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @div_const_x(%x: f64, %y: f64) -> f64 {
  %res = arith.divf %x, %y : f64
  return %res : f64
}

func.func @ddiv_const_x(%x: f64, %y: f64, %dy: f64) -> f64 {
  %0 = enzyme.fwddiff @div_const_x(%x, %y, %dy) <{ activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64, f64) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func private @fwddiffediv_const_x(
// CHECK-SAME: %[[x:.+]]: f64, %[[y:.+]]: f64, %[[dy:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[quot:.+]] = arith.divf %[[x]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[dyq:.+]] = arith.mulf %[[dy]], %[[quot]] fastmath<fast> : f64
// CHECK-NEXT:    %[[num:.+]] = arith.subf %[[zero]], %[[dyq]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[num]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %{{.+}} = arith.divf %[[x]], %[[y]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @div_strong_zero(%x: f64, %y: f64) -> f64 {
  %res = arith.divf %x, %y : f64
  return %res : f64
}

func.func @ddiv_strong_zero(%x: f64, %dx: f64, %y: f64, %dy: f64) -> f64 {
  %0 = enzyme.fwddiff @div_strong_zero(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], strong_zero=true }> : (f64, f64, f64, f64) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func private @fwddiffediv_strong_zero(
// CHECK-SAME: %[[x:.+]]: f64, %[[dx:.+]]: f64, %[[y:.+]]: f64, %[[dy:.+]]: f64) -> f64 {
// CHECK-NEXT:    %[[quot:.+]] = arith.divf %[[x]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[zero0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[cmp0:.+]] = arith.cmpf oeq, %[[dy]], %[[zero0]] fastmath<fast> : f64
// CHECK-NEXT:    %[[mul:.+]] = arith.mulf %[[dy]], %[[quot]] fastmath<fast> : f64
// CHECK-NEXT:    %[[dyq:.+]] = arith.select %[[cmp0]], %[[zero0]], %[[mul]] : f64
// CHECK-NEXT:    %[[num:.+]] = arith.subf %[[dx]], %[[dyq]] fastmath<fast> : f64
// CHECK-NEXT:    %[[zero1:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[cmp1:.+]] = arith.cmpf oeq, %[[num]], %[[zero1]] fastmath<fast> : f64
// CHECK-NEXT:    %[[div:.+]] = arith.divf %[[num]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[res:.+]] = arith.select %[[cmp1]], %[[zero1]], %[[div]] : f64
// CHECK-NEXT:    %{{.+}} = arith.divf %[[x]], %[[y]] : f64
// CHECK-NEXT:    return %[[res]] : f64
// CHECK-NEXT:  }

// -----

func.func @div_batched(%x: f64, %y: f64) -> f64 {
  %res = arith.divf %x, %y : f64
  return %res : f64
}

func.func @ddiv_batched(%x: f64, %dx: tensor<2xf64>, %y: f64, %dy: tensor<2xf64>) -> tensor<2xf64> {
  %0 = enzyme.fwddiff @div_batched(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], width=2 }> : (f64, tensor<2xf64>, f64, tensor<2xf64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func.func private @fwddiffe2div_batched(
// CHECK-SAME: %[[x:.+]]: f64, %[[dx:.+]]: tensor<2xf64>, %[[y:.+]]: f64, %[[dy:.+]]: tensor<2xf64>) -> tensor<2xf64> {
// CHECK-NEXT:    %[[quot:.+]] = arith.divf %[[x]], %[[y]] fastmath<fast> : f64
// CHECK-NEXT:    %[[bquot:.+]] = "enzyme.broadcast"(%[[quot]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %[[dyq:.+]] = arith.mulf %[[dy]], %[[bquot]] fastmath<fast> : tensor<2xf64>
// CHECK-NEXT:    %[[num:.+]] = arith.subf %[[dx]], %[[dyq]] fastmath<fast> : tensor<2xf64>
// CHECK-NEXT:    %[[by:.+]] = "enzyme.broadcast"(%[[y]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[num]], %[[by]] fastmath<fast> : tensor<2xf64>
// CHECK-NEXT:    %{{.+}} = arith.divf %[[x]], %[[y]] : f64
// CHECK-NEXT:    return %[[res]] : tensor<2xf64>
// CHECK-NEXT:  }

// -----

func.func @div_batched_const_y(%x: f64, %y: f64) -> f64 {
  %res = arith.divf %x, %y : f64
  return %res : f64
}

func.func @ddiv_batched_const_y(%x: f64, %dx: tensor<2xf64>, %y: f64) -> tensor<2xf64> {
  %0 = enzyme.fwddiff @div_batched_const_y(%x, %dx, %y) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], width=2 }> : (f64, tensor<2xf64>, f64) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func.func private @fwddiffe2div_batched_const_y(
// CHECK-SAME: %[[x:.+]]: f64, %[[dx:.+]]: tensor<2xf64>, %[[y:.+]]: f64) -> tensor<2xf64> {
// CHECK-NEXT:    %[[zero:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %[[bzero:.+]] = "enzyme.broadcast"(%[[zero]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %[[num:.+]] = arith.subf %[[dx]], %[[bzero]] fastmath<fast> : tensor<2xf64>
// CHECK-NEXT:    %[[by:.+]] = "enzyme.broadcast"(%[[y]]) <{shape = array<i64: 2>}> : (f64) -> tensor<2xf64>
// CHECK-NEXT:    %[[res:.+]] = arith.divf %[[num]], %[[by]] fastmath<fast> : tensor<2xf64>
// CHECK-NEXT:    %{{.+}} = arith.divf %[[x]], %[[y]] : f64
// CHECK-NEXT:    return %[[res]] : tensor<2xf64>
// CHECK-NEXT:  }

// -----

// Verify no intermediate overflow/underflow when y^2 overflows or underflows f32,
// verify d/dy (1 / (1 + y^2)) when (1 + y^2)^2 overflows f32,
// and verify strong_zero guards dy * (x / y) when x / y overflows to +inf.

func.func private @ratio_f32(%x: f32, %y: f32) -> f32 {
  %q = arith.divf %x, %y : f32
  return %q : f32
}

func.func private @atan2_deriv(%y: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %yy = arith.mulf %y, %y : f32
  %den = arith.addf %one, %yy : f32
  %res = arith.divf %one, %den : f32
  return %res : f32
}

// EVAL-LABEL: func.func @eval_overflow_divisor() -> f32
// EVAL-NEXT:    %[[HALF:.+]] = arith.constant 5.000000e-01 : f32
// EVAL-NEXT:    return %[[HALF]] : f32
func.func @eval_overflow_divisor() -> f32 {
  %x = arith.constant 2.0e+20 : f32
  %dx = arith.constant 2.0e+20 : f32
  %y = arith.constant 2.0e+20 : f32
  %dy = arith.constant 1.0e+20 : f32
  %res = enzyme.fwddiff @ratio_f32(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f32, f32, f32, f32) -> f32
  return %res : f32
}

// EVAL-LABEL: func.func @eval_underflow_divisor() -> f32
// EVAL-NEXT:    %[[HALF:.+]] = arith.constant 5.000000e-01 : f32
// EVAL-NEXT:    return %[[HALF]] : f32
func.func @eval_underflow_divisor() -> f32 {
  %x = arith.constant 2.0e-23 : f32
  %dx = arith.constant 2.0e-23 : f32
  %y = arith.constant 2.0e-23 : f32
  %dy = arith.constant 1.0e-23 : f32
  %res = enzyme.fwddiff @ratio_f32(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f32, f32, f32, f32) -> f32
  return %res : f32
}

// At y = 2^34 = 17179869184.0f, 1 + y^2 = 2^68 is finite in f32 while (1 + y^2)^2
// overflows f32 to +inf; d/dy (1 / (1 + y^2)) = -2*y / (1 + y^2)^2 = -2^-101.
// EVAL-LABEL: func.func @eval_atan2_hessian() -> f32
// EVAL-NEXT:    %[[HESS:.+]] = arith.constant -3.94430453E-31 : f32
// EVAL-NEXT:    return %[[HESS]] : f32
func.func @eval_atan2_hessian() -> f32 {
  %y = arith.constant 17179869184.0 : f32
  %dy = arith.constant 1.0 : f32
  %res = enzyme.fwddiff @atan2_deriv(%y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f32, f32) -> f32
  return %res : f32
}

// EVAL-LABEL: func.func @eval_strong_zero_inf_quotient() -> f32
// EVAL-NEXT:    %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// EVAL-NEXT:    return %[[ONE]] : f32
func.func @eval_strong_zero_inf_quotient() -> f32 {
  %x = arith.constant 1.0e+30 : f32
  %dx = arith.constant 1.0e-30 : f32
  %y = arith.constant 1.0e-30 : f32
  %dy = arith.constant 0.0 : f32
  %res = enzyme.fwddiff @ratio_f32(%x, %dx, %y, %dy) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>], strong_zero=true }> : (f32, f32, f32, f32) -> f32
  return %res : f32
}
