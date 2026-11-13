// RUN: %eopt --enzyme-batch %s | FileCheck %s

module {
  // 1. Mixed batched + unbatched tensor arguments:
  // pure computations on the unbatched argument stay unbatched, with a single
  // enzyme.broadcast inserted where the unbatched chain meets the batched value.
  func.func @mixed_tensor(%x: tensor<3xf64>, %u: tensor<3xf64>) -> tensor<3xf64> {
    %u1 = math.sin %u : tensor<3xf64>
    %u2 = arith.mulf %u1, %u : tensor<3xf64>
    %y = arith.addf %x, %u2 : tensor<3xf64>
    %z = arith.mulf %y, %u2 : tensor<3xf64>
    return %z : tensor<3xf64>
  }

  func.func @call_mixed_tensor(%x: tensor<4x3xf64>, %u: tensor<3xf64>) -> tensor<4x3xf64> {
    %r = enzyme.batch @mixed_tensor(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<4x3xf64>
    return %r : tensor<4x3xf64>
  }

  // CHECK-LABEL: func.func @call_mixed_tensor
  // CHECK-SAME: (%[[X:.+]]: tensor<4x3xf64>, %[[U:.+]]: tensor<3xf64>) -> tensor<4x3xf64>
  // CHECK-NEXT: %[[R:.+]] = call @batched_mixed_tensor(%[[X]], %[[U]]) : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<4x3xf64>
  // CHECK-NEXT: return %[[R]] : tensor<4x3xf64>

  // 2. Mixed batched + unbatched scalar arguments and shared scalar constant.
  func.func @mixed_scalar(%x: f64, %u: f64) -> f64 {
    %cst = arith.constant 2.0 : f64
    %u1 = arith.mulf %u, %cst : f64
    %u2 = math.exp %u1 : f64
    %x1 = arith.mulf %x, %cst : f64
    %y = arith.addf %x1, %u2 : f64
    return %y : f64
  }

  func.func @call_mixed_scalar(%x: tensor<4xf64>, %u: f64) -> tensor<4xf64> {
    %r = enzyme.batch @mixed_scalar(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4xf64>, f64) -> tensor<4xf64>
    return %r : tensor<4xf64>
  }

  // CHECK-LABEL: func.func @call_mixed_scalar
  // CHECK-SAME: (%[[SX:.+]]: tensor<4xf64>, %[[SU:.+]]: f64) -> tensor<4xf64>
  // CHECK-NEXT: %[[SR:.+]] = call @batched_mixed_scalar(%[[SX]], %[[SU]]) : (tensor<4xf64>, f64) -> tensor<4xf64>
  // CHECK-NEXT: return %[[SR]] : tensor<4xf64>

  // 3. Mixed batched and unbatched returns, plus requesting batched output on
  // an unbatched return value.
  func.func @mixed_returns(%x: tensor<3xf64>, %u: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
    %u2 = arith.mulf %u, %u : tensor<3xf64>
    %y = arith.addf %x, %u2 : tensor<3xf64>
    return %y, %u2 : tensor<3xf64>, tensor<3xf64>
  }

  func.func @call_mixed_returns_unbatched(%x: tensor<4x3xf64>, %u: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %r:2 = enzyme.batch @mixed_returns(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    return %r#0, %r#1 : tensor<4x3xf64>, tensor<3xf64>
  }

  func.func @call_mixed_returns_batched(%x: tensor<4x3xf64>, %u: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<4x3xf64>) {
    %r:2 = enzyme.batch @mixed_returns(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<4x3xf64>)
    return %r#0, %r#1 : tensor<4x3xf64>, tensor<4x3xf64>
  }

  // CHECK-LABEL: func.func @call_mixed_returns_unbatched
  // CHECK-NEXT: %[[MR1:.+]]:2 = call @batched_mixed_returns(%{{.+}}, %{{.+}}) : (tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
  // CHECK-NEXT: return %[[MR1]]#0, %[[MR1]]#1 : tensor<4x3xf64>, tensor<3xf64>

  // CHECK-LABEL: func.func @call_mixed_returns_batched
  // CHECK-NEXT: %[[MR2:.+]]:2 = call @batched_mixed_returns_0(%{{.+}}, %{{.+}}) : (tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<4x3xf64>)
  // CHECK-NEXT: return %[[MR2]]#0, %[[MR2]]#1 : tensor<4x3xf64>, tensor<4x3xf64>

  // 4. Non-splat dense constant broadcasts cleanly instead of failing resizeSplat.
  func.func @non_splat_const(%x: tensor<3xf64>) -> tensor<3xf64> {
    %cst = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
    %y = arith.addf %x, %cst : tensor<3xf64>
    return %y : tensor<3xf64>
  }

  func.func @call_non_splat_const(%x: tensor<4x3xf64>) -> tensor<4x3xf64> {
    %r = enzyme.batch @non_splat_const(%x) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>) -> tensor<4x3xf64>
    return %r : tensor<4x3xf64>
  }

  // 5. Helper calls: all-unbatched helper stays a direct call, mixed helper is
  // specialized with mixed batched/unbatched args and returns.
  func.func private @unbatched_helper(%u: tensor<3xf64>) -> tensor<3xf64> {
    %s = math.sin %u : tensor<3xf64>
    return %s : tensor<3xf64>
  }

  func.func private @mixed_helper(%x: tensor<3xf64>, %u: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
    %u2 = arith.mulf %u, %u : tensor<3xf64>
    %x2 = arith.addf %x, %u2 : tensor<3xf64>
    return %x2, %u2 : tensor<3xf64>, tensor<3xf64>
  }

  func.func @caller_with_helpers(%x: tensor<3xf64>, %u: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
    %u1 = func.call @unbatched_helper(%u) : (tensor<3xf64>) -> tensor<3xf64>
    %m:2 = func.call @mixed_helper(%x, %u1) : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    return %m#0, %m#1 : tensor<3xf64>, tensor<3xf64>
  }

  func.func @call_helpers(%x: tensor<4x3xf64>, %u: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %r:2 = enzyme.batch @caller_with_helpers(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    return %r#0, %r#1 : tensor<4x3xf64>, tensor<3xf64>
  }

  // 6. Caching distinguishes different argument batch masks and reuses matching ones.
  func.func @add2(%a: tensor<3xf64>, %b: tensor<3xf64>) -> tensor<3xf64> {
    %s = arith.addf %a, %b : tensor<3xf64>
    return %s : tensor<3xf64>
  }

  func.func @call_add2_masks(%x: tensor<4x3xf64>, %u: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<4x3xf64>, tensor<4x3xf64>) {
    %r0 = enzyme.batch @add2(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<4x3xf64>
    %r1 = enzyme.batch @add2(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<4x3xf64>
    %r2 = enzyme.batch @add2(%u, %x) <{batch_shape = array<i64: 4>}> : (tensor<3xf64>, tensor<4x3xf64>) -> tensor<4x3xf64>
    return %r0, %r1, %r2 : tensor<4x3xf64>, tensor<4x3xf64>, tensor<4x3xf64>
  }

  // CHECK-LABEL: func.func @call_add2_masks
  // CHECK-NEXT: %[[A0:.+]] = call @batched_add2(%{{.+}}, %{{.+}}) : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<4x3xf64>
  // CHECK-NEXT: %[[A1:.+]] = call @batched_add2(%{{.+}}, %{{.+}}) : (tensor<4x3xf64>, tensor<3xf64>) -> tensor<4x3xf64>
  // CHECK-NEXT: %[[A2:.+]] = call @batched_add2_0(%{{.+}}, %{{.+}}) : (tensor<3xf64>, tensor<4x3xf64>) -> tensor<4x3xf64>
  // CHECK-NEXT: return %[[A0]], %[[A1]], %[[A2]] : tensor<4x3xf64>, tensor<4x3xf64>, tensor<4x3xf64>

  // 7. Control flow: scf.for with unbatched bounds and mixed iter_args, plus
  // an unbatched init promoted by backedge dependence.
  func.func @scf_for_mixed(%x: tensor<3xf64>, %u: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %r:3 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc_b = %x, %acc_u = %u, %acc_prom = %u) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
      %next_u = arith.mulf %acc_u, %u : tensor<3xf64>
      %next_b = arith.addf %acc_b, %next_u : tensor<3xf64>
      %next_prom = arith.addf %acc_prom, %acc_b : tensor<3xf64>
      scf.yield %next_b, %next_u, %next_prom : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
    }
    return %r#0, %r#1, %r#2 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }

  func.func @call_scf_for_mixed(%x: tensor<4x3xf64>, %u: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>) {
    %r:3 = enzyme.batch @scf_for_mixed(%x, %u) <{batch_shape = array<i64: 4>}> : (tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>)
    return %r#0, %r#1, %r#2 : tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>
  }

  // 8. Control flow: scf.if with unbatched condition and mixed results.
  func.func @scf_if_mixed(%cond: i1, %x: tensor<3xf64>, %u: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
    %r:2 = scf.if %cond -> (tensor<3xf64>, tensor<3xf64>) {
      %a = arith.addf %x, %u : tensor<3xf64>
      scf.yield %a, %u : tensor<3xf64>, tensor<3xf64>
    } else {
      %b = arith.mulf %u, %u : tensor<3xf64>
      scf.yield %u, %b : tensor<3xf64>, tensor<3xf64>
    }
    return %r#0, %r#1 : tensor<3xf64>, tensor<3xf64>
  }

  func.func @call_scf_if_mixed(%cond: i1, %x: tensor<4x3xf64>, %u: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %r:2 = enzyme.batch @scf_if_mixed(%cond, %x, %u) <{batch_shape = array<i64: 4>}> : (i1, tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    return %r#0, %r#1 : tensor<4x3xf64>, tensor<3xf64>
  }

  // 9. All-unbatched inputs with batched output: compute once unbatched and
  // broadcast the return value.
  func.func @all_unbatched(%u: tensor<3xf64>) -> tensor<3xf64> {
    %s = math.sin %u : tensor<3xf64>
    %p = arith.mulf %s, %u : tensor<3xf64>
    return %p : tensor<3xf64>
  }

  func.func @call_all_unbatched(%u: tensor<3xf64>) -> tensor<4x3xf64> {
    %r = enzyme.batch @all_unbatched(%u) <{batch_shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
    return %r : tensor<4x3xf64>
  }

  // 10. CFG control flow with cf.cond_br merging batched and unbatched values.
  func.func @cf_cond_br_mixed(%cond: i1, %x: tensor<3xf64>, %u: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
    %u2 = math.sin %u : tensor<3xf64>
    cf.cond_br %cond, ^bb1(%x, %u2 : tensor<3xf64>, tensor<3xf64>), ^bb1(%u2, %u : tensor<3xf64>, tensor<3xf64>)
  ^bb1(%arg_b: tensor<3xf64>, %arg_u: tensor<3xf64>):
    return %arg_b, %arg_u : tensor<3xf64>, tensor<3xf64>
  }

  func.func @call_cf_cond_br_mixed(%cond: i1, %x: tensor<2x3x3xf64>, %u: tensor<3xf64>) -> (tensor<2x3x3xf64>, tensor<3xf64>) {
    %r:2 = enzyme.batch @cf_cond_br_mixed(%cond, %x, %u) <{batch_shape = array<i64: 2, 3>}> : (i1, tensor<2x3x3xf64>, tensor<3xf64>) -> (tensor<2x3x3xf64>, tensor<3xf64>)
    return %r#0, %r#1 : tensor<2x3x3xf64>, tensor<3xf64>
  }
}

// CHECK-LABEL: func.func private @batched_mixed_tensor
// CHECK-SAME: (%[[BX:.+]]: tensor<4x3xf64>, %[[BU:.+]]: tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[SIN:.+]] = math.sin %[[BU]] : tensor<3xf64>
// CHECK-NEXT: %[[MULU:.+]] = arith.mulf %[[SIN]], %[[BU]] : tensor<3xf64>
// CHECK-NEXT: %[[BCAST:.+]] = "enzyme.broadcast"(%[[MULU]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[BX]], %[[BCAST]] : tensor<4x3xf64>
// CHECK-NEXT: %[[MULB:.+]] = arith.mulf %[[ADD]], %[[BCAST]] : tensor<4x3xf64>
// CHECK-NEXT: return %[[MULB]] : tensor<4x3xf64>

// CHECK-LABEL: func.func private @batched_mixed_scalar
// CHECK-SAME: (%[[BSX:.+]]: tensor<4xf64>, %[[BSU:.+]]: f64) -> tensor<4xf64>
// CHECK-NEXT: %[[CST_U:.+]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT: %[[CST_B:.+]] = arith.constant dense<2.000000e+00> : tensor<4xf64>
// CHECK-NEXT: %[[SU1:.+]] = arith.mulf %[[BSU]], %[[CST_U]] : f64
// CHECK-NEXT: %[[SU2:.+]] = math.exp %[[SU1]] : f64
// CHECK-NEXT: %[[SBCAST:.+]] = "enzyme.broadcast"(%[[SU2]]) <{shape = array<i64: 4>}> : (f64) -> tensor<4xf64>
// CHECK-NEXT: %[[SX1:.+]] = arith.mulf %[[BSX]], %[[CST_B]] : tensor<4xf64>
// CHECK-NEXT: %[[SY:.+]] = arith.addf %[[SX1]], %[[SBCAST]] : tensor<4xf64>
// CHECK-NEXT: return %[[SY]] : tensor<4xf64>

// CHECK-LABEL: func.func private @batched_mixed_returns
// CHECK-SAME: (%[[MRX1:.+]]: tensor<4x3xf64>, %[[MRU1:.+]]: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
// CHECK-NEXT: %[[MRU1_SQ:.+]] = arith.mulf %[[MRU1]], %[[MRU1]] : tensor<3xf64>
// CHECK-NEXT: %[[MRU1_BC:.+]] = "enzyme.broadcast"(%[[MRU1_SQ]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[MRY1:.+]] = arith.addf %[[MRX1]], %[[MRU1_BC]] : tensor<4x3xf64>
// CHECK-NEXT: return %[[MRY1]], %[[MRU1_SQ]] : tensor<4x3xf64>, tensor<3xf64>

// CHECK-LABEL: func.func private @batched_mixed_returns_0
// CHECK-SAME: (%[[MRX2:.+]]: tensor<4x3xf64>, %[[MRU2:.+]]: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<4x3xf64>)
// CHECK-NEXT: %[[MRU2_SQ:.+]] = arith.mulf %[[MRU2]], %[[MRU2]] : tensor<3xf64>
// CHECK-NEXT: %[[MRU2_BC:.+]] = "enzyme.broadcast"(%[[MRU2_SQ]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[MRY2:.+]] = arith.addf %[[MRX2]], %[[MRU2_BC]] : tensor<4x3xf64>
// CHECK-NEXT: return %[[MRY2]], %[[MRU2_BC]] : tensor<4x3xf64>, tensor<4x3xf64>

// CHECK-LABEL: func.func private @batched_non_splat_const
// CHECK-SAME: (%[[NSX:.+]]: tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[NSCST:.+]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
// CHECK-NEXT: %[[NSBC:.+]] = "enzyme.broadcast"(%[[NSCST]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[NSY:.+]] = arith.addf %[[NSX]], %[[NSBC]] : tensor<4x3xf64>
// CHECK-NEXT: return %[[NSY]] : tensor<4x3xf64>

// CHECK-NOT: func.func private @batched_unbatched_helper

// CHECK-LABEL: func.func private @batched_caller_with_helpers
// CHECK-SAME: (%[[HX:.+]]: tensor<4x3xf64>, %[[HU:.+]]: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
// CHECK-NEXT: %[[HU1:.+]] = call @unbatched_helper(%[[HU]]) : (tensor<3xf64>) -> tensor<3xf64>
// CHECK-NEXT: %[[HM:.+]]:2 = call @batched_mixed_helper(%[[HX]], %[[HU1]]) : (tensor<4x3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
// CHECK-NEXT: return %[[HM]]#0, %[[HM]]#1 : tensor<4x3xf64>, tensor<3xf64>

// CHECK-LABEL: func.func private @batched_mixed_helper
// CHECK-SAME: (%[[MHX:.+]]: tensor<4x3xf64>, %[[MHU:.+]]: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
// CHECK-NEXT: %[[MHU2:.+]] = arith.mulf %[[MHU]], %[[MHU]] : tensor<3xf64>
// CHECK-NEXT: %[[MHBC:.+]] = "enzyme.broadcast"(%[[MHU2]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[MHX2:.+]] = arith.addf %[[MHX]], %[[MHBC]] : tensor<4x3xf64>
// CHECK-NEXT: return %[[MHX2]], %[[MHU2]] : tensor<4x3xf64>, tensor<3xf64>

// CHECK-LABEL: func.func private @batched_add2
// CHECK-SAME: (%[[BA0:.+]]: tensor<4x3xf64>, %[[BB0:.+]]: tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[BBC0:.+]] = "enzyme.broadcast"(%[[BB0]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[BS0:.+]] = arith.addf %[[BA0]], %[[BBC0]] : tensor<4x3xf64>
// CHECK-NEXT: return %[[BS0]] : tensor<4x3xf64>

// CHECK-LABEL: func.func private @batched_add2_0
// CHECK-SAME: (%[[BA1:.+]]: tensor<3xf64>, %[[BB1:.+]]: tensor<4x3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[BAC1:.+]] = "enzyme.broadcast"(%[[BA1]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[BS1:.+]] = arith.addf %[[BAC1]], %[[BB1]] : tensor<4x3xf64>
// CHECK-NEXT: return %[[BS1]] : tensor<4x3xf64>

// CHECK-LABEL: func.func private @batched_scf_for_mixed
// CHECK-SAME: (%[[FX:.+]]: tensor<4x3xf64>, %[[FU:.+]]: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>)
// CHECK-NEXT: %[[FUBC:.+]] = "enzyme.broadcast"(%[[FU]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: %[[FR:.+]]:3 = scf.for %[[IV:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ACCB:.+]] = %[[FX]], %[[ACCU:.+]] = %[[FU]], %[[ACCP:.+]] = %[[FUBC]]) -> (tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>)
// CHECK-NEXT:   %[[NU:.+]] = arith.mulf %[[ACCU]], %[[FU]] : tensor<3xf64>
// CHECK-NEXT:   %[[NUBC:.+]] = "enzyme.broadcast"(%[[NU]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:   %[[NB:.+]] = arith.addf %[[ACCB]], %[[NUBC]] : tensor<4x3xf64>
// CHECK-NEXT:   %[[NP:.+]] = arith.addf %[[ACCP]], %[[ACCB]] : tensor<4x3xf64>
// CHECK-NEXT:   scf.yield %[[NB]], %[[NU]], %[[NP]] : tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FR]]#0, %[[FR]]#1, %[[FR]]#2 : tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>

// CHECK-LABEL: func.func private @batched_scf_if_mixed
// CHECK-SAME: (%[[COND:.+]]: i1, %[[IX:.+]]: tensor<4x3xf64>, %[[IU:.+]]: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
// CHECK-NEXT: %[[IUBC:.+]] = "enzyme.broadcast"(%[[IU]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[IR:.+]]:2 = scf.if %[[COND]] -> (tensor<4x3xf64>, tensor<3xf64>) {
// CHECK-NEXT:   %[[IA:.+]] = arith.addf %[[IX]], %[[IUBC]] : tensor<4x3xf64>
// CHECK-NEXT:   scf.yield %[[IA]], %[[IU]] : tensor<4x3xf64>, tensor<3xf64>
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[IB:.+]] = arith.mulf %[[IU]], %[[IU]] : tensor<3xf64>
// CHECK-NEXT:   scf.yield %[[IUBC]], %[[IB]] : tensor<4x3xf64>, tensor<3xf64>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[IR]]#0, %[[IR]]#1 : tensor<4x3xf64>, tensor<3xf64>

// CHECK-LABEL: func.func private @batched_all_unbatched
// CHECK-SAME: (%[[AU:.+]]: tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: %[[AS:.+]] = math.sin %[[AU]] : tensor<3xf64>
// CHECK-NEXT: %[[AP:.+]] = arith.mulf %[[AS]], %[[AU]] : tensor<3xf64>
// CHECK-NEXT: %[[APBC:.+]] = "enzyme.broadcast"(%[[AP]]) <{shape = array<i64: 4>}> : (tensor<3xf64>) -> tensor<4x3xf64>
// CHECK-NEXT: return %[[APBC]] : tensor<4x3xf64>

// CHECK-LABEL: func.func private @batched_cf_cond_br_mixed
// CHECK-SAME: (%[[CCOND:.+]]: i1, %[[CX:.+]]: tensor<2x3x3xf64>, %[[CU:.+]]: tensor<3xf64>) -> (tensor<2x3x3xf64>, tensor<3xf64>)
// CHECK-NEXT: %[[CU2:.+]] = math.sin %[[CU]] : tensor<3xf64>
// CHECK-NEXT: %[[CU2BC:.+]] = "enzyme.broadcast"(%[[CU2]]) <{shape = array<i64: 2, 3>}> : (tensor<3xf64>) -> tensor<2x3x3xf64>
// CHECK-NEXT: cf.cond_br %[[CCOND]], ^[[BB1:bb[0-9]+]](%[[CX]], %[[CU2]] : tensor<2x3x3xf64>, tensor<3xf64>), ^[[BB1]](%[[CU2BC]], %[[CU]] : tensor<2x3x3xf64>, tensor<3xf64>)
// CHECK-NEXT: ^[[BB1]](%[[CB:.+]]: tensor<2x3x3xf64>, %[[CBU:.+]]: tensor<3xf64>):
// CHECK-NEXT: return %[[CB]], %[[CBU]] : tensor<2x3x3xf64>, tensor<3xf64>
