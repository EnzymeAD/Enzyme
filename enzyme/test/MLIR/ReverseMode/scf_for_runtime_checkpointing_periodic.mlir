// RUN: %eopt %s --enzyme-wrap="infn=dynamic outfn=dynamic_grad argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=dynamic_default outfn=dynamic_default_grad argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=static_default outfn=static_default_grad argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// Periodic checkpointing whose schedule comes from enzyme/checkpoint.h at run
// time (mode 1). The budget is the period; without one, the number of
// segments the compiled schedule would use, or, for a trip count known only
// at run time, 0, for which the runtime takes the square root of the trip
// count.

module {
  func.func @dynamic(%x: f64, %ub: index) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %ub step %c1 iter_args(%s = %x) -> (f64) {
      %sq = arith.mulf %s, %s : f64
      %c = math.cos %sq : f64
      scf.yield %c : f64
    } {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 4 : i64,
       enzyme.checkpoint_runtime}
    return %r : f64
  }

  func.func @dynamic_default(%x: f64, %ub: index) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %ub step %c1 iter_args(%s = %x) -> (f64) {
      %sq = arith.mulf %s, %s : f64
      %c = math.cos %sq : f64
      scf.yield %c : f64
    } {enzyme.enable_checkpointing = true, enzyme.checkpoint_runtime}
    return %r : f64
  }

  func.func @static_default(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%s = %x) -> (f64) {
      %sq = arith.mulf %s, %s : f64
      %c = math.cos %sq : f64
      scf.yield %c : f64
    } {enzyme.enable_checkpointing = true, enzyme.checkpoint_runtime}
    return %r : f64
  }
}

// CHECK-LABEL: func.func private @dynamic_grad(
// CHECK-SAME:      %{{.+}}: f64, %[[UB:.+]]: index, %{{.+}}: f64) -> f64
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i64
// CHECK:         %[[N:.+]] = arith.index_cast %[[UB]] : index to i64
// CHECK:         %[[H:.+]] = call @__enzyme_ckpt_schedule_begin(%[[C1]], %[[C4]], %[[N]])
// CHECK:         scf.while
// CHECK:           func.call @__enzyme_ckpt_schedule_next(%[[H]])
// CHECK:         scf.index_switch
// CHECK:         scf.for
// CHECK:           math.cos
// CHECK:         scf.while
// CHECK:           scf.index_switch
// CHECK:           case 5 {
// CHECK:             math.sin
// CHECK:         call @__enzyme_ckpt_schedule_end(%[[H]])

// CHECK-LABEL: func.func private @dynamic_default_grad(
// CHECK-SAME:      %{{.+}}: f64, %[[UB:.+]]: index, %{{.+}}: f64) -> f64
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i64
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i64
// CHECK:         %[[N:.+]] = arith.index_cast %[[UB]] : index to i64
// CHECK:         %[[H:.+]] = call @__enzyme_ckpt_schedule_begin(%[[C1]], %[[C0]], %[[N]])
// CHECK:         call @__enzyme_ckpt_schedule_end(%[[H]])

// Budget 0: the default split of enzyme/checkpoint_schedule.h, floor(sqrt(10))
// = 3 iterations a segment, 3 segments and a trailing one, as the compiled
// schedule cuts it.
// CHECK-LABEL: func.func private @static_default_grad(
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : i64
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i64
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i64
// CHECK:         %[[H:.+]] = call @__enzyme_ckpt_schedule_begin(%[[C1]], %[[C0]], %[[C10]])
// CHECK:         call @__enzyme_ckpt_schedule_end(%[[H]])
