// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// Binomial checkpointing whose schedule comes from enzyme/checkpoint.h at run
// time: Revolve (mode 2) with the period (3) as its budget. The snapshots
// stay in a buffer with one row per slot the schedule names, plus one for the
// state before the last step.

module {
  func.func @main(%arg0: f64) -> (f64) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 10 : index
    %step = arith.constant 1 : index
    %sum = scf.for %iv = %lb to %ub step %step iter_args(%s = %arg0) -> (f64) {
      %sq = arith.mulf %s, %s : f64
      %c = math.cos %sq : f64
      scf.yield %c : f64
    } {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 3 : i64, enzyme.checkpoint_runtime}
    return %sum : f64
  }
}

// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_begin(i64, i64, i64) -> !llvm.ptr
// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_next(!llvm.ptr) -> i32
// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_flag(!llvm.ptr) -> i32
// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_iteration(!llvm.ptr) -> i64
// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_start(!llvm.ptr) -> i64
// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_slot(!llvm.ptr) -> i64
// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_slots(!llvm.ptr) -> i64
// CHECK-DAG: func.func private @__enzyme_ckpt_schedule_end(!llvm.ptr)

// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[X:.+]]: f64, %[[DRET:.+]]: f64) -> f64
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : i64
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : i64
// Schedule 4, the binomial schedule the compiled form follows.
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : i64
// CHECK:         %[[H:.+]] = call @__enzyme_ckpt_schedule_begin(%[[C4]], %[[C3]], %[[C10]])
// CHECK:         %[[NS:.+]] = call @__enzyme_ckpt_schedule_slots(%[[H]])
// CHECK:         %[[LAST:.+]] = arith.index_cast %[[NS]] : i64 to index
// CHECK:         %[[ROWS:.+]] = arith.addi %[[LAST]], %{{.+}} : index
// CHECK:         %[[BUF:.+]] = memref.alloc(%[[ROWS]]) : memref<?xf64>

// The forward sweep, up to the first turn (4) or done (7).
// CHECK:         %[[FWD:.+]]:2 = scf.while (%[[S:.+]] = %[[X]]) : (f64) -> (index, f64) {
// CHECK:           %[[NEXT:.+]] = func.call @__enzyme_ckpt_schedule_next(%[[H]])
// CHECK:           %[[FLAG:.+]] = arith.index_cast %[[NEXT]] : i32 to index
// CHECK:           arith.cmpi ne, %[[FLAG]], %{{.+}} : index
// CHECK:           arith.cmpi ne, %[[FLAG]], %{{.+}} : index
// CHECK:           scf.condition
// CHECK:         } do {
// CHECK:         ^bb0(%[[F:.+]]: index, %[[ST:.+]]: f64):
// CHECK:           func.call @__enzyme_ckpt_schedule_iteration(%[[H]])
// CHECK:           func.call @__enzyme_ckpt_schedule_start(%[[H]])
// CHECK:           %[[SLOT64:.+]] = func.call @__enzyme_ckpt_schedule_slot(%[[H]])
// CHECK:           %[[SLOT:.+]] = arith.index_cast %[[SLOT64]] : i64 to index
// CHECK:           scf.index_switch %[[F]]
// CHECK-NEXT:      case 1 {
// CHECK-NEXT:        memref.store %[[ST]], %[[BUF]][%[[SLOT]]] : memref<?xf64>
// CHECK:           case 2 {
// CHECK-NEXT:        memref.load %[[BUF]][%[[SLOT]]] : memref<?xf64>
// CHECK:           case 3 {
// CHECK-NEXT:        scf.for
// CHECK:               math.cos
// CHECK:           default {

// The state before the last step goes to the extra row.
// CHECK:         memref.store %[[FWD]]#1, %[[BUF]][%[[LAST]]] : memref<?xf64>

// The reverse sweep: from that row, at the first turn, until done; a turn
// (FIRSTUTURN mapped to UTURN, 5) differentiates one step.
// CHECK:         %[[FL:.+]] = call @__enzyme_ckpt_schedule_flag(%[[H]])
// CHECK:         %[[FLI:.+]] = arith.index_cast %[[FL]] : i32 to index
// CHECK:         %[[S0:.+]] = memref.load %[[BUF]][%{{.+}}] : memref<?xf64>
// CHECK:         %[[REV:.+]]:3 = scf.while (%{{.+}} = %[[FLI]], %{{.+}} = %[[S0]], %{{.+}} = %[[DRET]]) : (index, f64, f64) -> (index, f64, f64) {
// CHECK:         } do {
// CHECK:           %[[ISFIRST:.+]] = arith.cmpi eq
// CHECK:           %[[SEL:.+]] = arith.select %[[ISFIRST]]
// CHECK:           scf.index_switch %[[SEL]]
// CHECK:           case 1 {
// CHECK-NEXT:        memref.store
// CHECK:           case 2 {
// CHECK-NEXT:        memref.load
// CHECK:           case 3 {
// CHECK-NEXT:        scf.for
// CHECK:               math.cos
// CHECK:           case 5 {
// CHECK:             arith.mulf
// CHECK:             math.sin
// CHECK:           func.call @__enzyme_ckpt_schedule_next(%[[H]])
// CHECK:         memref.dealloc %[[BUF]] : memref<?xf64>
// CHECK:         call @__enzyme_ckpt_schedule_end(%[[H]])
// CHECK:         return %[[REV]]#2 : f64
