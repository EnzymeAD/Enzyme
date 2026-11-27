// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-online-progress --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// Online (Stumm-Walther) checkpointing of an scf.for with a static trip
// count (8 iterations, budget = 3 snapshots).  The forward pass is an
// scf.while driven by the Online_r2 state machine; the reverse pass uses
// offline Revolve (BinomialProgressOp) for optimal reconstruction.

module {
  func.func @main(%arg0: f32) -> f32 {
    %lb   = arith.constant 0 : index
    %ub   = arith.constant 8 : index
    %step = arith.constant 1 : index

    %result = scf.for %iv = %lb to %ub step %step
        iter_args(%s = %arg0) -> f32 {
      %sq  = arith.mulf %s, %s   : f32
      %cos = math.cos  %sq       : f32
      scf.yield %cos : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.online_checkpointing,
       enzyme.checkpoint_period = 3 : i64}

    return %result : f32
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:    %[[X:.+]]: f32, %[[DRET:.+]]: f32

// ── Forward pass ─────────────────────────────────────────────────────────────
// Checkpoint state buffer (3 f32 snapshots of the loop-carried value).
// CHECK-DAG:     %[[STATE:.+]] = memref.alloc() : memref<3xf32>
// Checkpoint step-index buffer (records which forward step each slot holds).
// CHECK-DAG:     %[[IDX:.+]] = memref.alloc() : memref<3xindex>
// Online_r2 internal arrays (at least two of ch/ord_ch/num_rep survive DCE).
// CHECK-DAG:     memref.alloc() : memref<3xindex>

// The forward pass is an scf.while; Online_r2 drives STORE/FORWARD each step.
// CHECK:         scf.while

// The state machine is implemented as arith comparisons after lowering.
// CHECK:         arith.cmpi

// STORE branch: checkpoint the current iter-arg into slot sp.
// CHECK:         memref.store {{.*}}, %[[STATE]]
// CHECK:         memref.store {{.*}}, %[[IDX]]

// FORWARD branch: one step of the primal body.
// CHECK:         arith.mulf
// CHECK:         math.cos

// Online_r2 arrays freed after the while loop (before the reverse pass).
// CHECK:         memref.dealloc {{.*}} : memref<3xindex>

// ── Reverse pass ─────────────────────────────────────────────────────────────
// Outer loop: one iteration per primal step, running in reverse.
// CHECK:         scf.for %{{.*}} = %c0 to %{{.*}} step %c1

// Load the top checkpoint and its recorded primal step position.
// CHECK:         memref.load %[[STATE]]
// CHECK:         memref.load %[[IDX]]

// Inner Revolve reconstruction while loop rebuilds the state just before each step.
// CHECK:         scf.while

// Adjoint of cos(x*x): the chain rule produces -sin(x*x) * 2x.
// CHECK:         math.sin

// Checkpoint buffers freed after the reverse loop completes.
// CHECK-DAG:     memref.dealloc %[[STATE]]
// CHECK-DAG:     memref.dealloc %[[IDX]]

// CHECK:         return
