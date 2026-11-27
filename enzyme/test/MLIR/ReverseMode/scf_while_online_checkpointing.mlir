// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-online-progress --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

// Online checkpointing of an scf.while loop whose trip count is truly unknown
// before the loop runs — the canonical use case for the online algorithm.
//
// f(x, n) = cos(x*x) applied n times, where n is only known at runtime.
// The while loop terminates when a step counter reaches n.
// The generic op format is used to attach enzyme attributes since scf.while's
// custom assembly format does not accept a trailing attribute dictionary.

module {
  func.func @main(%arg0: f32, %n: index) -> f32 {
    %step = arith.constant 1 : index
    %lb   = arith.constant 0 : index

    %result:2 = "scf.while"(%arg0, %lb) ({
    ^bb0(%s: f32, %k: index):
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %s, %k : f32, index
    }, {
    ^bb0(%s_in: f32, %k_in: index):
      %sq  = arith.mulf %s_in, %s_in : f32
      %cos = math.cos  %sq           : f32
      %k1  = arith.addi %k_in, %step : index
      scf.yield %cos, %k1 : f32, index
    }) {enzyme.enable_checkpointing = true,
        enzyme.online_checkpointing,
        enzyme.checkpoint_period = 3 : i64} : (f32, index) -> (f32, index)

    return %result#0 : f32
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32

// ── Forward pass ─────────────────────────────────────────────────────────────
// Checkpoint buffers allocated with budget+1 slots so the inner Revolve
// reconstruction has a scratch slot without overflow.
// CHECK-DAG:     memref.alloc() : memref<4xf32>
// CHECK-DAG:     memref.alloc() : memref<4xindex>

// Online_r2 state arrays (size = budget = 3).
// CHECK-DAG:     memref.alloc() : memref<3xindex>

// The forward scaffold is an scf.while driven by the Online_r2 state machine;
// the primal's body (mulf + cos + addi) appears inside the FORWARD branch.
// CHECK:         scf.while
// CHECK:         arith.mulf
// CHECK:         math.cos

// Online_r2 state arrays freed after the forward pass.
// CHECK:         memref.dealloc {{.*}} : memref<3xindex>

// ── Reverse pass ─────────────────────────────────────────────────────────────
// Outer reverse loop over all discovered primal steps.
// CHECK:         scf.for

// Inner Revolve reconstruction while loop.
// CHECK:         scf.while

// Adjoint of cos(x*x): -sin(x*x)*2x.
// CHECK:         math.sin
// CHECK:         arith.negf

// Checkpoint buffers freed.
// CHECK-DAG:     memref.dealloc {{.*}} : memref<4xf32>
// CHECK-DAG:     memref.dealloc {{.*}} : memref<4xindex>

// CHECK:         return
