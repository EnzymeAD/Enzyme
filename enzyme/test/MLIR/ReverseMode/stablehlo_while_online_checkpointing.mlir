// StableHLO counterpart of checkpointing_all_schemes.mlir.
//
// Tests reverse-mode AD of stablehlo.while with two strategies:
//   PLAIN   — full-tape (enzyme.push/pop Cache)
//   ONLINE  — Online_r2 (Stumm-Walther 2010), budget = 3, pure StableHLO V2
//
// Body: f(x, n) = cos(x²) iterated n times, where n is unknown at compile time.
// The condition (step < n) drives termination; the while carries a step counter
// as tensor<i64>.
//
// Both must produce the adjoint of cos(x²): -sin(x²) · 2x, i.e. stablehlo.sine
// and stablehlo.negate in the reverse pass.
//
// REQUIRES: stablehlo
//   This test needs enzymemlir-opt built with ENZYME_ENABLE_STABLEHLO.
//   Skipped when the feature is not available (standard CMake build).

// RUN: %eopt %s --enzyme-wrap="infn=f_plain outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=PLAIN
// RUN: %eopt %s --enzyme-wrap="infn=f_online outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s --check-prefix=ONLINE

module {

  // ── PLAIN: stablehlo.while, no checkpointing (full tape) ─────────────────
  // Trip count unknown at compile time; AD discovers it at runtime via a
  // numIters counter in the scaffold while.
  func.func @f_plain(%arg0: tensor<f32>, %n: tensor<i64>) -> tensor<f32> {
    %zero = "stablehlo.constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %one  = "stablehlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>

    %r:2 = "stablehlo.while"(%arg0, %zero) ({
    ^bb0(%s: tensor<f32>, %k: tensor<i64>):
      %cond = "stablehlo.compare"(%k, %n)
        {comparison_direction = #stablehlo<comparison_direction LT>}
        : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%cond) : (tensor<i1>) -> ()
    }, {
    ^bb0(%s_in: tensor<f32>, %k_in: tensor<i64>):
      %sq  = "stablehlo.multiply"(%s_in, %s_in) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %cos = "stablehlo.cosine"(%sq) : (tensor<f32>) -> tensor<f32>
      %k1  = "stablehlo.add"(%k_in, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%cos, %k1) : (tensor<f32>, tensor<i64>) -> ()
    }) : (tensor<f32>, tensor<i64>) -> (tensor<f32>, tensor<i64>)

    return %r#0 : tensor<f32>
  }

  // ── ONLINE: stablehlo.while, Online_r2 (Stumm-Walther 2010), budget = 3 ─
  // Same loop body, but with enzyme.online_checkpointing attributes.
  // The V2 implementation keeps the forward scaffold and reverse reconstruction
  // entirely in StableHLO (no SCF, no memref, no OnlineProgressOp).
  func.func @f_online(%arg0: tensor<f32>, %n: tensor<i64>) -> tensor<f32> {
    %zero = "stablehlo.constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %one  = "stablehlo.constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>

    %r:2 = "stablehlo.while"(%arg0, %zero) ({
    ^bb0(%s: tensor<f32>, %k: tensor<i64>):
      %cond = "stablehlo.compare"(%k, %n)
        {comparison_direction = #stablehlo<comparison_direction LT>}
        : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%cond) : (tensor<i1>) -> ()
    }, {
    ^bb0(%s_in: tensor<f32>, %k_in: tensor<i64>):
      %sq  = "stablehlo.multiply"(%s_in, %s_in) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %cos = "stablehlo.cosine"(%sq) : (tensor<f32>) -> tensor<f32>
      %k1  = "stablehlo.add"(%k_in, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%cos, %k1) : (tensor<f32>, tensor<i64>) -> ()
    }) {enzyme.enable_checkpointing = true,
        enzyme.online_checkpointing,
        enzyme.checkpoint_period = 3 : i64}
      : (tensor<f32>, tensor<i64>) -> (tensor<f32>, tensor<i64>)

    return %r#0 : tensor<f32>
  }
}

// All checks below verify the adjoint formula for cos(x²):
//
//   d/dx cos(x²) = -sin(x²) · 2x
//
// Each scheme recovers x differently (pop from cache vs. load from checkpoint
// buffer) then applies the same formula.

// ── PLAIN: full-tape stablehlo.while ─────────────────────────────────────────
// The scaffold while pushes every primal state to an enzyme.Cache; the reverse
// pass pops them in LIFO order.
//
// PLAIN-LABEL: func.func @f_plain(
// PLAIN-SAME:    %{{.*}}: tensor<f32>, %{{.*}}: tensor<i64>, %{{.*}}: tensor<f32>
//
// Forward scaffold while with enzyme.push of the active state.
// PLAIN:         !enzyme.Cache<tensor<f32>>
// PLAIN:         "stablehlo.while"
// PLAIN:         "enzyme.push"
//
// Reverse pass: pop from cache, then adjoint formula.
// PLAIN:         "enzyme.pop"
// PLAIN:         "stablehlo.sine"
// PLAIN:         "stablehlo.negate"
// PLAIN:         return

// ── ONLINE: Online_r2 pure StableHLO V2 ─────────────────────────────────────
// The forward pass uses a single stablehlo.while that carries:
//   - the primal state (tensor<f32>)
//   - a step counter (tensor<i64>)
//   - stacked checkpoint buffers: tensor<4xf32> for data, tensor<4xi64> for
//     step indices (budget+1 = 4 slots)
//   - Online_r2 state machine scalars and arrays
//
// The reverse pass is an outer stablehlo.while that counts backwards, with
// an inner stablehlo.while for offline Revolve reconstruction from the
// nearest checkpoint.
//
// ONLINE-LABEL: func.func @f_online(
// ONLINE-SAME:    %{{.*}}: tensor<f32>, %{{.*}}: tensor<i64>, %{{.*}}: tensor<f32>
//
// Forward scaffold while — budget+1 = 4 checkpoint slots carried as iter-args.
// The stacked checkpoint data buffer has shape tensor<4xf32> (one f32 per slot).
// ONLINE:         "stablehlo.while"
//
// Reverse pass: outer stablehlo.while counts backwards over discovered steps.
// ONLINE:         "stablehlo.while"
//
// Inner Revolve reconstruction: a nested stablehlo.while replays the primal
// from the nearest checkpoint to reconstruct the needed intermediate state.
// ONLINE:         "stablehlo.while"
//
// Adjoint of cos(x²): stablehlo.sine and stablehlo.negate.
// ONLINE:         "stablehlo.sine"
// ONLINE:         "stablehlo.negate"
// ONLINE:         return
