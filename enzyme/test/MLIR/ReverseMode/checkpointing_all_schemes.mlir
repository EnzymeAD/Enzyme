// Compare four differentiation strategies for the same f(x, n) = cos(x*x)^n
// body.  All four must produce equivalent adjoints (math.sin + arith.negf).
//
// PLAIN   — scf.for  , no checkpointing  → full dynamic tape (memref<?xf32>)
// BINOM   — scf.for  , binomial          → static memref<3x..> + Revolve scf.while
// WTAPE   — scf.while, no checkpointing  → enzyme.push/pop dynamic stack
// WONLINE — scf.while, online (SW 2010)  → static memref<3x..> + Online_r2 scf.while
//
// The while-loop variants carry a counter so the condition drives termination
// without any a-priori knowledge of the trip count.

// RUN: %eopt %s --enzyme-wrap="infn=f_plain outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=PLAIN
// RUN: %eopt %s --enzyme-wrap="infn=f_binom outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=BINOM
// RUN: %eopt %s --enzyme-wrap="infn=f_while_plain outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=WTAPE
// RUN: %eopt %s --enzyme-wrap="infn=f_while_online outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-online-progress --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=WONLINE

module {
  // No checkpointing: scf.for with dynamic upper bound %n.
  func.func @f_plain(%arg0: f32, %n: index) -> f32 {
    %lb = arith.constant 0 : index
    %st = arith.constant 1 : index
    %r  = scf.for %iv = %lb to %n step %st iter_args(%s = %arg0) -> f32 {
      %sq  = arith.mulf %s, %s : f32
      %cos = math.cos  %sq     : f32
      scf.yield %cos : f32
    }
    return %r : f32
  }

  // Binomial (offline Revolve) checkpointing: scf.for, budget = 3.
  func.func @f_binom(%arg0: f32, %n: index) -> f32 {
    %lb = arith.constant 0 : index
    %st = arith.constant 1 : index
    %r  = scf.for %iv = %lb to %n step %st iter_args(%s = %arg0) -> f32 {
      %sq  = arith.mulf %s, %s : f32
      %cos = math.cos  %sq     : f32
      scf.yield %cos : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 3 : i64}
    return %r : f32
  }

  // Full-tape scf.while: condition k < n makes the loop run exactly n times,
  // but the AD pass sees no a-priori bound — it discovers the count at runtime
  // by carrying a numIters counter in the scaffold while.
  func.func @f_while_plain(%arg0: f32, %n: index) -> f32 {
    %lb = arith.constant 0 : index
    %st = arith.constant 1 : index
    %r:2 = scf.while (%s = %arg0, %k = %lb) : (f32, index) -> (f32, index) {
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %s, %k : f32, index
    } do {
    ^bb0(%s_in: f32, %k_in: index):
      %sq  = arith.mulf %s_in, %s_in : f32
      %cos = math.cos  %sq           : f32
      %k1  = arith.addi %k_in, %st   : index
      scf.yield %cos, %k1 : f32, index
    }
    return %r#0 : f32
  }

  // Online (Stumm-Walther 2010) checkpointing scf.while: same loop, same
  // trip count, but checkpoints placed adaptively by Online_r2.
  // Generic format is required to attach the enzyme attribute dictionary.
  func.func @f_while_online(%arg0: f32, %n: index) -> f32 {
    %lb = arith.constant 0 : index
    %st = arith.constant 1 : index
    %r:2 = "scf.while"(%arg0, %lb) ({
    ^bb0(%s: f32, %k: index):
      %cond = arith.cmpi slt, %k, %n : index
      scf.condition(%cond) %s, %k : f32, index
    }, {
    ^bb0(%s_in: f32, %k_in: index):
      %sq  = arith.mulf %s_in, %s_in : f32
      %cos = math.cos  %sq           : f32
      %k1  = arith.addi %k_in, %st   : index
      scf.yield %cos, %k1 : f32, index
    }) {enzyme.enable_checkpointing = true,
        enzyme.online_checkpointing,
        enzyme.checkpoint_period = 3 : i64} : (f32, index) -> (f32, index)
    return %r#0 : f32
  }
}

// All four schemes must compute the same adjoint formula per iteration step.
// The body is cos(s*s), so the chain rule gives:
//
//   d(cos(s²))/ds = -sin(s²) · 2s
//
// and the per-step adjoint (with incoming gradient dG) is:
//
//   new_dG = dG · (-sin(s²)) · 2s
//          = dG · negf(sin(mulf(s,s))) · addf(mulf(·,s), mulf(·,s))
//
// Each scheme recovers s differently (load/pop/reconstruct) then applies this
// formula identically.  The checks below pin down that exact sequence in every
// scheme so that all four are verified to compute the same result.

// ------------ f_plain: scf.for, full-tape -----------------------------------
// Full tape allocates a runtime-sized memref sized by the trip count %n.
// PLAIN-LABEL: func.func @f_plain(
// PLAIN-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
// Forward loop stores every primal iterate.
// PLAIN:         memref.alloc(%{{.*}}) : memref<?xf32>
// PLAIN:         scf.for
// PLAIN:         enzyme.store
// Reverse loop: recover s from tape, then apply the adjoint formula.
// PLAIN:         scf.for
// PLAIN:         %[[S_PLAIN:.+]] = enzyme.load %{{.*}} : memref<?xf32>
// PLAIN:         %[[SQ_PLAIN:.+]] = arith.mulf %[[S_PLAIN]], %[[S_PLAIN]]
// PLAIN:         %[[SIN_PLAIN:.+]] = math.sin %[[SQ_PLAIN]]
// PLAIN-NEXT:    %[[NEG_PLAIN:.+]] = arith.negf %[[SIN_PLAIN]]
// PLAIN:         %[[SCALED_PLAIN:.+]] = arith.mulf %{{.*}}, %[[NEG_PLAIN]]
// PLAIN:         arith.mulf %[[SCALED_PLAIN]], %[[S_PLAIN]]
// PLAIN:         arith.mulf %[[SCALED_PLAIN]], %[[S_PLAIN]]
// PLAIN:         arith.addf
// PLAIN:         memref.dealloc {{.*}} : memref<?xf32>
// PLAIN:         return

// ------------ f_binom: scf.for, binomial checkpointing ----------------------
// Revolve uses a statically-sized checkpoint buffer (budget = 3 slots).
// BINOM-LABEL: func.func @f_binom(
// BINOM-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
// BINOM-DAG:     memref.alloc() : memref<3xf32>
// BINOM-DAG:     memref.alloc() : memref<3xindex>
// Forward scf.for checkpoints carried values into the static buffers.
// BINOM:         scf.for
// Outer reverse scf.for drives one adjoint step per primal step; inner
// scf.while runs the offline Revolve reconstruction that delivers s.
// BINOM:         scf.for
// BINOM:         scf.while
// Adjoint formula — anchored to math.sin rather than to the preceding mulf
// because the inner Revolve reconstruction loop also contains mulf ops (for
// the primal remat step); math.sin appears exactly once, in the adjoint.
// BINOM:         %[[SIN_BINOM:.+]] = math.sin
// BINOM-NEXT:    %[[NEG_BINOM:.+]] = arith.negf %[[SIN_BINOM]]
// BINOM:         %[[SCALED_BINOM:.+]] = arith.mulf %{{.*}}, %[[NEG_BINOM]]
// BINOM:         arith.mulf %[[SCALED_BINOM]], %{{.*}}
// BINOM:         arith.mulf %[[SCALED_BINOM]], %{{.*}}
// BINOM:         arith.addf
// BINOM:         memref.dealloc {{.*}} : memref<3xf32>
// BINOM:         memref.dealloc {{.*}} : memref<3xindex>
// BINOM:         return

// ------------ f_while_plain: scf.while, full-tape ---------------------------
// Full tape for scf.while uses a dynamically-growing enzyme.Cache stack
// rather than a pre-allocated memref (since the trip count is unknown).
// WTAPE-LABEL: func.func @f_while_plain(
// WTAPE-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
// The scaffold scf.while carries a numIters counter and pushes the active
// f32 state to the cache at the start of each iteration.
// WTAPE:         !enzyme.Cache<f32>
// WTAPE:         scf.while
// WTAPE:         enzyme.push
// Reverse scf.for: pop s from the LIFO tape, then apply the adjoint formula.
// enzyme.pop stays in generic form "enzyme.pop"(...) after canonicalize, so
// the check uses {{.*}} to match both custom and generic assembly formats.
// WTAPE:         scf.for
// WTAPE:         %[[S_WTAPE:.+]] = {{.*}}enzyme.pop
// WTAPE:         %[[SQ_WTAPE:.+]] = arith.mulf %[[S_WTAPE]], %[[S_WTAPE]]
// WTAPE:         %[[SIN_WTAPE:.+]] = math.sin %[[SQ_WTAPE]]
// WTAPE-NEXT:    %[[NEG_WTAPE:.+]] = arith.negf %[[SIN_WTAPE]]
// WTAPE:         %[[SCALED_WTAPE:.+]] = arith.mulf %{{.*}}, %[[NEG_WTAPE]]
// WTAPE:         arith.mulf %[[SCALED_WTAPE]], %[[S_WTAPE]]
// WTAPE:         arith.mulf %[[SCALED_WTAPE]], %[[S_WTAPE]]
// WTAPE:         arith.addf
// WTAPE:         return

// ------------ f_while_online: scf.while, online (Stumm-Walther) -------------
// Online_r2 adapts checkpoint placement at runtime; uses static buffers for
// both checkpoint data and the Online_r2 state arrays.
// WONLINE-LABEL: func.func @f_while_online(
// WONLINE-SAME:   %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
// Checkpoint storage with budget+1 slots (extra slot for inner Revolve scratch).
// WONLINE-DAG:    memref.alloc() : memref<4xf32>
// WONLINE-DAG:    memref.alloc() : memref<4xindex>
// Online_r2 internal state arrays (freed after the forward pass).
// WONLINE-DAG:    memref.alloc() : memref<3xindex>
// Forward scf.while driven by the Online_r2 state machine (STORE/FORWARD).
// WONLINE:        scf.while
// Online_r2 state arrays freed before the reverse pass.
// WONLINE:        memref.dealloc {{.*}} : memref<3xindex>
// Outer reverse scf.for + inner scf.while reconstructs s via offline Revolve.
// WONLINE:        scf.for
// WONLINE:        scf.while
// Adjoint formula — anchored to math.sin (appears exactly once, in the
// adjoint step; the inner remat loop only contains math.cos for the primal).
// WONLINE:        %[[SIN_WONLINE:.+]] = math.sin
// WONLINE-NEXT:   %[[NEG_WONLINE:.+]] = arith.negf %[[SIN_WONLINE]]
// WONLINE:        %[[SCALED_WONLINE:.+]] = arith.mulf %{{.*}}, %[[NEG_WONLINE]]
// WONLINE:        arith.mulf %[[SCALED_WONLINE]], %{{.*}}
// WONLINE:        arith.mulf %[[SCALED_WONLINE]], %{{.*}}
// WONLINE:        arith.addf
// WONLINE:        memref.dealloc {{.*}} : memref<4xf32>
// WONLINE:        memref.dealloc {{.*}} : memref<4xindex>
// WONLINE:        return
