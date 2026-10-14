// Verify that uncheckpointed reverse, binomial checkpointing, and online
// checkpointing all produce numerically equivalent adjoints for the same loop.
// Each RUN line differentiates one of the three identically-bodied functions.
// All three must contain math.sin and arith.negf (adjoint of cos(x*x)).

// RUN: %eopt %s --enzyme-wrap="infn=f_plain outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=PLAIN
// RUN: %eopt %s --enzyme-wrap="infn=f_binom outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=BINOM
// RUN: %eopt %s --enzyme-wrap="infn=f_online outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzyme-online-progress --lower-enzyme-binomial-progress --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s --check-prefix=ONLINE

// Three functions with identical bodies, differing only in checkpointing attrs.
// Body: f(x) = cos(x*x) iterated n times, n unknown at compile time.
// A dynamic upper bound makes online checkpointing meaningful (trip count not
// known at compile time; online algorithm adapts checkpoint placement at run
// time without a pre-computed Revolve schedule).  df/dx involves -sin(x^2)*2x.

module {
  // No checkpointing: %n is const-enzyme to keep the signature comparable.
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

  // Binomial checkpointing; %n is a dynamic runtime bound.
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

  // Online checkpointing; %n is a dynamic runtime bound — trip count not known
  // at compile time, which is the intended use case for the online algorithm.
  func.func @f_online(%arg0: f32, %n: index) -> f32 {
    %lb = arith.constant 0 : index
    %st = arith.constant 1 : index
    %r  = scf.for %iv = %lb to %n step %st iter_args(%s = %arg0) -> f32 {
      %sq  = arith.mulf %s, %s : f32
      %cos = math.cos  %sq     : f32
      scf.yield %cos : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.online_checkpointing,
       enzyme.checkpoint_period = 3 : i64}
    return %r : f32
  }
}

// Checks follow the IR output sequence: signature → allocs → loops →
// adjoint ops → deallocs.  Plain comment lines have no prefix to avoid
// being parsed as check patterns.

// Full-tape (no checkpointing): (f32 active, index const, f32 dret).
// PLAIN-LABEL: func.func @f_plain(
// PLAIN-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
// PLAIN:         math.sin
// PLAIN:         arith.negf

// Revolve (binomial): (f32 active, index const, f32 dret).
// BINOM-LABEL: func.func @f_binom(
// BINOM-SAME:    %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
// BINOM:         memref.alloc() : memref<3xf32>
// BINOM:         memref.alloc() : memref<3xindex>
// BINOM:         scf.for
// BINOM:         scf.while
// BINOM:         math.sin
// BINOM:         arith.negf
// BINOM:         memref.dealloc {{.*}} : memref<3xf32>
// BINOM:         memref.dealloc {{.*}} : memref<3xindex>

// Stumm-Walther (online): (f32 active, index const, f32 dret).
// ONLINE-LABEL: func.func @f_online(
// ONLINE-SAME:   %{{.*}}: f32, %{{.*}}: index, %{{.*}}: f32
// ONLINE:        memref.alloc() : memref<3xf32>
// ONLINE:        memref.alloc() : memref<3xindex>
// ONLINE:        memref.alloc() : memref<3xindex>
// ONLINE:        scf.while
// ONLINE:        scf.while
// ONLINE:        math.sin
// ONLINE:        arith.negf
// ONLINE:        memref.dealloc {{.*}} : memref<3xf32>
// ONLINE:        memref.dealloc {{.*}} : memref<3xindex>
