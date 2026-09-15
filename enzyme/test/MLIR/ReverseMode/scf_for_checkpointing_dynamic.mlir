// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --canonicalize | FileCheck %s

module {
  func.func @main(%arg0: f32, %ub: index) -> (f32) {
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
    %sum = scf.for %iv = %lb to %ub step %step iter_args(%s = %arg0) -> (f32) {
      %sq = arith.mulf %s, %s : f32
      %c = math.cos %sq : f32
      scf.yield %c : f32
    } {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 4 : i64}
    return %sum : f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: index, %arg2: f32) -> f32 {
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index

// The segment length, ceil(numIters / 4), is the only runtime half of the
// decomposition; the checkpoint buffer is sized by the period.
// CHECK-NEXT:    %[[fwdRound:.+]] = arith.addi %arg1, %c3 : index
// CHECK-NEXT:    %[[fwdInner:.+]] = arith.divui %[[fwdRound]], %c4 : index
// CHECK-NEXT:    %[[buf:.+]] = memref.alloc() : memref<4xf32>

// Forward: exactly 4 segments, one checkpoint each, the recompute loop bounded
// by the saturating clamp.
// CHECK-NEXT:    %[[fwd:.+]] = scf.for %[[j:.+]] = %c0 to %c4 step %c1 iter_args(%[[st:.+]] = %arg0) -> (f32) {
// CHECK-NEXT:      memref.store %[[st]], %[[buf]][%[[j]]] : memref<4xf32>
// CHECK-NEXT:      %[[fbase:.+]] = arith.muli %[[j]], %[[fwdInner]] : index
// CHECK-NEXT:      %[[fbaseC:.+]] = arith.minui %[[fbase]], %arg1 : index
// CHECK-NEXT:      %[[fleft:.+]] = arith.subi %arg1, %[[fbaseC]] : index
// CHECK-NEXT:      %[[flen:.+]] = arith.minui %[[fwdInner]], %[[fleft]] : index
// CHECK-NEXT:      %[[fseg:.+]] = scf.for %{{.+}} = %c0 to %[[flen]] step %c1 iter_args(%[[fs:.+]] = %[[st]]) -> (f32) {
// CHECK-NEXT:        %[[fsq:.+]] = arith.mulf %[[fs]], %[[fs]] : f32
// CHECK-NEXT:        %[[fcos:.+]] = math.cos %[[fsq]] : f32
// CHECK-NEXT:        scf.yield %[[fcos]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[fseg]] : f32
// CHECK-NEXT:    }

// The reverse pass recomputes the segment length from the cached trip count
// rather than transporting the forward pass's own value.
// CHECK-NEXT:    %[[revRound:.+]] = arith.addi %arg1, %c3 : index
// CHECK-NEXT:    %[[revInner:.+]] = arith.divui %[[revRound]], %c4 : index

// Reverse: the same 4 segments, replayed back to front (segment 3 - j), each
// re-clamped against its own base.
// CHECK-NEXT:    %[[rev:.+]] = scf.for %[[i:.+]] = %c0 to %c4 step %c1 iter_args(%[[d:.+]] = %arg2) -> (f32) {
// CHECK-NEXT:      %[[slot:.+]] = arith.subi %c3, %[[i]] : index
// CHECK-NEXT:      %[[ckpt:.+]] = memref.load %[[buf]][%[[slot]]] : memref<4xf32>
// CHECK-NEXT:      %[[k:.+]] = arith.subi %c3, %[[i]] : index
// CHECK-NEXT:      %[[rbase:.+]] = arith.muli %[[k]], %[[revInner]] : index
// CHECK-NEXT:      %[[rbaseC:.+]] = arith.minui %[[rbase]], %arg1 : index
// CHECK-NEXT:      %[[rleft:.+]] = arith.subi %arg1, %[[rbaseC]] : index
// CHECK-NEXT:      %[[rlen:.+]] = arith.minui %[[revInner]], %[[rleft]] : index
// CHECK-NEXT:      %[[tape:.+]] = memref.alloc(%[[rlen]]) : memref<?xf32>
// CHECK-NEXT:      %[[replay:.+]] = scf.for %[[l:.+]] = %c0 to %[[rlen]] step %c1 iter_args(%[[rs:.+]] = %[[ckpt]]) -> (f32) {
// CHECK-NEXT:        enzyme.store %[[rs]], %[[tape]][%[[l]]] ([%[[rlen]]]) : memref<?xf32>
// CHECK-NEXT:        %[[rsq:.+]] = arith.mulf %[[rs]], %[[rs]] : f32
// CHECK-NEXT:        %[[rcos:.+]] = math.cos %[[rsq]] : f32
// CHECK-NEXT:        scf.yield %[[rcos]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[adj:.+]] = scf.for %[[m:.+]] = %c0 to %[[rlen]] step %c1 iter_args(%[[ad:.+]] = %[[d]]) -> (f32) {
// CHECK-NEXT:        %[[last:.+]] = arith.subi %[[rlen]], %c1 : index
// CHECK-NEXT:        %[[ridx:.+]] = arith.subi %[[last]], %[[m]] : index
// CHECK-NEXT:        %[[v:.+]] = enzyme.load %[[tape]][%[[ridx]]] ([%[[rlen]]]) : memref<?xf32>
// CHECK-NEXT:        %[[vsq:.+]] = arith.mulf %[[v]], %[[v]] : f32
// CHECK-NEXT:        %[[sin:.+]] = math.sin %[[vsq]] fastmath<fast> : f32
// CHECK-NEXT:        %[[neg:.+]] = arith.negf %[[sin]] fastmath<fast> : f32
// CHECK-NEXT:        %[[dsq:.+]] = arith.mulf %[[ad]], %[[neg]] fastmath<fast> : f32
// CHECK-NEXT:        %[[dl:.+]] = arith.mulf %[[dsq]], %[[v]] fastmath<fast> : f32
// CHECK-NEXT:        %[[dr:.+]] = arith.mulf %[[dsq]], %[[v]] fastmath<fast> : f32
// CHECK-NEXT:        %[[ds:.+]] = arith.addf %[[dl]], %[[dr]] fastmath<fast> : f32
// CHECK-NEXT:        scf.yield %[[ds]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[tape]] : memref<?xf32>
// CHECK-NEXT:      scf.yield %[[adj]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[buf]] : memref<4xf32>
// CHECK-NEXT:    return %[[rev]] : f32
// CHECK-NEXT:  }
