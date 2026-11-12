// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

module {

  func.func @main(%m: memref<f32>) -> f32 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 9 : index
    %one = arith.constant 1 : index

    %init = arith.constant 0.0 : f32
    %0 = scf.for %i = %lb to %ub step %one iter_args(%it = %init) -> (f32) {
      %val = memref.load %m[] : memref<f32>
      %mul = arith.mulf %val, %it : f32
      scf.yield %mul : f32
    } {enzyme.enable_checkpointing = true}

    return %0 : f32
  }

}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:    %[[ARG0:.+]]: memref<f32>, %[[ARG1:.+]]: memref<f32>, %[[ARG2:.+]]: f32) {
// CHECK-NEXT:    %[[C2:.+]] = arith.constant 2 : index
// CHECK-NEXT:    %[[C3:.+]] = arith.constant 3 : index
// CHECK-NEXT:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[ALLOC:.+]] = memref.alloc() : memref<3xf32>
// CHECK-NEXT:    %{{.+}} = scf.for %[[OUT_IV:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[OUT_IT:.+]] = %[[CST]]) -> (f32) {
// CHECK-NEXT:      memref.store %[[OUT_IT]], %[[ALLOC]][%[[OUT_IV]]] : memref<3xf32>
// CHECK-NEXT:      %[[FWD_INNER:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[IN_IT:.+]] = %[[OUT_IT]]) -> (f32) {
// CHECK-NEXT:        %[[VAL:.+]] = memref.load %[[ARG0]][] : memref<f32>
// CHECK-NEXT:        %[[MUL:.+]] = arith.mulf %[[VAL]], %[[IN_IT]] : f32
// CHECK-NEXT:        scf.yield %[[MUL]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[FWD_INNER]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.+}} = scf.for %[[REV_IV:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[REV_OUT_ADJ:.+]] = %[[ARG2]]) -> (f32) {
// CHECK-NEXT:      %[[SEG:.+]] = arith.subi %[[C2]], %[[REV_IV]] : index
// CHECK-NEXT:      %[[CKPT_VAL:.+]] = memref.load %[[ALLOC]][%[[SEG]]] : memref<3xf32>
// CHECK-NEXT:      %[[CACHED_VAL:.+]] = memref.alloc() : memref<3xf32>
// CHECK-NEXT:      %[[CACHED_IT:.+]] = memref.alloc() : memref<3xf32>
// CHECK-NEXT:      %{{.+}} = scf.for %[[REP_IV:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[REP_IT:.+]] = %[[CKPT_VAL]]) -> (f32) {
// CHECK-NEXT:        memref.store %[[REP_IT]], %[[CACHED_IT]][%[[REP_IV]]] : memref<3xf32>
// CHECK-NEXT:        %[[RVAL:.+]] = memref.load %[[ARG0]][] : memref<f32>
// CHECK-NEXT:        memref.store %[[RVAL]], %[[CACHED_VAL]][%[[REP_IV]]] : memref<3xf32>
// CHECK-NEXT:        %[[RMUL:.+]] = arith.mulf %[[RVAL]], %[[REP_IT]] : f32
// CHECK-NEXT:        scf.yield %[[RMUL]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[REV_INNER:.+]] = scf.for %[[ADJ_IV:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ADJ_IT:.+]] = %[[REV_OUT_ADJ]]) -> (f32) {
// CHECK-NEXT:        %[[REV_STEP:.+]] = arith.subi %[[C2]], %[[ADJ_IV]] : index
// CHECK-NEXT:        %[[LV:.+]] = memref.load %[[CACHED_VAL]][%[[REV_STEP]]] : memref<3xf32>
// CHECK-NEXT:        %[[LI:.+]] = memref.load %[[CACHED_IT]][%[[REV_STEP]]] : memref<3xf32>
// CHECK-NEXT:        %[[DMUL_V:.+]] = arith.mulf %[[ADJ_IT]], %[[LI]] fastmath<fast> : f32
// CHECK-NEXT:        %[[DMUL_I:.+]] = arith.mulf %[[ADJ_IT]], %[[LV]] fastmath<fast> : f32
// CHECK-NEXT:        %[[DARG1:.+]] = memref.load %[[ARG1]][] : memref<f32>
// CHECK-NEXT:        %[[DARG1_NEXT:.+]] = arith.addf %[[DARG1]], %[[DMUL_V]] fastmath<fast> : f32
// CHECK-NEXT:        memref.store %[[DARG1_NEXT]], %[[ARG1]][] : memref<f32>
// CHECK-NEXT:        scf.yield %[[DMUL_I]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[CACHED_IT]] : memref<3xf32>
// CHECK-NEXT:      memref.dealloc %[[CACHED_VAL]] : memref<3xf32>
// CHECK-NEXT:      scf.yield %[[REV_INNER]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[ALLOC]] : memref<3xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
