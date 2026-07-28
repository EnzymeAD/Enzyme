// RUN: %eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops | FileCheck %s

// Check that the accumulated gradients of a memref that is both read and
// written to inside the loop (%buf[%i] is read, %buf[%c0] is written every
// iteration) end up accumulated into the shadow memref (%arg1), the dup of
// the original argument, rather than being lost or accumulated somewhere
// else.

func.func @reduce_sum(%buf: memref<10xf64>) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0.0 : f64

  %sum = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %buf[%i] : memref<10xf64>
    %new_acc = arith.addf %acc, %val : f64
    memref.store %new_acc, %buf[%c0] : memref<10xf64>
    scf.yield %new_acc : f64
  } {enzyme.enable_checkpointing = false,
     enzyme.checkpoint_period=4, enzyme.disable_mincut=true}

  return %sum : f64
}

// CHECK-LABEL:   func.func @reduce_sum(
// CHECK-SAME:      %[[ARG0:.*]]: memref<10xf64>, %[[ARG1:.*]]: memref<10xf64>, %[[ARG2:.*]]: f64) {
// CHECK:           %[[C9:.*]] = arith.constant 9 : index
// CHECK:           %[[C10:.*]] = arith.constant 10 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<10xmemref<10xf64>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<10xindex>
// CHECK:           %[[ALLOC_1:.*]] = memref.alloc() : memref<10xindex>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[CST]]) -> (f64) {
// CHECK:             memref.store %[[IV]], %[[ALLOC_0]]{{\[}}%[[IV]]] : memref<10xindex>
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[IV]]] : memref<10xf64>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[ACC]], %[[LOAD_0]] : f64
// Every iteration caches the shadow memref itself (%[[ARG1]]), not a fresh
// per-iteration copy -- it is a mutable handle whose identity must be
// preserved across iterations.
// CHECK:             memref.store %[[ARG1]], %[[ALLOC]]{{\[}}%[[IV]]] : memref<10xmemref<10xf64>>
// CHECK:             memref.store %[[C0]], %[[ALLOC_1]]{{\[}}%[[IV]]] : memref<10xindex>
// CHECK:             memref.store %[[ADDF_0]], %[[ARG0]]{{\[}}%[[C0]]] : memref<10xf64>
// CHECK:             scf.yield %[[ADDF_0]] : f64
// CHECK:           }
// CHECK:           %[[ADDF_1:.*]] = arith.addf %[[ARG2]], %[[CST]] : f64
// CHECK:           %[[FOR_1:.*]] = scf.for %[[IV_REV:.*]] = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%[[DACC:.*]] = %[[ADDF_1]]) -> (f64) {
// CHECK:             %[[IDX:.*]] = arith.subi %[[C9]], %[[IV_REV]] : index
// CHECK:             %[[ADDF_2:.*]] = arith.addf %[[DACC]], %[[CST]] : f64
// CHECK:             %[[SHADOW:.*]] = memref.load %[[ALLOC]]{{\[}}%[[IDX]]] : memref<10xmemref<10xf64>>
// CHECK:             %[[STOREIDX:.*]] = memref.load %[[ALLOC_1]]{{\[}}%[[IDX]]] : memref<10xindex>
// CHECK:             %[[DVAL_0:.*]] = memref.load %[[SHADOW]]{{\[}}%[[STOREIDX]]] : memref<10xf64>
// CHECK:             %[[ADDF_3:.*]] = arith.addf %[[ADDF_2]], %[[DVAL_0]] : f64
// CHECK:             memref.store %[[CST]], %[[SHADOW]]{{\[}}%[[STOREIDX]]] : memref<10xf64>
// CHECK:             %[[ADDF_4:.*]] = arith.addf %[[ADDF_3]], %[[CST]] : f64
// CHECK:             %[[ADDF_5:.*]] = arith.addf %[[ADDF_3]], %[[CST]] : f64
// CHECK:             %[[LOADIDX:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[IDX]]] : memref<10xindex>
// CHECK:             %[[DVAL_1:.*]] = memref.load %[[SHADOW]]{{\[}}%[[LOADIDX]]] : memref<10xf64>
// CHECK:             %[[ADDF_6:.*]] = arith.addf %[[DVAL_1]], %[[ADDF_5]] : f64
// The gradient contribution from the load is accumulated back into the
// shadow memref (%[[SHADOW]], i.e. %[[ARG1]]) in place.
// CHECK:             memref.store %[[ADDF_6]], %[[SHADOW]]{{\[}}%[[LOADIDX]]] : memref<10xf64>
// CHECK:             scf.yield %[[ADDF_4]] : f64
// CHECK:           } {enzyme.disable_mincut = true}
// CHECK:           memref.dealloc %[[ALLOC_1]] : memref<10xindex>
// CHECK:           memref.dealloc %[[ALLOC_0]] : memref<10xindex>
// CHECK:           memref.dealloc %[[ALLOC]] : memref<10xmemref<10xf64>>
// CHECK:           return
// CHECK:         }
