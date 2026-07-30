// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s
// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --lower-llvm-ext --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s --check-prefix=LOWER

// Binomial checkpointing of a bare !llvm.ptr, the shape a CUDA program hits:
// the loop mutates memory reached through a pointer whose extent is not in its
// type, so the size comes from llvm_ext.ptr_size_hint and the clone buffer holds
// pointer *handles* rather than flattened contents.
//
// The second RUN line covers lower-llvm-ext, which has to recover each clone's
// memory space at the free -- the free operates on a handle loaded out of the
// buffer, so the space is only reachable by tracing back to the stores.

module {
  func.func @main(%p: !llvm.ptr) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = arith.constant 0.0 : f32

    %sz = llvm.mlir.constant(40 : i64) : i64
    llvm_ext.ptr_size_hint %p, %sz : !llvm.ptr, i64

    %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%it = %init) -> (f32) {
      %v = llvm.load %p : !llvm.ptr -> f32
      %m = arith.mulf %v, %it : f32
      llvm.store %m, %p : f32, !llvm.ptr
      scf.yield %m : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 4 : i64}

    return %0 : f32
  }
}

// CHECK-LABEL: func.func @main(
// A budget-sized buffer of pointer handles, plus one clone per slot allocated
// up front. The size on every alloc/memcpy is the hinted extent.
// CHECK:         %[[SZ:.+]] = llvm.mlir.constant(40 : i64) : i64
// CHECK-DAG:     %[[SLOTS:.+]] = memref.alloc() : memref<4x!llvm.ptr>
// CHECK:         %[[C0:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm_ext.memcpy %[[C0]], %arg0, %[[SZ]]
// CHECK-NEXT:    memref.store %[[C0]], %[[SLOTS]][%c0] : memref<4x!llvm.ptr>
// CHECK:         %[[C1:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm_ext.memcpy %[[C1]], %arg0, %[[SZ]]
// CHECK-NEXT:    memref.store %[[C1]], %[[SLOTS]][%c1] : memref<4x!llvm.ptr>
// CHECK:         %[[C2:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr
// CHECK:         memref.store %[[C2]], %[[SLOTS]][%c2] : memref<4x!llvm.ptr>
// CHECK:         %[[C3:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr
// CHECK:         memref.store %[[C3]], %[[SLOTS]][%c3] : memref<4x!llvm.ptr>

// Forward: snapshot into slot %k with a copy, no new allocation.
// CHECK:         scf.for %[[K:.+]] = %c0 to %c4 step %c1
// CHECK:           %[[FWDSLOT:.+]] = memref.load %[[SLOTS]][%[[K]]] : memref<4x!llvm.ptr>
// CHECK-NEXT:      llvm_ext.memcpy %[[FWDSLOT]], %arg0, %[[SZ]]

// The working clone the reverse pass replays into, outside the reverse loop.
// CHECK:         %[[WORK:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm_ext.memcpy %[[WORK]], %arg0, %[[SZ]]

// Reverse: the slot index is the stack-pointer iter arg minus one, never a
// function of the reverse induction variable.
// CHECK:         scf.for %{{.+}} = %c0 to %c10 step %c1 iter_args(%[[SP:.+]] = %c4
// CHECK-NEXT:      %[[CAPO:.+]] = arith.subi %[[SP]], %c1 : index
// CHECK:           memref.load %{{.+}}[%[[CAPO]]] : memref<4xf32>
// CHECK:           memref.load %{{.+}}[%[[CAPO]]] : memref<4xindex>
// CHECK:           %[[REVSLOT:.+]] = memref.load %[[SLOTS]][%[[CAPO]]] : memref<4x!llvm.ptr>
// CHECK-NEXT:      llvm_ext.memcpy %[[WORK]], %[[REVSLOT]], %[[SZ]]

// The remat re-places a checkpoint; the pointer snapshot moves with it.
// CHECK:           scf.while ({{.*}}%[[ACAPO:.+]] = %[[CAPO]]
// CHECK:             %[[ACSLOT:.+]] = memref.load %[[SLOTS]][%[[ACAPO]]] : memref<4x!llvm.ptr>
// CHECK-NEXT:        llvm_ext.memcpy %[[ACSLOT]], %[[WORK]], %[[SZ]]

// Teardown: working clone, then every slot's clone, then the handle buffer.
// CHECK:         llvm_ext.free %[[WORK]]
// CHECK-NEXT:    scf.for %[[J:.+]] = %c0 to %c4 step %c1 {
// CHECK-NEXT:      %[[FREESLOT:.+]] = memref.load %[[SLOTS]][%[[J]]] : memref<4x!llvm.ptr>
// CHECK-NEXT:      llvm_ext.free %[[FREESLOT]]
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[SLOTS]] : memref<4x!llvm.ptr>

// After lower-llvm-ext the host allocator is used throughout (memory space 0),
// including for the frees of handles loaded back out of the buffer. Recovering
// the space there needs the store-side trace; without it the pass would fail.
// LOWER-DAG:   llvm.func @malloc(i64) -> !llvm.ptr
// LOWER-DAG:   llvm.func @free(!llvm.ptr)
// LOWER-LABEL: func.func @main(
// LOWER:         llvm.call @malloc
// LOWER:         "llvm.intr.memcpy"
// LOWER:         memref.load
// LOWER:         llvm.call @free
// LOWER-NOT:     llvm_ext.
