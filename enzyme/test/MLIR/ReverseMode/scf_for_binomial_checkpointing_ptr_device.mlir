// RUN: %eopt %s --raise-llvm-ext --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s
// RUN: %eopt %s --allow-unregistered-dialect --raise-llvm-ext --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --lower-llvm-ext --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s --check-prefix=LOWER

// The shape a CUDA program hits: the checkpointed loop mutates memory reached
// through a pointer that really lives on the device, but whose C type says
// nothing about that (cudaMalloc hands back a plain `float *`). The third
// argument of __enzyme_ptr_size_hint annotates the memory space; raise-llvm-ext
// turns it into an addrspacecast, and cloning follows the cast, so the
// snapshots are device allocations rather than host ones.
//
// The second RUN line covers lower-llvm-ext; the copies it emits are enzymexla
// ops, which live outside this repository, hence --allow-unregistered-dialect.

module {
  llvm.func @__enzyme_ptr_size_hint(!llvm.ptr, i64, i64)

  func.func @main(%p: !llvm.ptr) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = arith.constant 0.0 : f32

    %sz = llvm.mlir.constant(40 : i64) : i64
    %space = llvm.mlir.constant(1 : i64) : i64
    llvm.call @__enzyme_ptr_size_hint(%p, %sz, %space) : (!llvm.ptr, i64, i64) -> ()

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
// The annotated space reaches every allocation, copy and free: the handles are
// device pointers, so the buffer holding them is one of those too.
// CHECK:         %[[SZ:.+]] = llvm.mlir.constant(40 : i64) : i64
// CHECK:         %[[DEV:.+]] = llvm.addrspacecast %arg0 : !llvm.ptr to !llvm.ptr<1>
// CHECK:         llvm_ext.ptr_size_hint %[[DEV]], %[[SZ]] : !llvm.ptr<1>, i64
// CHECK:         %[[C0:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr<1>
// CHECK-NEXT:    llvm_ext.memcpy %[[C0]], %[[DEV]], %[[SZ]] : !llvm.ptr<1>, !llvm.ptr<1>, i64
// CHECK:         %[[SLOTS:.+]] = memref.alloc() : memref<4x!llvm.ptr<1>>
// CHECK-NEXT:    memref.store %[[C0]], %[[SLOTS]][%c0] : memref<4x!llvm.ptr<1>>

// Snapshot into slot %k, and the working clone the reverse pass replays into.
// CHECK:         %[[FWDSLOT:.+]] = memref.load %[[SLOTS]][%{{.+}}] : memref<4x!llvm.ptr<1>>
// CHECK-NEXT:    llvm_ext.memcpy %[[FWDSLOT]], %{{.+}}, %[[SZ]] : !llvm.ptr<1>, !llvm.ptr<1>, i64
// CHECK:         %[[WORK:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr<1>

// Teardown frees device handles, never host ones.
// CHECK:         llvm_ext.free %[[WORK]] : !llvm.ptr<1>
// CHECK:           %[[FREESLOT:.+]] = memref.load %[[SLOTS]][%{{.+}}] : memref<4x!llvm.ptr<1>>
// CHECK-NEXT:      llvm_ext.free %[[FREESLOT]] : !llvm.ptr<1>
// CHECK:         memref.dealloc %[[SLOTS]] : memref<4x!llvm.ptr<1>>

// lower-llvm-ext routes all of it through the GPU runtime: no malloc/free of
// what is device memory, which is the bug the annotation exists to avoid.
// LOWER-LABEL: func.func @main(
// LOWER-NOT:     llvm.call @malloc
// LOWER-NOT:     llvm.call @free
// LOWER:         gpu.alloc
// LOWER:         "enzymexla.memcpy"
// LOWER:         gpu.dealloc
// LOWER-NOT:     llvm_ext.
