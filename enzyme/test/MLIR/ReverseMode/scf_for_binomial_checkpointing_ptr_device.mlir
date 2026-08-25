// RUN: %eopt %s --raise-llvm-ext --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s
// RUN: %eopt %s --allow-unregistered-dialect --raise-llvm-ext --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --lower-llvm-ext --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s --check-prefix=LOWER

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
// CHECK:         %[[TOTAL:.+]] = arith.constant 160 : i64
// CHECK:         %[[SZ:.+]] = llvm.mlir.constant(40 : i64) : i64
// CHECK:         %[[DEV:.+]] = llvm.addrspacecast %arg0 : !llvm.ptr to !llvm.ptr<1>
// CHECK:         llvm_ext.ptr_size_hint %[[DEV]], %[[SZ]] : !llvm.ptr<1>, i64
// CHECK:         %[[SLOTS:.+]] = llvm_ext.alloc %[[TOTAL]] : (i64) -> !llvm.ptr<1>

// CHECK:         %[[FWDSLOT:.+]] = llvm.getelementptr %[[SLOTS]][%{{.+}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    llvm_ext.memcpy %[[FWDSLOT]], %{{.+}}, %[[SZ]] : !llvm.ptr<1>, !llvm.ptr<1>, i64
// CHECK:         %[[WORK:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr<1>

// CHECK:         llvm_ext.free %[[WORK]] : !llvm.ptr<1>
// CHECK-NEXT:    llvm_ext.free %[[SLOTS]] : !llvm.ptr<1>

// LOWER-LABEL: func.func @main(
// LOWER-NOT:     llvm.call @malloc
// LOWER-NOT:     llvm.call @free
// LOWER:         gpu.alloc
// LOWER:         "enzymexla.memcpy"
// LOWER:         gpu.dealloc
// LOWER-NOT:     llvm_ext.
