// RUN: %eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

// Binomial checkpointing of a loop that reads a bare !llvm.ptr through a
// getelementptr derived from the induction variable, and mutates that same
// memory as it goes. What this pins beyond
// scf_for_binomial_checkpointing_ptr_mutable (which reads the pointer directly,
// and hints its extent with an llvm.mlir.constant rather than an arith one):
//
//  - the replayed segment re-derives the GEP against the per-checkpoint clone,
//    not against the caller's pointer, so a segment replayed after the buffer
//    has been overwritten still reads what the primal read there;
//  - the adjoint of the load lands in the shadow at the same derived offset.
//
// As in that test, the clone slots hold pointer *handles* -- one clone
// allocated per slot up front -- since a pointer's extent is not in its type.

module {
  func.func @reduce_sum(%buf: !llvm.ptr) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = arith.constant 0.0 : f64

    %sz = arith.constant 10 : i64
    llvm_ext.ptr_size_hint %buf, %sz : !llvm.ptr, i64

    %sum = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> (f64) {
      %idx = arith.index_cast %i : index to i64
      %derived = llvm.getelementptr inbounds %buf[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %val = llvm.load %derived : !llvm.ptr -> f64

      %new_acc = arith.addf %acc, %val : f64
      llvm.store %new_acc, %buf : f64, !llvm.ptr
      scf.yield %new_acc : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period=4,
       enzyme.disable_mincut=true}

    return %sum : f64
  }
}

// CHECK:  func.func @reduce_sum(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: f64) {
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c40_i64 = arith.constant 40 : i64
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c10_i64 = arith.constant 10 : i64
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    llvm_ext.ptr_size_hint %arg0, %c10_i64 : !llvm.ptr, i64
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf64>
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<4xindex>
// CHECK-NEXT:    %0 = llvm_ext.alloc %c40_i64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %1:2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %c0, %arg5 = %cst) -> (index, f64) {
// CHECK-NEXT:      memref.store %arg5, %alloc[%arg3] : memref<4xf64>
// CHECK-NEXT:      memref.store %arg4, %alloc_0[%arg3] : memref<4xindex>
// CHECK-NEXT:      %5 = arith.subi %c10, %arg4 : index
// CHECK-NEXT:      %6 = arith.subi %c4, %arg3 : index
// CHECK-NEXT:      %7 = arith.minui %6, %5 : index
// CHECK-NEXT:      %8 = enzyme.binomial_progress %5, %7 : index
// CHECK-NEXT:      %9 = arith.index_cast %arg3 : index to i64
// CHECK-NEXT:      %10 = arith.muli %9, %c10_i64 : i64
// CHECK-NEXT:      %11 = llvm.getelementptr %0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:      llvm_ext.memcpy %11, %arg0, %c10_i64 : !llvm.ptr, !llvm.ptr, i64
// CHECK-NEXT:      %12 = scf.for %arg6 = %c0 to %8 step %c1 iter_args(%arg7 = %arg5) -> (f64) {
// CHECK-NEXT:        %14 = arith.addi %arg4, %arg6 : index
// CHECK-NEXT:        %15 = arith.index_cast %14 : index to i64
// CHECK-NEXT:        %16 = llvm.getelementptr %arg0[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:        %17 = llvm.load %16 : !llvm.ptr -> f64
// CHECK-NEXT:        %18 = arith.addf %arg7, %17 : f64
// CHECK-NEXT:        llvm.store %18, %arg0 : f64, !llvm.ptr
// CHECK-NEXT:        scf.yield %18 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      %13 = arith.addi %arg4, %8 : index
// CHECK-NEXT:      scf.yield %13, %12 : index, f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    %2 = arith.addf %arg2, %cst fastmath<fast> : f64
// CHECK-NEXT:    %3 = llvm_ext.alloc %c10_i64 : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm_ext.memcpy %3, %arg0, %c10_i64 : !llvm.ptr, !llvm.ptr, i64
// CHECK-NEXT:    %4:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %c4, %arg5 = %2) -> (index, f64) {
// CHECK-NEXT:      %5 = arith.subi %arg4, %c1 : index
// CHECK-NEXT:      %6 = arith.subi %c10, %arg3 : index
// CHECK-NEXT:      %7 = memref.load %alloc[%5] : memref<4xf64>
// CHECK-NEXT:      %8 = memref.load %alloc_0[%5] : memref<4xindex>
// CHECK-NEXT:      %9 = arith.index_cast %5 : index to i64
// CHECK-NEXT:      %10 = arith.muli %9, %c10_i64 : i64
// CHECK-NEXT:      %11 = llvm.getelementptr %0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:      llvm_ext.memcpy %3, %11, %c10_i64 : !llvm.ptr, !llvm.ptr, i64
// CHECK-NEXT:      %12:3 = scf.while (%arg6 = %8, %arg7 = %5, %arg8 = %7) : (index, index, f64) -> (index, index, f64) {
// CHECK-NEXT:        %26 = arith.addi %arg6, %c1 : index
// CHECK-NEXT:        %27 = arith.cmpi slt, %26, %6 : index
// CHECK-NEXT:        scf.condition(%27) %arg6, %arg7, %arg8 : index, index, f64
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg6: index, %arg7: index, %arg8: f64):
// CHECK-NEXT:        %26 = arith.subi %6, %arg6 : index
// CHECK-NEXT:        %27 = arith.subi %c4, %arg7 : index
// CHECK-NEXT:        %28 = arith.minui %27, %26 : index
// CHECK-NEXT:        %29 = enzyme.binomial_progress %26, %28 : index
// CHECK-NEXT:        memref.store %arg8, %alloc[%arg7] : memref<4xf64>
// CHECK-NEXT:        memref.store %arg6, %alloc_0[%arg7] : memref<4xindex>
// CHECK-NEXT:        %30 = arith.index_cast %arg7 : index to i64
// CHECK-NEXT:        %31 = arith.muli %30, %c10_i64 : i64
// CHECK-NEXT:        %32 = llvm.getelementptr %0[%31] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:        llvm_ext.memcpy %32, %3, %c10_i64 : !llvm.ptr, !llvm.ptr, i64
// CHECK-NEXT:        %33 = arith.addi %arg6, %29 : index
// CHECK-NEXT:        %34 = arith.cmpi eq, %33, %6 : index
// CHECK-NEXT:        %35 = arith.subi %33, %c1 : index
// CHECK-NEXT:        %36 = arith.select %34, %35, %33 : index
// CHECK-NEXT:        %37 = scf.for %arg9 = %arg6 to %36 step %c1 iter_args(%arg10 = %arg8) -> (f64) {
// CHECK-NEXT:          %39 = arith.index_cast %arg9 : index to i64
// CHECK-NEXT:          %40 = llvm.getelementptr inbounds %3[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:          %41 = llvm.load %40 : !llvm.ptr -> f64
// CHECK-NEXT:          %42 = arith.addf %arg10, %41 : f64
// CHECK-NEXT:          llvm.store %42, %3 : f64, !llvm.ptr
// CHECK-NEXT:          scf.yield %42 : f64
// CHECK-NEXT:        } {enzyme.disable_mincut = true}
// CHECK-NEXT:        %38 = arith.addi %arg7, %c1 : index
// CHECK-NEXT:        scf.yield %33, %38, %37 : index, index, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %13 = arith.subi %c9, %arg3 : index
// CHECK-NEXT:      %14 = arith.index_cast %13 : index to i64
// CHECK-NEXT:      %15 = llvm.getelementptr inbounds %arg1[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:      %16 = llvm.getelementptr inbounds %3[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:      %17 = llvm.load %16 : !llvm.ptr -> f64
// CHECK-NEXT:      %18 = arith.addf %12#2, %17 : f64
// CHECK-NEXT:      llvm.store %18, %3 : f64, !llvm.ptr
// CHECK-NEXT:      %19 = arith.addf %arg5, %cst fastmath<fast> : f64
// CHECK-NEXT:      %20 = llvm.load %arg1 : !llvm.ptr -> f64
// CHECK-NEXT:      %21 = arith.addf %19, %20 fastmath<fast> : f64
// CHECK-NEXT:      llvm.store %cst, %arg1 : f64, !llvm.ptr
// CHECK-NEXT:      %22 = arith.addf %21, %cst fastmath<fast> : f64
// CHECK-NEXT:      %23 = arith.addf %21, %cst fastmath<fast> : f64
// CHECK-NEXT:      %24 = llvm.load %15 : !llvm.ptr -> f64
// CHECK-NEXT:      %25 = arith.addf %24, %23 fastmath<fast> : f64
// CHECK-NEXT:      llvm.store %25, %15 : f64, !llvm.ptr
// CHECK-NEXT:      scf.yield %12#1, %22 : index, f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf64>
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<4xindex>
// CHECK-NEXT:    llvm_ext.free %3 : !llvm.ptr
// CHECK-NEXT:    llvm_ext.free %0 : !llvm.ptr
// CHECK-NEXT:    return
// CHECK-NEXT:  }
