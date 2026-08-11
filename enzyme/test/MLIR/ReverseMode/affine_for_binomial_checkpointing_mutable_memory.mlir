// RUN: %eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

module {
  func.func @reduce_sum(%buf: memref<10xf64>) -> f64 {
    %init = arith.constant 0.0 : f64

    %sum = affine.for %i = 0 to 10 iter_args(%acc = %init) -> (f64) {
      %val = affine.load %buf[%i] : memref<10xf64>
      %new_acc = arith.addf %acc, %val : f64
      affine.store %new_acc, %buf[0] : memref<10xf64>
      affine.yield %new_acc : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period=4,
       enzyme.disable_mincut=true}

    return %sum : f64
  }
}

// affine.load/affine.store indexing through the checkpointed loop's own
// induction variable (%i) must not be cloned verbatim into the checkpointing
// scaffold's recompute/remat loops: those are always scf.for regardless of
// the differentiated loop's dialect, so %i's replacement there is a plain
// (non-affine-dimension) index Value, and affine.load/store would reject it
// at that point ("operand cannot be used as a dimension id"). This is the
// regression the test guards: LoopCheckpointing's cloneOp hook must expand
// such ops into memref.load/memref.store (via affine::expandAffineMap)
// instead of cloning them as-is -- see %11/%30 below, both memref.load, and
// %arg0/%alloc_6 (their memref operand) both being plain, generically-typed
// Values rather than valid affine dims/symbols.
//
// Otherwise this mirrors scf_for_checkpointing_binomial_mutable_memory.mlir
// line for line: the mutable outside reference (%arg0) gets one clone per
// checkpoint slot (%alloc_1 through %alloc_4) and a buffer of handles to them
// (%alloc_5), all up front so taking a checkpoint is a plain copy into an
// existing allocation, plus one working clone (%alloc_6) that the reverse
// pass replays into.

// CHECK:  func.func @reduce_sum(%arg0: memref<10xf64>, %arg1: memref<10xf64>, %arg2: f64) {
// CHECK-NEXT:    %c9 = arith.constant 9 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c10 = arith.constant 10 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %alloc = memref.alloc() : memref<4xf64>
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<4xindex>
// CHECK-NEXT:    %alloc_1 = memref.alloc() : memref<10xf64>
// CHECK-NEXT:    memref.copy %arg0, %alloc_1 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:    %alloc_2 = memref.alloc() : memref<10xf64>
// CHECK-NEXT:    memref.copy %arg0, %alloc_2 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:    %alloc_3 = memref.alloc() : memref<10xf64>
// CHECK-NEXT:    memref.copy %arg0, %alloc_3 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:    %alloc_4 = memref.alloc() : memref<10xf64>
// CHECK-NEXT:    memref.copy %arg0, %alloc_4 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:    %alloc_5 = memref.alloc() : memref<4xmemref<10xf64>>
// CHECK-NEXT:    memref.store %alloc_1, %alloc_5[%c0] : memref<4xmemref<10xf64>>
// CHECK-NEXT:    memref.store %alloc_2, %alloc_5[%c1] : memref<4xmemref<10xf64>>
// CHECK-NEXT:    memref.store %alloc_3, %alloc_5[%c2] : memref<4xmemref<10xf64>>
// CHECK-NEXT:    memref.store %alloc_4, %alloc_5[%c3] : memref<4xmemref<10xf64>>
// CHECK-NEXT:    %0:2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %c0, %arg5 = %cst) -> (index, f64) {
// CHECK-NEXT:      memref.store %arg5, %alloc[%arg3] : memref<4xf64>
// CHECK-NEXT:      memref.store %arg4, %alloc_0[%arg3] : memref<4xindex>
// CHECK-NEXT:      %3 = arith.subi %c10, %arg4 : index
// CHECK-NEXT:      %4 = arith.subi %c4, %arg3 : index
// CHECK-NEXT:      %5 = arith.minui %4, %3 : index
// CHECK-NEXT:      %6 = enzyme.binomial_progress %3, %5 : index
// CHECK-NEXT:      %7 = memref.load %alloc_5[%arg3] : memref<4xmemref<10xf64>>
// CHECK-NEXT:      memref.copy %arg0, %7 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:      %8 = scf.for %arg6 = %c0 to %6 step %c1 iter_args(%arg7 = %arg5) -> (f64) {
// CHECK-NEXT:        %10 = arith.addi %arg4, %arg6 : index
// CHECK-NEXT:        %11 = memref.load %arg0[%10] : memref<10xf64>
// CHECK-NEXT:        %12 = arith.addf %arg7, %11 : f64
// CHECK-NEXT:        memref.store %12, %arg0[%c0] : memref<10xf64>
// CHECK-NEXT:        scf.yield %12 : f64
// CHECK-NEXT:      } {enzyme.disable_mincut = true}
// CHECK-NEXT:      %9 = arith.addi %arg4, %6 : index
// CHECK-NEXT:      scf.yield %9, %8 : index, f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    %1 = arith.addf %arg2, %cst fastmath<fast> : f64
// CHECK-NEXT:    %alloc_6 = memref.alloc() : memref<10xf64>
// CHECK-NEXT:    memref.copy %arg0, %alloc_6 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:    %2:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %c4, %arg5 = %1) -> (index, f64) {
// CHECK-NEXT:      %3 = arith.subi %arg4, %c1 : index
// CHECK-NEXT:      %4 = arith.subi %c10, %arg3 : index
// CHECK-NEXT:      %5 = memref.load %alloc[%3] : memref<4xf64>
// CHECK-NEXT:      %6 = memref.load %alloc_0[%3] : memref<4xindex>
// CHECK-NEXT:      %7 = memref.load %alloc_5[%3] : memref<4xmemref<10xf64>>
// CHECK-NEXT:      memref.copy %7, %alloc_6 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:      %8:3 = scf.while (%arg6 = %6, %arg7 = %3, %arg8 = %5) : (index, index, f64) -> (index, index, f64) {
// CHECK-NEXT:        %19 = arith.addi %arg6, %c1 : index
// CHECK-NEXT:        %20 = arith.cmpi slt, %19, %4 : index
// CHECK-NEXT:        scf.condition(%20) %arg6, %arg7, %arg8 : index, index, f64
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg6: index, %arg7: index, %arg8: f64):
// CHECK-NEXT:        %19 = arith.subi %4, %arg6 : index
// CHECK-NEXT:        %20 = arith.subi %c4, %arg7 : index
// CHECK-NEXT:        %21 = arith.minui %20, %19 : index
// CHECK-NEXT:        %22 = enzyme.binomial_progress %19, %21 : index
// CHECK-NEXT:        memref.store %arg8, %alloc[%arg7] : memref<4xf64>
// CHECK-NEXT:        memref.store %arg6, %alloc_0[%arg7] : memref<4xindex>
// CHECK-NEXT:        %23 = memref.load %alloc_5[%arg7] : memref<4xmemref<10xf64>>
// CHECK-NEXT:        memref.copy %alloc_6, %23 : memref<10xf64> to memref<10xf64>
// CHECK-NEXT:        %24 = arith.addi %arg6, %22 : index
// CHECK-NEXT:        %25 = arith.cmpi eq, %24, %4 : index
// CHECK-NEXT:        %26 = arith.subi %24, %c1 : index
// CHECK-NEXT:        %27 = arith.select %25, %26, %24 : index
// CHECK-NEXT:        %28 = scf.for %arg9 = %arg6 to %27 step %c1 iter_args(%arg10 = %arg8) -> (f64) {
// CHECK-NEXT:          %30 = memref.load %alloc_6[%arg9] : memref<10xf64>
// CHECK-NEXT:          %31 = arith.addf %arg10, %30 : f64
// CHECK-NEXT:          memref.store %31, %alloc_6[%c0] : memref<10xf64>
// CHECK-NEXT:          scf.yield %31 : f64
// CHECK-NEXT:        } {enzyme.disable_mincut = true}
// CHECK-NEXT:        %29 = arith.addi %arg7, %c1 : index
// CHECK-NEXT:        scf.yield %24, %29, %28 : index, index, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = arith.subi %c9, %arg3 : index
// CHECK-NEXT:      %10 = memref.load %alloc_6[%9] : memref<10xf64>
// CHECK-NEXT:      %11 = arith.addf %8#2, %10 : f64
// CHECK-NEXT:      memref.store %11, %alloc_6[%c0] : memref<10xf64>
// CHECK-NEXT:      %12 = arith.addf %arg5, %cst fastmath<fast> : f64
// CHECK-NEXT:      %13 = affine.load %arg1[0] : memref<10xf64>
// CHECK-NEXT:      %14 = arith.addf %12, %13 fastmath<fast> : f64
// CHECK-NEXT:      affine.store %cst, %arg1[0] : memref<10xf64>
// CHECK-NEXT:      %15 = arith.addf %14, %cst fastmath<fast> : f64
// CHECK-NEXT:      %16 = arith.addf %14, %cst fastmath<fast> : f64
// CHECK-NEXT:      %17 = memref.load %arg1[%9] : memref<10xf64>
// CHECK-NEXT:      %18 = arith.addf %17, %16 fastmath<fast> : f64
// CHECK-NEXT:      memref.store %18, %arg1[%9] : memref<10xf64>
// CHECK-NEXT:      scf.yield %8#1, %15 : index, f64
// CHECK-NEXT:    } {enzyme.disable_mincut = true}
// CHECK-NEXT:    memref.dealloc %alloc : memref<4xf64>
// CHECK-NEXT:    memref.dealloc %alloc_0 : memref<4xindex>
// CHECK-NEXT:    memref.dealloc %alloc_6 : memref<10xf64>
// CHECK-NEXT:    scf.for %arg3 = %c0 to %c4 step %c1 {
// CHECK-NEXT:      %3 = memref.load %alloc_5[%arg3] : memref<4xmemref<10xf64>>
// CHECK-NEXT:      memref.dealloc %3 : memref<10xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %alloc_5 : memref<4xmemref<10xf64>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
