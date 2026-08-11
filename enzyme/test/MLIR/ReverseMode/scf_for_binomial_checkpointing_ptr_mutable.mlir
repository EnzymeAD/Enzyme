// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s
// RUN: %eopt %s --allow-unregistered-dialect --enzyme-wrap="infn=main outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --lower-llvm-ext --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math | FileCheck %s --check-prefix=LOWER

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
// CHECK:         %[[TOTAL:.+]] = arith.constant 160 : i64
// CHECK:         %[[SZ:.+]] = llvm.mlir.constant(40 : i64) : i64
// CHECK:         %[[SLOTS:.+]] = llvm_ext.alloc %[[TOTAL]] : (i64) -> !llvm.ptr

// CHECK:         scf.for %[[K:.+]] = %c0 to %c4 step %c1
// CHECK:           %[[KI:.+]] = arith.index_cast %[[K]] : index to i64
// CHECK-NEXT:      %[[OFF:.+]] = arith.muli %[[KI]], %[[SZ]] : i64
// CHECK-NEXT:      %[[FWDSLOT:.+]] = llvm.getelementptr %[[SLOTS]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:      llvm_ext.memcpy %[[FWDSLOT]], %arg0, %[[SZ]]

// CHECK:         %[[WORK:.+]] = llvm_ext.alloc %[[SZ]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm_ext.memcpy %[[WORK]], %arg0, %[[SZ]]

// CHECK:         scf.for %{{.+}} = %c0 to %c10 step %c1 iter_args(%[[SP:.+]] = %c4
// CHECK-NEXT:      %[[CAPO:.+]] = arith.subi %[[SP]], %c1 : index
// CHECK:           memref.load %{{.+}}[%[[CAPO]]] : memref<4xf32>
// CHECK:           memref.load %{{.+}}[%[[CAPO]]] : memref<4xindex>
// CHECK:           %[[CAPOI:.+]] = arith.index_cast %[[CAPO]] : index to i64
// CHECK-NEXT:      %[[REVOFF:.+]] = arith.muli %[[CAPOI]], %[[SZ]] : i64
// CHECK-NEXT:      %[[REVSLOT:.+]] = llvm.getelementptr %[[SLOTS]][%[[REVOFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:      llvm_ext.memcpy %[[WORK]], %[[REVSLOT]], %[[SZ]]

// CHECK:           scf.while ({{.*}}%[[ACAPO:.+]] = %[[CAPO]]
// CHECK:             %[[ACAPOI:.+]] = arith.index_cast %[[ACAPO]] : index to i64
// CHECK-NEXT:        %[[ACOFF:.+]] = arith.muli %[[ACAPOI]], %[[SZ]] : i64
// CHECK-NEXT:        %[[ACSLOT:.+]] = llvm.getelementptr %[[SLOTS]][%[[ACOFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK-NEXT:        llvm_ext.memcpy %[[ACSLOT]], %[[WORK]], %[[SZ]]

// CHECK:         llvm_ext.free %[[WORK]]
// CHECK-NEXT:    llvm_ext.free %[[SLOTS]]

// LOWER-DAG:   llvm.func @malloc(i64) -> !llvm.ptr
// LOWER-DAG:   llvm.func @free(!llvm.ptr)
// LOWER-LABEL: func.func @main(
// LOWER:         llvm.call @malloc
// LOWER:         "enzymexla.memcpy"
// LOWER:         llvm.call @free
// LOWER-NOT:     llvm_ext.
