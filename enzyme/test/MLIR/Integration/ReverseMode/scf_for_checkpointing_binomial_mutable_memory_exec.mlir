// REQUIRES: mlir-runner
//
// Same numeric check as scf_for_checkpointing_mutable_memory_exec.mlir, but
// for binomial (Revolve) checkpointing.
//
// RUN: (%eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops --canonicalize --lower-enzyme-binomial-progress --convert-enzyme-to-memref | tail -n +2 | head -n -2; cat %S/Inputs/exec_main_10xf64.mlir.inc) | %mlir-opt --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-arith-to-llvm --finalize-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | %mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

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
  } {enzyme.enable_checkpointing = true,
     enzyme.binomial_checkpointing,
     enzyme.checkpoint_period=4,
     enzyme.disable_mincut=true}

  return %sum : f64
}

// CHECK: Unranked Memref base@ = {{0x[0-9a-f]*}} rank = 1 offset = 0 sizes = [10] strides = [1] data =
// CHECK-NEXT: [55, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// CHECK: Unranked Memref base@ = {{0x[0-9a-f]*}} rank = 1 offset = 0 sizes = [10] strides = [1] data =
// CHECK-NEXT: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
