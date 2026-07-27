// REQUIRES: mlir-runner
//
// Executes the reverse-mode derivative of a scf.for loop whose body both
// reads and writes the same memref (see ../../ReverseMode/scf_for_mutable_memory.mlir
// for the FileCheck-only IR-shape test), then checks the actual numerical
// result: the primal `buf` should end with the same final mutation trace as
// running the loop directly, and the gradient of `sum` w.r.t. `buf` should be
// all-ones (since the mutation of buf[0] is never re-read, `sum` is just a
// sum of the original buf entries).
//
// RUN: (%eopt %s --enzyme-wrap="infn=reduce_sum outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --enzyme-simplify-math --remove-unnecessary-enzyme-ops | tail -n +2 | head -n -2; cat %S/Inputs/exec_main_10xf64.mlir.inc) | %mlir-opt --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-arith-to-llvm --finalize-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | %mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

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

// CHECK: Unranked Memref base@ = {{0x[0-9a-f]*}} rank = 1 offset = 0 sizes = [10] strides = [1] data =
// CHECK-NEXT: [55, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// CHECK: Unranked Memref base@ = {{0x[0-9a-f]*}} rank = 1 offset = 0 sizes = [10] strides = [1] data =
// CHECK-NEXT: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
