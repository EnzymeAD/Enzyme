// REQUIRES: mlir-runner
//
// A loop checkpointed with the schedule of enzyme/checkpoint.h at run time
// that reads and writes memory from outside it: the snapshots have to hold a
// clone of it, and the reverse pass has to replay into a working clone of
// its own. The final memory and the gradient must be those without
// checkpointing and with the compiled binomial schedule.
//
// RUN: %host_cc -shared -fPIC -DENZYME_CHECKPOINT_RUNTIME -x c %enzyme_include/enzyme/checkpoint.h -o %t.so
// RUN: (%eopt %s --enzyme-wrap="infn=f_ref outfn=g_ref argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=f_bin outfn=g_bin argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=f_rt_bin outfn=g_rt_bin argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=f_rt_per outfn=g_rt_per argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --lower-enzyme-binomial-progress --enzyme-simplify-math --canonicalize --convert-enzyme-to-memref | head -n -2; cat %S/Inputs/runtime_checkpointing_mutable_driver.mlir.inc) | %mlir-opt --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-math-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | %mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%t.so | FileCheck %s

module {
  func.func @f_ref(%m: memref<2xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %k1 = arith.constant 0.9 : f64
    %k2 = arith.constant 0.3 : f64
    %k3 = arith.constant 0.1 : f64
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %zero) -> (f64) {
      %t = memref.load %m[%c0] : memref<2xf64>
      %u = memref.load %m[%c1] : memref<2xf64>
      %st = math.sin %t : f64
      %a = arith.mulf %st, %k1 : f64
      %b = arith.mulf %u, %k2 : f64
      %t2 = arith.addf %a, %b : f64
      memref.store %t2, %m[%c0] : memref<2xf64>
      %ct = math.cos %t : f64
      %u1 = arith.mulf %u, %ct : f64
      %u2 = arith.addf %u1, %k3 : f64
      memref.store %u2, %m[%c1] : memref<2xf64>
      %p = arith.mulf %t2, %u : f64
      %acc2 = arith.addf %acc, %p : f64
      scf.yield %acc2 : f64
    } {enzyme.disable_mincut = true}
    return %r : f64
  }
  func.func @f_bin(%m: memref<2xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %k1 = arith.constant 0.9 : f64
    %k2 = arith.constant 0.3 : f64
    %k3 = arith.constant 0.1 : f64
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %zero) -> (f64) {
      %t = memref.load %m[%c0] : memref<2xf64>
      %u = memref.load %m[%c1] : memref<2xf64>
      %st = math.sin %t : f64
      %a = arith.mulf %st, %k1 : f64
      %b = arith.mulf %u, %k2 : f64
      %t2 = arith.addf %a, %b : f64
      memref.store %t2, %m[%c0] : memref<2xf64>
      %ct = math.cos %t : f64
      %u1 = arith.mulf %u, %ct : f64
      %u2 = arith.addf %u1, %k3 : f64
      memref.store %u2, %m[%c1] : memref<2xf64>
      %p = arith.mulf %t2, %u : f64
      %acc2 = arith.addf %acc, %p : f64
      scf.yield %acc2 : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64,
       enzyme.disable_mincut = true}
    return %r : f64
  }
  func.func @f_rt_bin(%m: memref<2xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %k1 = arith.constant 0.9 : f64
    %k2 = arith.constant 0.3 : f64
    %k3 = arith.constant 0.1 : f64
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %zero) -> (f64) {
      %t = memref.load %m[%c0] : memref<2xf64>
      %u = memref.load %m[%c1] : memref<2xf64>
      %st = math.sin %t : f64
      %a = arith.mulf %st, %k1 : f64
      %b = arith.mulf %u, %k2 : f64
      %t2 = arith.addf %a, %b : f64
      memref.store %t2, %m[%c0] : memref<2xf64>
      %ct = math.cos %t : f64
      %u1 = arith.mulf %u, %ct : f64
      %u2 = arith.addf %u1, %k3 : f64
      memref.store %u2, %m[%c1] : memref<2xf64>
      %p = arith.mulf %t2, %u : f64
      %acc2 = arith.addf %acc, %p : f64
      scf.yield %acc2 : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64,
       enzyme.checkpoint_runtime, enzyme.disable_mincut = true}
    return %r : f64
  }
  func.func @f_rt_per(%m: memref<2xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %k1 = arith.constant 0.9 : f64
    %k2 = arith.constant 0.3 : f64
    %k3 = arith.constant 0.1 : f64
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %zero) -> (f64) {
      %t = memref.load %m[%c0] : memref<2xf64>
      %u = memref.load %m[%c1] : memref<2xf64>
      %st = math.sin %t : f64
      %a = arith.mulf %st, %k1 : f64
      %b = arith.mulf %u, %k2 : f64
      %t2 = arith.addf %a, %b : f64
      memref.store %t2, %m[%c0] : memref<2xf64>
      %ct = math.cos %t : f64
      %u1 = arith.mulf %u, %ct : f64
      %u2 = arith.addf %u1, %k3 : f64
      memref.store %u2, %m[%c1] : memref<2xf64>
      %p = arith.mulf %t2, %u : f64
      %acc2 = arith.addf %acc, %p : f64
      scf.yield %acc2 : f64
    } {enzyme.enable_checkpointing = true, enzyme.checkpoint_runtime,
       enzyme.disable_mincut = true}
    return %r : f64
  }
}

// One line per variant, in the order above: the memory the loop leaves, and
// the gradient with respect to its initial contents.
// CHECK: [[M0:[-+.e0-9]+]], [[M1:[-+.e0-9]+]], [[D0:[-+.e0-9]+]], [[D1:[-+.e0-9]+]]
// CHECK-NEXT: [[M0]], [[M1]], [[D0]], [[D1]]
// CHECK-NEXT: [[M0]], [[M1]], [[D0]], [[D1]]
// CHECK-NEXT: [[M0]], [[M1]], [[D0]], [[D1]]
