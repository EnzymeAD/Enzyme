// REQUIRES: mlir-runner
//
// Gradients of a loop checkpointed with the schedule of enzyme/checkpoint.h
// at run time (enzyme.checkpoint_runtime): binomial with budgets below and
// above some trip counts, periodic with a period and with the default for a
// dynamic trip count. Each must equal the gradient with the compiled
// binomial schedule, and match a central difference of the primal. The
// loop starts at 3 with step 2 and its body reads the induction variable,
// and the result is squared, so that the gradient also depends on the loop
// result the forward pass computes.
//
// RUN: %host_cc -shared -fPIC -DENZYME_CHECKPOINT_RUNTIME -x c %enzyme_include/enzyme/checkpoint.h -o %t.so
// RUN: (%eopt %s --enzyme-wrap="infn=f_bin outfn=g_bin argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=f_rt_bin2 outfn=g_rt_bin2 argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=f_rt_bin5 outfn=g_rt_bin5 argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=f_rt_per3 outfn=g_rt_per3 argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --enzyme-wrap="infn=f_rt_per outfn=g_rt_per argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --lower-enzyme-binomial-progress --enzyme-simplify-math --canonicalize --convert-enzyme-to-memref | head -n -2; cat %S/Inputs/runtime_checkpointing_driver.mlir.inc) | %mlir-opt --convert-scf-to-cf --expand-strided-metadata --lower-affine --convert-math-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | env ENZYME_CKPT_VERBOSE=1 %mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%t.so 2> %t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=STATS < %t.err

module {
  func.func @f(%x: f64, %n: index) -> f64 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %n2 = arith.muli %n, %c2 : index
    %ub = arith.addi %n2, %c3 : index
    %k = arith.constant 0.1 : f64
    %w = arith.constant 0.2 : f64
    %h = arith.constant 0.95 : f64
    %r = scf.for %i = %c3 to %ub step %c2 iter_args(%s = %x) -> (f64) {
      %ii = arith.index_cast %i : index to i64
      %fi = arith.sitofp %ii : i64 to f64
      %a = arith.mulf %fi, %k : f64
      %c = math.cos %a : f64
      %sn = math.sin %s : f64
      %t = arith.mulf %sn, %c : f64
      %t2 = arith.mulf %t, %w : f64
      %u = arith.mulf %s, %h : f64
      %v = arith.addf %t2, %u : f64
      scf.yield %v : f64
    }
    %rr = arith.mulf %r, %r : f64
    return %rr : f64
  }

  func.func @f_bin(%x: f64, %n: index) -> f64 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %n2 = arith.muli %n, %c2 : index
    %ub = arith.addi %n2, %c3 : index
    %k = arith.constant 0.1 : f64
    %w = arith.constant 0.2 : f64
    %h = arith.constant 0.95 : f64
    %r = scf.for %i = %c3 to %ub step %c2 iter_args(%s = %x) -> (f64) {
      %ii = arith.index_cast %i : index to i64
      %fi = arith.sitofp %ii : i64 to f64
      %a = arith.mulf %fi, %k : f64
      %c = math.cos %a : f64
      %sn = math.sin %s : f64
      %t = arith.mulf %sn, %c : f64
      %t2 = arith.mulf %t, %w : f64
      %u = arith.mulf %s, %h : f64
      %v = arith.addf %t2, %u : f64
      scf.yield %v : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64}
    %rr = arith.mulf %r, %r : f64
    return %rr : f64
  }

  func.func @f_rt_bin2(%x: f64, %n: index) -> f64 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %n2 = arith.muli %n, %c2 : index
    %ub = arith.addi %n2, %c3 : index
    %k = arith.constant 0.1 : f64
    %w = arith.constant 0.2 : f64
    %h = arith.constant 0.95 : f64
    %r = scf.for %i = %c3 to %ub step %c2 iter_args(%s = %x) -> (f64) {
      %ii = arith.index_cast %i : index to i64
      %fi = arith.sitofp %ii : i64 to f64
      %a = arith.mulf %fi, %k : f64
      %c = math.cos %a : f64
      %sn = math.sin %s : f64
      %t = arith.mulf %sn, %c : f64
      %t2 = arith.mulf %t, %w : f64
      %u = arith.mulf %s, %h : f64
      %v = arith.addf %t2, %u : f64
      scf.yield %v : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing, enzyme.checkpoint_period = 2 : i64,
       enzyme.checkpoint_runtime}
    %rr = arith.mulf %r, %r : f64
    return %rr : f64
  }

  func.func @f_rt_bin5(%x: f64, %n: index) -> f64 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %n2 = arith.muli %n, %c2 : index
    %ub = arith.addi %n2, %c3 : index
    %k = arith.constant 0.1 : f64
    %w = arith.constant 0.2 : f64
    %h = arith.constant 0.95 : f64
    %r = scf.for %i = %c3 to %ub step %c2 iter_args(%s = %x) -> (f64) {
      %ii = arith.index_cast %i : index to i64
      %fi = arith.sitofp %ii : i64 to f64
      %a = arith.mulf %fi, %k : f64
      %c = math.cos %a : f64
      %sn = math.sin %s : f64
      %t = arith.mulf %sn, %c : f64
      %t2 = arith.mulf %t, %w : f64
      %u = arith.mulf %s, %h : f64
      %v = arith.addf %t2, %u : f64
      scf.yield %v : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing, enzyme.checkpoint_period = 5 : i64,
       enzyme.checkpoint_runtime}
    %rr = arith.mulf %r, %r : f64
    return %rr : f64
  }

  func.func @f_rt_per3(%x: f64, %n: index) -> f64 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %n2 = arith.muli %n, %c2 : index
    %ub = arith.addi %n2, %c3 : index
    %k = arith.constant 0.1 : f64
    %w = arith.constant 0.2 : f64
    %h = arith.constant 0.95 : f64
    %r = scf.for %i = %c3 to %ub step %c2 iter_args(%s = %x) -> (f64) {
      %ii = arith.index_cast %i : index to i64
      %fi = arith.sitofp %ii : i64 to f64
      %a = arith.mulf %fi, %k : f64
      %c = math.cos %a : f64
      %sn = math.sin %s : f64
      %t = arith.mulf %sn, %c : f64
      %t2 = arith.mulf %t, %w : f64
      %u = arith.mulf %s, %h : f64
      %v = arith.addf %t2, %u : f64
      scf.yield %v : f64
    } {enzyme.enable_checkpointing = true,
       enzyme.checkpoint_period = 3 : i64, enzyme.checkpoint_runtime}
    %rr = arith.mulf %r, %r : f64
    return %rr : f64
  }

  func.func @f_rt_per(%x: f64, %n: index) -> f64 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %n2 = arith.muli %n, %c2 : index
    %ub = arith.addi %n2, %c3 : index
    %k = arith.constant 0.1 : f64
    %w = arith.constant 0.2 : f64
    %h = arith.constant 0.95 : f64
    %r = scf.for %i = %c3 to %ub step %c2 iter_args(%s = %x) -> (f64) {
      %ii = arith.index_cast %i : index to i64
      %fi = arith.sitofp %ii : i64 to f64
      %a = arith.mulf %fi, %k : f64
      %c = math.cos %a : f64
      %sn = math.sin %s : f64
      %t = arith.mulf %sn, %c : f64
      %t2 = arith.mulf %t, %w : f64
      %u = arith.mulf %s, %h : f64
      %v = arith.addf %t2, %u : f64
      scf.yield %v : f64
    } {enzyme.enable_checkpointing = true, enzyme.checkpoint_runtime}
    %rr = arith.mulf %r, %r : f64
    return %rr : f64
  }
}

// One line per trip count n: n, the gradient with the compiled schedule and
// with the runtime ones, and 1 if it matches the central difference.
// CHECK: 0, [[G:[-+.e0-9]+]], [[G]], [[G]], [[G]], [[G]], 1
// CHECK-NEXT: 1, [[G:[-+.e0-9]+]], [[G]], [[G]], [[G]], [[G]], 1
// CHECK-NEXT: 2, [[G:[-+.e0-9]+]], [[G]], [[G]], [[G]], [[G]], 1
// CHECK-NEXT: 7, [[G:[-+.e0-9]+]], [[G]], [[G]], [[G]], [[G]], 1
// CHECK-NEXT: 10, [[G:[-+.e0-9]+]], [[G]], [[G]], [[G]], [[G]], 1
// CHECK-NEXT: 37, [[G:[-+.e0-9]+]], [[G]], [[G]], [[G]], [[G]], 1

// What the runtime reports when each schedule ends, for n = 37: Revolve with
// 2 and 5 slots, periodic with 3 segments (3 + 13 - 1 slots) and with the
// default of floor(sqrt(37)) = 6 segments (6 + 7 - 1 slots).
// STATS-DAG: enzyme checkpoint: {{[0-9]+}} forward steps, 37 taped steps, 2 slots
// STATS-DAG: enzyme checkpoint: {{[0-9]+}} forward steps, 37 taped steps, 5 slots
// STATS-DAG: enzyme checkpoint: {{[0-9]+}} forward steps, 37 taped steps, 15 slots
// STATS-DAG: enzyme checkpoint: {{[0-9]+}} forward steps, 37 taped steps, 12 slots
