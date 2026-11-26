// RUN: %eopt --canonicalize %s | FileCheck %s

// An enzyme_activenoneed return whose seed is a constant zero. The canonicalizer
// drops that seed, so it must also mark the return enzyme_constnoneed; keeping
// enzyme_activenoneed leaves the op a seed short, and the AD pass fails with
// "Too few arguments to autodiff op". The unused gradient of %x makes the
// pattern rewrite the op regardless.

module {
  func.func @square2(%x: f32, %y: f32) -> (f32, f32) {
    %x2 = arith.mulf %x, %x : f32
    %y2 = arith.mulf %y, %y : f32
    return %x2, %y2 : f32, f32
  }

  func.func @zero_seed(%x: f32, %y: f32, %dr1: f32) -> f32 {
    %zero = arith.constant 0.000000e+00 : f32
    %r:2 = enzyme.autodiff @square2(%x, %y, %zero, %dr1) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_activenoneed>]}> : (f32, f32, f32, f32) -> (f32, f32)
    return %r#1 : f32
  }
}

// CHECK-LABEL: func.func @zero_seed(%[[X:.+]]: f32, %[[Y:.+]]: f32, %[[DR1:.+]]: f32)
// CHECK: enzyme.autodiff @square2(%[[X]], %[[Y]], %[[DR1]]){{.*}}activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>]{{.*}}ret_activity = [#enzyme.activity<enzyme_constnoneed>, #enzyme.activity<enzyme_activenoneed>]
