// Verify that enzyme.autodiff and enzyme.fwddiff ops — the ops constructed by
// the MLIRCAPIEnzyme C API — parse and round-trip correctly.
//
// RUN: %eopt %s | FileCheck %s

module {
  func.func @square(%x: f64) -> f64 {
    %0 = arith.mulf %x, %x : f64
    return %0 : f64
  }

  // CHECK-LABEL: func.func @test_autodiff
  func.func @test_autodiff(%x: f64, %dx: f64) -> f64 {
    // CHECK: enzyme.autodiff @square(%{{.*}}, %{{.*}})
    // CHECK-SAME: activity = [#enzyme<activity enzyme_dup>]
    // CHECK-SAME: ret_activity = [#enzyme<activity enzyme_active>]
    %0 = enzyme.autodiff @square(%x, %dx) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_active>]} : (f64, f64) -> f64
    return %0 : f64
  }

  // CHECK-LABEL: func.func @test_fwddiff
  func.func @test_fwddiff(%x: f64, %dx: f64) -> f64 {
    // CHECK: enzyme.fwddiff @square(%{{.*}}, %{{.*}})
    // CHECK-SAME: activity = [#enzyme<activity enzyme_dup>]
    // CHECK-SAME: ret_activity = [#enzyme<activity enzyme_dup>]
    %0 = enzyme.fwddiff @square(%x, %dx) {activity = [#enzyme<activity enzyme_dup>], ret_activity = [#enzyme<activity enzyme_dup>]} : (f64, f64) -> f64
    return %0 : f64
  }
}
