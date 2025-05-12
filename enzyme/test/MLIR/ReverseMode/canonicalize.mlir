// RUN: %eopt --canonicalize %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }

  // Test 1: enzyme_active -> enzyme_activenoneed when value is unused
  func.func @test1(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64
    %r = enzyme.autodiff @square(%x, %dx) { 
      activity=[#enzyme<activity enzyme_active>], 
      ret_activity=[#enzyme<activity enzyme_active>] 
    } : (f64, f64) -> f64
    // CHECK: enzyme.autodiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_activenoneed>\]}}
    return %cst : f64
  }

  // Test 2: enzyme_const -> enzyme_constnoneed when value is unused
  func.func @test2(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64
    %r = enzyme.autodiff @square(%x, %dx) { 
      activity=[#enzyme<activity enzyme_active>], 
      ret_activity=[#enzyme<activity enzyme_const>] 
    } : (f64, f64) -> f64
    // CHECK: enzyme.autodiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_constnoneed>\]}}
    return %cst : f64
  }

  // Test 3: enzyme_activenoneed -> enzyme_constnoneed when value is unused
  func.func @test3(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64
    %r = enzyme.autodiff @square(%x, %dx) { 
      activity=[#enzyme<activity enzyme_active>], 
      ret_activity=[#enzyme<activity enzyme_activenoneed>] 
    } : (f64, f64) -> f64
    // CHECK: enzyme.autodiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_constnoneed>\]}}
    return %cst : f64
  }

  // Test 4: No change when value is used
  func.func @test4(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.autodiff @square(%x, %dx) { 
      activity=[#enzyme<activity enzyme_active>], 
      ret_activity=[#enzyme<activity enzyme_active>] 
    } : (f64, f64) -> f64
    // CHECK: enzyme.autodiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_active>\]}}
    return %r : f64
  }

  // Test 5: Multiple return values with mixed usage
  func.func @test5(%x : f64, %dx : f64) -> (f64, f64) {
    %cst = arith.constant 1.0000e+1 : f64
    %r1, %r2 = enzyme.autodiff @square(%x, %dx) { 
      activity=[#enzyme<activity enzyme_active>], 
      ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>] 
    } : (f64, f64) -> (f64, f64)
    // CHECK: enzyme.autodiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_constnoneed>\]}}
    return %r1, %cst : f64, f64
  }
} 