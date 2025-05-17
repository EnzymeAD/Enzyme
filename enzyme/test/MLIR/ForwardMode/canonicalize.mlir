// RUN: %eopt --canonicalize %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  
  func.func @dsq(%x : f64, %dx : f64) -> f64 {
    %p,%r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64,f64)
  
    // CHECK: %[[VAL:.*]] = enzyme.fwddiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_dupnoneed>\]}}
    // CHECK: return %[[VAL]] : f64
    return %r : f64
  }
  
  // -----

  func.func @dsq2(%x : f64, %dx : f64) -> f64 {
    %p,%r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64,f64)
  
    // CHECK: %[[VAL:.*]] = enzyme.fwddiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_const>\]}}
    // CHECK: return %[[VAL]] : f64
    return %p : f64
  }
  
  // -----

  func.func @dsq3(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64  
    %p = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_const>] } : (f64, f64) -> f64
    // CHECK: enzyme.fwddiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_constnoneed>\]}}
    return %cst : f64
  }
  
  // -----

  func.func @dsq4(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64  
    %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> f64
    // CHECK: enzyme.fwddiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_constnoneed>\]}}
    return %cst : f64
  }


  // -----

  func.func @dsq6(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64  
    enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_constnoneed>] } : (f64, f64) -> ()
    // CHECK: enzyme.fwddiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_constnoneed>\]}}
    return %cst : f64
  }
  
  // -----
  // Greedy test  
  func.func @dsq5(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64  
    %p, %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64,f64)
    // CHECK: enzyme.fwddiff @square(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_constnoneed>\]}}
    return %cst : f64
  }
}

