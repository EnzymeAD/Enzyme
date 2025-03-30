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
}

