// RUN: %eopt --canonicalize %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }

  func.func @copy_memref(%x : f64) -> memref<1xf64> {
    %tmp = memref.alloc() : memref<1xf64>
    %c0 = arith.constant 0 : index
    memref.store %x, %tmp[%c0] : memref<1xf64>
    return %tmp : memref<1xf64>
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

  func.func @dsq2_mutable(%x : f64, %dx : f64) -> memref<1xf64> {
    %p,%r = enzyme.fwddiff @copy_memref(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (memref<1xf64>,memref<1xf64>)
  
    // CHECK: %[[VAL:.*]]:2 = enzyme.fwddiff @copy_memref(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_dup>\]}}
    // CHECK: return %[[VAL]]#0 : memref<1xf64>
    return %p : memref<1xf64>
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

  func.func @dsq4_mutable(%x : f64, %dx : f64) -> f64 {
    %cst = arith.constant 1.0000e+1 : f64  
    %r = enzyme.fwddiff @copy_memref(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> memref<1xf64>
    // CHECK: %[[VAL:.*]] = enzyme.fwddiff @copy_memref(%arg0, %arg1) {{.*ret_activity = \[#enzyme<activity enzyme_dupnoneed>\]}}
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

  func.func @dsq7(%x : f64, %dx : f64) -> (f64,f64) {
    %cst = arith.constant 0.0000e+00 : f64  
    %p, %r = enzyme.fwddiff @square(%x, %cst) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64, f64)
    // CHECK: %{{.*}}:2 = enzyme.fwddiff @square(%arg0) {activity = [#enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_dup>]}
    return %p, %r : f64, f64
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

