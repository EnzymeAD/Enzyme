// RUN: %eopt --canonicalize %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }

  func.func @square2(%x: f32, %y: f32) -> (f32, f32) {
    %next = arith.mulf %x, %x : f32
    %next2 = arith.mulf %y, %y : f32
    return %next, %next2 : f32, f32
  }

  // Test 1: no change
  func.func @test1(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> (f32,f32,f32,f32) {
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr0,%dr1) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK: %{{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg2, %arg3){{.*}}activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}
    return %r#0,%r#1,%r#2,%r#3 : f32,f32,f32,f32
  }
  

  // Test 2: active -> activenoneed, const -> constnoneed for ret_activity
  func.func @test2(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> (f32,f32) {
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr1) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>] } : (f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK: %{{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg3){{.*}}activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_constnoneed>, #enzyme<activity enzyme_activenoneed>]{{.*}}
    return %r#2, %r#3 : f32,f32
  }

  // Test 3: active -> const for inp_activity
  func.func @test3(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> (f32,f32,f32) {
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr0,%dr1) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK: {{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg2, %arg3){{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}
    return %r#0, %r#1, %r#3 : f32,f32,f32
  }

  // Test 4: remove everything
  func.func @test4(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> f32 {
    %cst = arith.constant 1.0000e+1 : f32
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr0,%dr1) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK: enzyme.autodiff @square2(%arg0, %arg1, %arg2, %arg3){{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>]{{.*}}ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_activenoneed>]{{.*}}
    return %cst : f32
  }

  // Test 5: active -> const for ret_activity (iff derivative is 0)
  func.func @test5(%x: f32, %y: f32, %dr0: f32) -> (f32,f32,f32,f32) {
    %cst = arith.constant 0.0000e+00 : f32
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr0,%cst) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK: %{{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg2, %cst){{.*}}activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]{{.*}}
    return %r#0,%r#1,%r#2,%r#3 : f32,f32,f32,f32
  }
} 
