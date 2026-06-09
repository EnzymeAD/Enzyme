// RUN: %eopt --enzyme-activity-opt --split-input-file %s | FileCheck %s

module {
  func.func @square(%x : f64) -> f64{
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }

  func.func @square2(%x: f32, %y: f32) -> (f32, f32) {
    %o1 = arith.mulf %x, %x : f32
    %o2 = arith.mulf %y, %y : f32
    return %o1, %o2 : f32, f32
  }

  // func.func @test1(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> (f32,f32,f32) {
  //   %r:3 = enzyme.autodiff @square2(%x,%y,%dr0,%dr1) { activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32)
  //   // CHECK-LABEL: func.func @test1
  //   // CHECK: %{{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg3){{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>]{{.*}}
  //   return %r#0,%r#1,%r#2 : f32,f32,f32
  // }
  
  // -----

  // Fwddiff but we need to use results of activity analysis 
  func.func @test2(%x: f32, %y: f32, %dx: f32, %dy : f32) -> (f32,f32,f32,f32) {
    %r:3 = enzyme.fwddiff @square2(%x, %dx, %y, %dy) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>] } : (f32,f32,f32,f32) -> (f32,f32,f32)
    // CHECK-LABEL: func.func @test2
    // CHECK: %{{.*}} = enzyme.fwddiff @square2(%arg0, %arg2, %arg3){{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>]{{.*}}ret_activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>]{{.*}}

    return %r#0,%r#1,%r#2,%r#3 : f32,f32,f32,f32
  }
}

