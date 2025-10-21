// RUN: %eopt --canonicalize %s | FileCheck %s
// RUN: %eopt --inline-enzyme-regions --canonicalize %s | FileCheck %s --check-prefix=INLINE
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
    // CHECK-LABEL: func.func @test1
    // CHECK: %{{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg2, %arg3){{.*}}activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}
    
    // INLINE-LABEL: func.func @test1
    // INLINE: %0:4 = enzyme.autodiff_region(%arg0, %arg1, %arg2, %arg3) {
    // INLINE-NEXT: ^bb0(%arg4: f32, %arg5: f32):
    // INLINE-NEXT:   %1 = arith.mulf %arg4, %arg4 : f32
    // INLINE-NEXT:   %2 = arith.mulf %arg5, %arg5 : f32
    // INLINE-NEXT:   enzyme.yield %1, %2 : f32, f32
    // INLINE-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], fn = "square2", fn_attrs = {}, ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]} : (f32, f32, f32, f32) -> (f32, f32, f32, f32)
    return %r#0,%r#1,%r#2,%r#3 : f32,f32,f32,f32
  }

  // Test 2: active -> activenoneed, const -> constnoneed for ret_activity
  func.func @test2(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> (f32,f32) {
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr1) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>] } : (f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK-LABEL: func.func @test2
    // CHECK: %{{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg3){{.*}}activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_constnoneed>, #enzyme<activity enzyme_activenoneed>]{{.*}}
    
    // INLINE-LABEL: func.func @test2
    // INLINE: %0:2 = enzyme.autodiff_region(%arg0, %arg1, %arg3) {
    // INLINE-NEXT: ^bb0(%arg4: f32, %arg5: f32):
    // INLINE-NEXT:   %1 = arith.mulf %arg4, %arg4 : f32
    // INLINE-NEXT:   %2 = arith.mulf %arg5, %arg5 : f32
    // INLINE-NEXT:   enzyme.yield %1, %2 : f32, f32
    // INLINE-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], fn = "square2", ret_activity = [#enzyme<activity enzyme_constnoneed>, #enzyme<activity enzyme_activenoneed>]} : (f32, f32, f32) -> (f32, f32)
    return %r#2, %r#3 : f32,f32
  }

  // Test 3: active -> const for inp_activity
  func.func @test3(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> (f32,f32,f32) {
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr0,%dr1) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK-LABEL: func.func @test3
    // CHECK: {{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg2, %arg3){{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}

    // INLINE-LABEL: func.func @test3
    // INLINE: %0:3 = enzyme.autodiff_region(%arg0, %arg1, %arg2, %arg3) {
    // INLINE-NEXT: ^bb0(%arg4: f32, %arg5: f32):
    // INLINE-NEXT:   %1 = arith.mulf %arg4, %arg4 : f32
    // INLINE-NEXT:   %2 = arith.mulf %arg5, %arg5 : f32
    // INLINE-NEXT:   enzyme.yield %1, %2 : f32, f32
    // INLINE-NEXT: } attributes {activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>], fn = "square2", ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]} : (f32, f32, f32, f32) -> (f32, f32, f32)
    return %r#0, %r#1, %r#3 : f32,f32,f32
  }

  // Test 4: remove everything
  func.func @test4(%x: f32, %y: f32, %dr0: f32, %dr1: f32) -> f32 {
    %cst = arith.constant 1.0000e+1 : f32
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr0,%dr1) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK-LABEL: func.func @test4
    // CHECK: enzyme.autodiff @square2(%arg0, %arg1, %arg2, %arg3){{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>]{{.*}}ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_activenoneed>]{{.*}}

    // INLINE-LABEL: func.func @test4
    // INLINE: %cst = arith.constant 1.000000e+01 : f32
    // INLINE-NEXT: enzyme.autodiff_region(%arg0, %arg1, %arg2, %arg3) {
    // INLINE-NEXT: ^bb0(%arg4: f32, %arg5: f32):
    // INLINE-NEXT:   %0 = arith.mulf %arg4, %arg4 : f32
    // INLINE-NEXT:   %1 = arith.mulf %arg5, %arg5 : f32
    // INLINE-NEXT:   enzyme.yield %0, %1 : f32, f32
    // INLINE-NEXT: } attributes {activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>], fn = "square2", ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_activenoneed>]} : (f32, f32, f32, f32) -> ()
    return %cst : f32
  }

  // Test 5: active -> const for ret_activity (iff derivative is 0)
  func.func @test5(%x: f32, %y: f32, %dr0: f32) -> (f32,f32,f32,f32) {
    %cst = arith.constant 0.0000e+00 : f32
    %r:4 = enzyme.autodiff @square2(%x,%y,%dr0,%cst) { activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>] } : (f32,f32,f32,f32) -> (f32,f32,f32,f32)
    // CHECK-LABEL: func.func @test5
    // CHECK: %{{.*}} = enzyme.autodiff @square2(%arg0, %arg1, %arg2){{.*}}activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]{{.*}}ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]{{.*}}

    // INLINE-LABEL: func.func @test5
    // INLINE: %0:4 = enzyme.autodiff_region(%arg0, %arg1, %arg2) {
    // INLINE-NEXT: ^bb0(%arg3: f32, %arg4: f32):
    // INLINE-NEXT:   %1 = arith.mulf %arg3, %arg3 : f32
    // INLINE-NEXT:   %2 = arith.mulf %arg4, %arg4 : f32
    // INLINE-NEXT:   enzyme.yield %1, %2 : f32, f32
    // INLINE-NEXT: } attributes {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], fn = "square2", ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (f32, f32, f32) -> (f32, f32, f32, f32)
    return %r#0,%r#1,%r#2,%r#3 : f32,f32,f32,f32
  }
} 
