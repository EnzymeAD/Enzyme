// RUN: %eopt --split-input-file --hoist-enzyme-regions %s | FileCheck %s
func.func @foo(%arg0: f64, %arg1: f64,%xx: f64) -> f64 {

  %yy_cst = arith.constant 100.0 : f64
  %0 = enzyme.autodiff_region(%arg0, %arg1) {
  ^bb0(%arg2: f64):
    // hoistable constant ops
    %c0 = arith.constant 0.0 : f64
    %c1 = arith.constant 1.0 : f64
    %c2 = arith.constant 2.0 : f64
    %cx = arith.mulf %c2, %xx : f64

    %sq = arith.mulf %arg2, %arg2 : f64
    %sqx = arith.mulf %sq, %cx : f64

    // hoistable loops
    %yy0 = arith.constant 2.5 : f64
    %one = arith.constant 1 : index
    %ten = arith.constant 10 : index
    %yy = scf.for %iv = %one to %ten step %one iter_args(%yy_iter = %yy0) -> (f64) {
      %tm = arith.mulf %yy_iter, %yy_cst : f64
      %ta = scf.for %jv = %one to %ten step %one iter_args(%tm_iter = %tm) -> (f64) { 
        %ta = arith.addf %tm, %cx : f64
        scf.yield %ta : f64
      }
      scf.yield %ta : f64
    }

    %sqxy = arith.mulf %sqx, %yy : f64 
    %zz0 = arith.addf %sqx, %c0 : f64  
    %zz = scf.for %iv = %one to %ten step %one iter_args(%zz_iter = %zz0) ->(f64) {
      %zm = arith.addf %zz_iter, %sqx : f64
      %zout = arith.mulf %zm, %zz_iter : f64
      scf.yield %zout : f64
    }

    %sqxyz = arith.mulf %zz, %sqxy : f64
    enzyme.yield %sqxyz : f64
  } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @foo
// CHECK-SAME: (%arg0: f64, %arg1: f64, %arg2: f64) -> f64
// CHECK: %c10 = arith.constant 10 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %cst = arith.constant 2.500000e+00 : f64
// CHECK: %cst_0 = arith.constant 2.000000e+00 : f64
// CHECK: %cst_1 = arith.constant 0.000000e+00 : f64
// CHECK: %cst_2 = arith.constant 1.000000e+02 : f64
// CHECK: %0 = arith.mulf %arg2, %cst_0 : f64
// CHECK: %1 = scf.for %{{.*}} = %c1 to %c10 step %c1 iter_args(%{{.*}} = %cst) -> (f64) {
// CHECK:   %{{.*}} = arith.mulf %{{.*}}, %cst_2 : f64
// CHECK:   %{{.*}} = scf.for %{{.*}} = %c1 to %c10 step %c1 iter_args(%{{.*}} = %{{.*}}) -> (f64) {
// CHECK:     %{{.*}} = arith.addf %{{.*}}, %0 : f64
// CHECK:     scf.yield %{{.*}} : f64
// CHECK:   }
// CHECK:   scf.yield %{{.*}} : f64
// CHECK: }
// CHECK: %2 = enzyme.autodiff_region(%arg0, %arg1) {
// CHECK: ^bb0(%{{.*}}: f64):
// CHECK:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK:   %{{.*}} = arith.mulf %{{.*}}, %0 : f64
// CHECK:   %{{.*}} = arith.mulf %{{.*}}, %1 : f64
// CHECK:   %{{.*}} = arith.addf %{{.*}}, %cst_1 : f64
// CHECK:   %{{.*}} = scf.for %{{.*}} = %c1 to %c10 step %c1 iter_args(%{{.*}} = %{{.*}}) -> (f64) {
// CHECK:     %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK:     %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK:     scf.yield %{{.*}} : f64
// CHECK:   }
// CHECK:   %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK:   enzyme.yield %{{.*}} : f64
// CHECK: } attributes {{.*}} : (f64, f64) -> f64

// -----

func.func @bar(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: i1) -> f64 {
  %0 = enzyme.autodiff_region(%arg0, %arg1) {
  ^bb0(%arg4: f64):
    %cst = arith.constant 2.000000e+00 : f64
    %1 = arith.mulf %cst, %arg2 : f64
    %2 = arith.mulf %arg4, %arg4 : f64
    cf.cond_br %arg3, ^bb1(%1 : f64), ^bb2(%arg2 : f64)
    ^bb1(%3: f64):  // pred: ^bb0
    %4 = arith.mulf %3, %3 : f64
    cf.br ^bb3(%4 : f64)
    ^bb2(%5: f64):  // pred: ^bb0
    %tmp = arith.addf %1, %1 : f64
    %6 = arith.addf %5, %tmp : f64
    cf.br ^bb3(%6 : f64)
    ^bb3(%7: f64):  // 2 preds: ^bb1, ^bb2
    %9 = arith.addf %1, %1 : f64
    %8 = arith.mulf %2, %7 : f64
    %10 = arith.mulf %8, %9 : f64
    enzyme.yield %10 : f64
  } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
  return %0 : f64
}


// CHECK: func.func @bar(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: i1) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.mulf %arg2, %cst : f64
// CHECK-NEXT:   %1 = arith.addf %0, %0 : f64
// CHECK-NEXT:   %2 = enzyme.autodiff_region(%arg0, %arg1) {
// CHECK-NEXT:   ^bb0(%arg4: f64):
// CHECK-NEXT:     %3 = arith.mulf %arg4, %arg4 : f64
// CHECK-NEXT:     cf.cond_br %arg3, ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     %4 = arith.mulf %0, %0 : f64
// CHECK-NEXT:     cf.br ^bb3(%4 : f64)
// CHECK-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     %5 = arith.addf %0, %0 : f64
// CHECK-NEXT:     %6 = arith.addf %arg2, %5 : f64
// CHECK-NEXT:     cf.br ^bb3(%6 : f64)
// CHECK-NEXT:   ^bb3(%7: f64):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:     %8 = arith.mulf %3, %7 : f64
// CHECK-NEXT:     %9 = arith.mulf %8, %1 : f64
// CHECK-NEXT:     enzyme.yield %9 : f64
// CHECK-NEXT:   } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
// CHECK-NEXT:   return %2 : f64
// CHECK-NEXT: }

// -----

func.func @baz(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: i1) -> f64 { 
  %0 = enzyme.autodiff_region(%arg0, %arg1) {
  ^bb0(%arg4: f64):
    %cst = arith.constant 2.000000e+00 : f64 // h
    %1 = arith.mulf %cst, %arg2 : f64 // h 
    %2 = arith.mulf %arg4, %arg4 : f64 // h
    cf.cond_br %arg3, ^bb1(%1 : f64), ^bb2(%arg2 : f64)
    ^bb1(%3: f64):  // pred: ^bb0
    %4 = arith.mulf %3, %3 : f64 
    cf.br ^bb3(%4 : f64)
    ^bb2(%5: f64):  // pred: ^bb0
    %tmp = arith.addf %1, %1 : f64
    %6 = arith.addf %5, %tmp : f64
    cf.br ^bb3(%6 : f64)
    ^bb3(%7: f64):  // 2 preds: ^bb1, ^bb2
    %9 = arith.addf %1, %1 : f64
    %8 = arith.mulf %2, %7 : f64
    %10 = arith.mulf %8, %9 : f64
    enzyme.yield %10 : f64
  } attributes {activity = [#enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
  return %0 : f64
}


// CHECK: func.func @baz(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: i1) -> f64 {
// CHECK-NEXT:   %cst = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:   %0 = arith.mulf %arg2, %cst : f64
// CHECK-NEXT:   %1 = arith.mulf %arg0, %arg0 : f64
// CHECK-NEXT:   %2 = arith.addf %0, %0 : f64
// CHECK-NEXT:   %3 = enzyme.autodiff_region(%arg0, %arg1) {
// CHECK-NEXT:   ^bb0(%arg4: f64):
// CHECK-NEXT:     cf.cond_br %arg3, ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     %4 = arith.mulf %0, %0 : f64
// CHECK-NEXT:     cf.br ^bb3(%4 : f64)
// CHECK-NEXT:   ^bb2:  // pred: ^bb0
// CHECK-NEXT:     %5 = arith.addf %0, %0 : f64
// CHECK-NEXT:     %6 = arith.addf %arg2, %5 : f64
// CHECK-NEXT:     cf.br ^bb3(%6 : f64)
// CHECK-NEXT:   ^bb3(%7: f64):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:     %8 = arith.mulf %1, %7 : f64
// CHECK-NEXT:     %9 = arith.mulf %8, %2 : f64
// CHECK-NEXT:     enzyme.yield %9 : f64
// CHECK-NEXT:   } attributes {activity = [#enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (f64, f64) -> f64
// CHECK-NEXT:   return %3 : f64
// CHECK-NEXT: }
