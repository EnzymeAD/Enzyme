// RUN: enzymemlir-opt --split-input-file --hoist-enzyme-regions %s | FileCheck %s
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
