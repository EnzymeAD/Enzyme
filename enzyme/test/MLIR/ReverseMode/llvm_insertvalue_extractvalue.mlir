// RUN: %eopt --outline-enzyme-regions --enzyme %s | FileCheck %s

// Test reverse mode AD for llvm.extractvalue on active (f64) struct elements.
// See https://github.com/EnzymeAD/Enzyme/issues/2811
llvm.func @extract(%s: !llvm.struct<(f64, f64)>, %seed: f64)
                  -> !llvm.struct<(f64, f64)> {
  %g = enzyme.autodiff_region(%s, %seed) {
  ^bb0(%a_s: !llvm.struct<(f64, f64)>):
    %f0 = llvm.extractvalue %a_s[0] : !llvm.struct<(f64, f64)>
    %f1 = llvm.extractvalue %a_s[1] : !llvm.struct<(f64, f64)>
    %m  = arith.mulf %f0, %f1 : f64
    enzyme.yield %m : f64
  } attributes {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>],
    fn = "compute_extract"
  } : (!llvm.struct<(f64, f64)>, f64) -> !llvm.struct<(f64, f64)>
  llvm.return %g : !llvm.struct<(f64, f64)>
}

// CHECK-LABEL: func.func private @diffecompute_extract
// CHECK-SAME: (%arg0: !llvm.struct<(f64, f64)>, %arg1: f64) -> !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[V0:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[V1:.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[V2:.+]] = arith.mulf %[[V1]], %arg1 : f64
// CHECK-NEXT:   %[[V3:.+]] = arith.mulf %[[V0]], %arg1 : f64
// CHECK-NEXT:   %[[Z0:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[Z1:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[IV0:.+]] = llvm.insertvalue %[[Z1]], %[[Z0]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[IV1:.+]] = llvm.insertvalue %[[V2]], %[[IV0]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[Z2:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[Z3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[IV2:.+]] = llvm.insertvalue %[[V3]], %[[Z2]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[IV3:.+]] = llvm.insertvalue %[[Z3]], %[[IV2]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[ADD:.+]] = enzyme.foadd %[[IV1]], %[[IV3]]
// CHECK-NEXT:   return %[[ADD]] : !llvm.struct<(f64, f64)>

// -----

// Test reverse mode AD for llvm.insertvalue on active (f64) struct elements.
// See https://github.com/EnzymeAD/Enzyme/issues/2812
llvm.func @insert(%x: f64, %seed: !llvm.struct<(f64, f64)>) -> f64 {
  %g = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%a_x: f64):
    %z  = llvm.mlir.constant(0.0 : f64) : f64
    %u  = llvm.mlir.undef : !llvm.struct<(f64, f64)>
    %s0 = llvm.insertvalue %a_x, %u[0]  : !llvm.struct<(f64, f64)>
    %s1 = llvm.insertvalue %z,   %s0[1] : !llvm.struct<(f64, f64)>
    enzyme.yield %s1 : !llvm.struct<(f64, f64)>
  } attributes {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>],
    fn = "compute_insert"
  } : (f64, !llvm.struct<(f64, f64)>) -> f64
  llvm.return %g : f64
}

// CHECK-LABEL: func.func private @diffecompute_insert
// CHECK-SAME: (%arg0: f64, %arg1: !llvm.struct<(f64, f64)>) -> f64
// CHECK-NEXT:   %[[EV:.+]] = llvm.extractvalue %arg1[0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   return %[[EV]] : f64
