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

// CHECK-LABEL: func.func private @diffeextract_to_diff0
// CHECK-SAME: (%arg0: !llvm.struct<(f64, f64)>, %arg1: f64) -> !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[V0:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[V1:.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[V2:.+]] = arith.mulf %[[V1]], %arg1 : f64
// CHECK-DAG:    %[[V3:.+]] = arith.mulf %[[V0]], %arg1 : f64
// CHECK-DAG:    %[[Z0:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[Z1:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:    %[[IV0:.+]] = llvm.insertvalue %[[Z1]], %[[Z0]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[IV1:.+]] = llvm.insertvalue %[[V2]], %[[IV0]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[Z2:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[Z3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:    %[[IV2:.+]] = llvm.insertvalue %[[V3]], %[[Z2]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[IV3:.+]] = llvm.insertvalue %[[Z3]], %[[IV2]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:    %[[ADD:.+]] = enzyme.foadd %[[IV1]], %[[IV3]]
// CHECK:        return %[[ADD]] : !llvm.struct<(f64, f64)>

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

// CHECK-LABEL: func.func private @diffeinsert_to_diff0
// CHECK-SAME: (%arg0: f64, %arg1: !llvm.struct<(f64, f64)>) -> f64
// CHECK-DAG:    %[[EV:.+]] = llvm.extractvalue %arg1[0] : !llvm.struct<(f64, f64)>
// CHECK:        return %[[EV]] : f64
