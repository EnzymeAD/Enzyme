// RUN: %eopt --enzyme %s | FileCheck %s

// Test reverse mode AD for llvm.extractvalue on active (f64) struct elements.
// See https://github.com/EnzymeAD/Enzyme/issues/2811

func.func @extract_to_diff0(%arg0: !llvm.struct<(f64, f64)>) -> f64 {
  %f0 = llvm.extractvalue %arg0[0] : !llvm.struct<(f64, f64)>
  %f1 = llvm.extractvalue %arg0[1] : !llvm.struct<(f64, f64)>
  %m  = arith.mulf %f0, %f1 : f64
  return %m : f64
}

llvm.func @extract(%s: !llvm.struct<(f64, f64)>, %seed: f64)
                  -> !llvm.struct<(f64, f64)> {
  %g = enzyme.autodiff @extract_to_diff0(%s, %seed) {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (!llvm.struct<(f64, f64)>, f64) -> !llvm.struct<(f64, f64)>
  llvm.return %g : !llvm.struct<(f64, f64)>
}

// CHECK-LABEL: func.func private @diffeextract_to_diff0
// CHECK-SAME: (%arg0: !llvm.struct<(f64, f64)>, %arg1: f64) -> !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[GINIT:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<!llvm.struct<(f64, f64)>>
// CHECK-NEXT:   %[[P0:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[S0:.+]] = llvm.insertvalue %[[CST0]], %[[P0]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[CST1:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[S1:.+]] = llvm.insertvalue %[[CST1]], %[[S0]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   "enzyme.set"(%[[GINIT]], %[[S1]]) : (!enzyme.Gradient<!llvm.struct<(f64, f64)>>, !llvm.struct<(f64, f64)>) -> ()
// CHECK-NEXT:   %[[G0:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:   %[[CST2:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   "enzyme.set"(%[[G0]], %[[CST2]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:   %[[G1:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:   %[[CST3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   "enzyme.set"(%[[G1]], %[[CST3]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:   %[[C0:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:   %[[C1:.+]] = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK-NEXT:   %[[GMUL:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:   %[[CST4:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   "enzyme.set"(%[[GMUL]], %[[CST4]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:   %[[EV0:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[EV1:.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   "enzyme.push"(%[[C1]], %[[EV0]]) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:   "enzyme.push"(%[[C0]], %[[EV1]]) : (!enzyme.Cache<f64>, f64) -> ()
// CHECK-NEXT:   %[[MUL:.+]] = arith.mulf %[[EV0]], %[[EV1]] : f64
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %[[GETM:.+]] = "enzyme.get"(%[[GMUL]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:   %[[SUM:.+]] = arith.addf %[[GETM]], %arg1 : f64
// CHECK-NEXT:   "enzyme.set"(%[[GMUL]], %[[SUM]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:   %[[DRES:.+]] = "enzyme.get"(%[[GMUL]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:   %[[CST5:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   "enzyme.set"(%[[GMUL]], %[[CST5]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:   %[[POP0:.+]] = "enzyme.pop"(%[[C1]]) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:   %[[POP1:.+]] = "enzyme.pop"(%[[C0]]) : (!enzyme.Cache<f64>) -> f64
// CHECK-NEXT:   %[[MUL0:.+]] = arith.mulf %[[DRES]], %[[POP1]] : f64
// CHECK-NEXT:   %[[GET1:.+]] = "enzyme.get"(%[[G1]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:   %[[ADD0:.+]] = arith.addf %[[GET1]], %[[MUL0]] : f64
// CHECK-NEXT:   "enzyme.set"(%[[G1]], %[[ADD0]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:   %[[MUL1:.+]] = arith.mulf %[[DRES]], %[[POP0]] : f64
// CHECK-NEXT:   %[[GET0:.+]] = "enzyme.get"(%[[G0]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK-NEXT:   %[[ADD1:.+]] = arith.addf %[[GET0]], %[[MUL1]] : f64
// CHECK-NEXT:   "enzyme.set"(%[[G0]], %[[ADD1]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:        return %{{.+}} : !llvm.struct<(f64, f64)>
// CHECK-NEXT: }

// -----

// Test reverse mode AD for llvm.insertvalue on active (f64) struct elements.
// See https://github.com/EnzymeAD/Enzyme/issues/2812

func.func @insert_to_diff0(%arg0: f64) -> !llvm.struct<(f64, f64)> {
  %z  = llvm.mlir.constant(0.0 : f64) : f64
  %u  = llvm.mlir.undef : !llvm.struct<(f64, f64)>
  %s0 = llvm.insertvalue %arg0, %u[0]  : !llvm.struct<(f64, f64)>
  %s1 = llvm.insertvalue %z,   %s0[1] : !llvm.struct<(f64, f64)>
  return %s1 : !llvm.struct<(f64, f64)>
}

llvm.func @insert(%x: f64, %seed: !llvm.struct<(f64, f64)>) -> f64 {
  %g = enzyme.autodiff @insert_to_diff0(%x, %seed) {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f64, !llvm.struct<(f64, f64)>) -> f64
  llvm.return %g : f64
}

// CHECK-LABEL: func.func private @diffeinsert_to_diff0
// CHECK-SAME: (%arg0: f64, %arg1: !llvm.struct<(f64, f64)>) -> f64
// CHECK-NEXT:   %[[G0:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK-NEXT:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   "enzyme.set"(%[[G0]], %[[CST0]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK-NEXT:   %[[GS0:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<!llvm.struct<(f64, f64)>>
// CHECK-NEXT:   %[[P0:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[Z0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[S0:.+]] = llvm.insertvalue %[[Z0]], %[[P0]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[Z1:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[S1:.+]] = llvm.insertvalue %[[Z1]], %[[S0]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   "enzyme.set"(%[[GS0]], %[[S1]]) : (!enzyme.Gradient<!llvm.struct<(f64, f64)>>, !llvm.struct<(f64, f64)>) -> ()
// CHECK-NEXT:   %[[GS1:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<!llvm.struct<(f64, f64)>>
// CHECK-NEXT:   %[[P1:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[Z2:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[S2:.+]] = llvm.insertvalue %[[Z2]], %[[P1]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[Z3:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:   %[[S3:.+]] = llvm.insertvalue %[[Z3]], %[[S2]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   "enzyme.set"(%[[GS1]], %[[S3]]) : (!enzyme.Gradient<!llvm.struct<(f64, f64)>>, !llvm.struct<(f64, f64)>) -> ()
// CHECK-NEXT:   %[[CNST:.+]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
// CHECK-NEXT:   %[[UNDEF:.+]] = llvm.mlir.undef : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[IV0:.+]] = llvm.insertvalue %arg0, %[[UNDEF]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[IV1:.+]] = llvm.insertvalue %[[CNST]], %[[IV0]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %[[GETS1:.+]] = "enzyme.get"(%[[GS1]]) : (!enzyme.Gradient<!llvm.struct<(f64, f64)>>) -> !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[P2:.+]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[EV0:.+]] = llvm.extractvalue %[[GETS1]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[EV1:.+]] = llvm.extractvalue %arg1[0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[ADD0:.+]] = arith.addf %[[EV0]], %[[EV1]] : f64
// CHECK-NEXT:   %[[INS0:.+]] = llvm.insertvalue %[[ADD0]], %[[P2]][0] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[EV2:.+]] = llvm.extractvalue %[[GETS1]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[EV3:.+]] = llvm.extractvalue %arg1[1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   %[[ADD1:.+]] = arith.addf %[[EV2]], %[[EV3]] : f64
// CHECK-NEXT:   %[[INS1:.+]] = llvm.insertvalue %[[ADD1]], %[[INS0]][1] : !llvm.struct<(f64, f64)>
// CHECK-NEXT:   "enzyme.set"(%[[GS1]], %[[INS1]]) : (!enzyme.Gradient<!llvm.struct<(f64, f64)>>, !llvm.struct<(f64, f64)>) -> ()
// CHECK:        return %{{.+}} : f64
// CHECK-NEXT: }
