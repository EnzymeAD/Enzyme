// RUN: %eopt --enzyme %s | FileCheck -v %s

module {
  func.func @nrm2(%arg0: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %1 = scf.parallel (%arg1) = (%c0) to (%dim) step (%c1) init (%cst) -> (f64) {
      %2 = memref.load %arg0[%arg1] : memref<?xf64>
      %3 = arith.mulf %2, %2 : f64
      scf.reduce(%3 : f64) {
        ^bb0(%arg9: f64, %arg10: f64):
          %9 = arith.addf %arg9, %arg10 : f64
          scf.reduce.return %9 : f64
      }
    }
    return %1 : f64
  }

  func.func @dnrm2(%arg0: memref<?xf64>, %arg1: memref<?xf64>) -> f64  {
    %3 = enzyme.fwddiff @nrm2(%arg0, %arg1) { 
        activity=[#enzyme<activity enzyme_dup>], 
        ret_activity=[#enzyme<activity enzyme_dupnoneed>] 
      } : (memref<?xf64>, memref<?xf64>) -> (f64)
    return %3 : f64
  }

  // CHECK: @fwddiffenrm2(%[[arg0:.+]]: memref<?xf64>, %[[arg1:.+]]: memref<?xf64>) -> f64 {
  // CHECK:   %[[c0:.+]] = arith.constant 0 : index
  // CHECK:   %[[c1:.+]] = arith.constant 1 : index
  // CHECK:   %[[cst:.+]] = arith.constant 0.000000e+00 : f64
  // CHECK:   %[[cst_0:.+]] = arith.constant 0.000000e+00 : f64
  // CHECK:   %[[dim:.+]] = memref.dim %[[arg0]], %[[c0]] : memref<?xf64>
  // CHECK:   %[[x0:.+]]:2 = scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%dim) step (%[[c1]]) init (%[[cst_0]], %[[cst]]) -> (f64, f64) {
  // CHECK:     %[[x1:.+]] = memref.load %[[arg1]][%[[arg2]]] : memref<?xf64>
  // CHECK:     %[[x2:.+]] = memref.load %[[arg0]][%[[arg2]]] : memref<?xf64>
  // CHECK:     %[[x3:.+]] = arith.mulf %[[x1]], %[[x2]] : f64
  // CHECK:     %[[x4:.+]] = arith.mulf %[[x1]], %[[x2]] : f64
  // CHECK:     %[[x5:.+]] = arith.addf %[[x3]], %[[x4]] : f64
  // CHECK:     %[[x6:.+]] = arith.mulf %[[x2]], %[[x2]] : f64
  // CHECK:     scf.reduce(%[[x6]], %[[x5]] : f64, f64) {
  // CHECK:     ^bb0(%[[arg3:.+]]: f64, %[[arg4:.+]]: f64):
  // CHECK:       %[[x7:.+]] = arith.addf %[[arg3]], %[[arg4]] : f64
  // CHECK:       scf.reduce.return %[[x7]] : f64
  // CHECK:     }, {
  // CHECK:     ^bb0(%[[arg3:.+]]: f64, %[[arg4:.+]]: f64):
  // CHECK:       %[[x7:.+]] = arith.addf %[[arg3]], %[[arg4]] : f64
  // CHECK:       scf.reduce.return %[[x7]] : f64
  // CHECK:     }
  // CHECK:   }
  // CHECK:   return %[[x0]]#1 : f64
  // CHECK: }

}
