// RUN: %eopt --enzyme %s | FileCheck -v %s

module {
  func.func @matvec(%arg0: memref<?x?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf64>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf64>
    scf.parallel (%arg3) = (%c0) to (%dim) step (%c1) {
      %0 = scf.for %arg4 = %c0 to %dim_0 step %c1 iter_args(%arg5 = %cst) -> (f64) {
        %1 = memref.load %arg0[%arg3, %arg4] : memref<?x?xf64>
        %2 = memref.load %arg1[%arg4] : memref<?xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = arith.addf %arg5, %3 : f64
        scf.yield %4 : f64
      }
      memref.store %0, %arg2[%arg3] : memref<?xf64>
    }
    return
  }

  func.func @dmatvec(%arg0: memref<?x?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?x?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) {
    enzyme.fwddiff @matvec(%arg0, %arg3, %arg1, %arg4, %arg2, %arg5) { 
        activity=[#enzyme<activity enzyme_dup>, 
                  #enzyme<activity enzyme_dup>, 
                  #enzyme<activity enzyme_dup>], 
        ret_activity=[] 
      } : (memref<?x?xf64>, memref<?x?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) -> ()
    return
  }

// CHECK: @fwddiffematvec(%[[arg0:.+]]: memref<?x?xf64>, %[[arg1:.+]]: memref<?x?xf64>, %[[arg2:.+]]: memref<?xf64>, %[[arg3:.+]]: memref<?xf64>, %[[arg4:.+]]: memref<?xf64>, %[[arg5:.+]]: memref<?xf64>) {
// CHECK:   %[[cst:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:   %[[cst_0:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:   %[[c1:.+]] = arith.constant 1 : index
// CHECK:   %[[c0:.+]] = arith.constant 0 : index
// CHECK:   %[[dim:.+]] = memref.dim %[[arg0:.+]], %[[c0]] : memref<?x?xf64>
// CHECK:   %[[dim_1:.+]] = memref.dim %[[arg0:.+]], %[[c1]] : memref<?x?xf64>
// CHECK:   scf.parallel (%[[arg6:.+]]) = (%[[c0]]) to (%dim) step (%[[c1]]) {
// CHECK:     %[[x0:.+]]:2 = scf.for %[[arg7:.+]] = %[[c0]] to %dim_1 step %[[c1]] iter_args(%[[arg8:.+]] = %[[cst_0]], %[[arg9:.+]] = %[[cst]]) -> (f64, f64) {
// CHECK:       %[[x1:.+]] = memref.load %[[arg1]][%[[arg6]], %[[arg7]]] : memref<?x?xf64>
// CHECK:       %[[x2:.+]] = memref.load %[[arg0]][%[[arg6]], %[[arg7]]] : memref<?x?xf64>
// CHECK:       %[[x3:.+]] = memref.load %[[arg3]][%[[arg7]]] : memref<?xf64>
// CHECK:       %[[x4:.+]] = memref.load %[[arg2]][%[[arg7]]] : memref<?xf64>
// CHECK:       %[[x5:.+]] = arith.mulf %[[x1]], %[[x4]] : f64
// CHECK:       %[[x6:.+]] = arith.mulf %[[x3]], %[[x2]] : f64
// CHECK:       %[[x7:.+]] = arith.addf %[[x5]], %[[x6]] : f64
// CHECK:       %[[x8:.+]] = arith.mulf %[[x2]], %[[x4]] : f64
// CHECK:       %[[x9:.+]] = arith.addf %[[arg9]], %[[x7]] : f64
// CHECK:       %[[x10:.+]] = arith.addf %[[arg8]], %[[x8]] : f64
// CHECK:       scf.yield %[[x10]], %[[x9]] : f64, f64
// CHECK:     }
// CHECK:     memref.store %[[0]]#1, %[[arg5]][%[[arg6]]] : memref<?xf64>
// CHECK:     memref.store %[[0]]#0, %[[arg4]][%[[arg6]]] : memref<?xf64>
// CHECK:     scf.reduce 
// CHECK:   }
// CHECK:   return
// CHECK: }

}
