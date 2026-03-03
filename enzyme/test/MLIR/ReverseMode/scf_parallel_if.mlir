// RUN: %eopt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse | FileCheck %s

module {
  func.func @scale(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = arith.cmpi sgt, %dim, %c0 : index
    scf.if %0 {
      scf.parallel (%arg3) = (%c0) to (%dim) step (%c1) {
        %1 = memref.load %arg0[%arg3] : memref<?xf64>
        %2 = memref.load %arg1[%arg3] : memref<?xf64>
        %3 = arith.mulf %1, %2 : f64
        memref.store %3, %arg2[%arg3] : memref<?xf64>
      }
    }
    return
  }

  func.func @dscale(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) {
    enzyme.autodiff @scale(%arg0, %arg3, %arg1, %arg2, %arg4) {
        activity=[#enzyme<activity enzyme_dup>, 
                  #enzyme<activity enzyme_const>, 
                  #enzyme<activity enzyme_dup>],
        ret_activity=[]
      } : (memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) -> ()

    return
  }

  // CHECK: @diffescale(%[[arg0:.+]]: memref<?xf64>, %[[arg1:.+]]: memref<?xf64>, %[[arg2:.+]]: memref<?xf64>, %[[arg3:.+]]: memref<?xf64>, %[[arg4:.+]]: memref<?xf64>) {
  // CHECK:   %[[c1:.+]] = arith.constant 1 : index
  // CHECK:   %[[c0:.+]] = arith.constant 0 : index
  // CHECK:   %[[cst:.+]] = arith.constant 0.000000e+00 : f64
  // CHECK:   %[[dim:.+]] = memref.dim %[[arg0]], %[[c0]] : memref<?xf64>
  // CHECK:   %[[x0:.+]] = arith.cmpi sgt, %[[dim]], %[[c0]] : index
  // CHECK:   scf.if %[[x0]] {
  // CHECK:     %[[alloc:.+]] = memref.alloc(%[[dim]]) : memref<?xf64>
  // CHECK:     scf.parallel (%[[arg5:.+]]) = (%[[c0]]) to (%[[dim]]) step (%[[c1]]) {
  // CHECK:       %[[x1:.+]] = memref.load %arg0[%[[arg5]]] : memref<?xf64>
  // CHECK:       %[[x2:.+]] = memref.load %arg2[%[[arg5]]] : memref<?xf64>
  // CHECK:       memref.store %[[x2]], %[[alloc]][%[[arg5]]] : memref<?xf64>
  // CHECK:       %[[x3:.+]] = arith.mulf %[[x1]], %[[x2]] : f64
  // CHECK:       memref.store %[[x3]], %arg3[%[[arg5]]] : memref<?xf64>
  // CHECK:       scf.reduce 
  // CHECK:     }
  // CHECK:     scf.parallel (%[[arg5:.+]]) = (%[[c0]]) to (%[[dim]]) step (%[[c1]]) {
  // CHECK:       %[[x1:.+]] = memref.load %[[alloc]][%[[arg5]]] : memref<?xf64>
  // CHECK:       %[[x2:.+]] = memref.load %arg4[%[[arg5]]] : memref<?xf64>
  // CHECK:       memref.store %[[cst]], %arg4[%[[arg5]]] : memref<?xf64>
  // CHECK:       %[[x3:.+]] = arith.mulf %[[x2]], %[[x1]] : f64
  // CHECK:       %[[x4:.+]] = memref.atomic_rmw addf %[[x3]], %arg1[%[[arg5]]] : (f64, memref<?xf64>) -> f64
  // CHECK:       scf.reduce 
  // CHECK:     }
  // CHECK:     memref.dealloc %[[alloc]] : memref<?xf64>
  // CHECK:   }
  // CHECK:   return
  // CHECK: }

}
