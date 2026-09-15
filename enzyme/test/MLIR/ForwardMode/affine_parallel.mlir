// RUN: %eopt --enzyme %s | FileCheck %s

module {
  func.func @square(%x : memref<?xf64>, %y : memref<?xf64>, %n : index) {
    affine.parallel (%i) = (0) to (symbol(%n)) {
      %v = affine.load %x[%i] : memref<?xf64>
      %s = arith.mulf %v, %v : f64
      affine.store %s, %y[%i] : memref<?xf64>
    }
    return
  }

  func.func @dsquare(%x : memref<?xf64>, %dx : memref<?xf64>, %y : memref<?xf64>, %dy : memref<?xf64>, %n : index) {
    enzyme.fwddiff @square(%x, %dx, %y, %dy, %n) {
      activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
      ret_activity=[]
    } : (memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, index) -> ()
    return
  }
}

// CHECK:  func.func private @fwddiffesquare(%[[x:.+]]: memref<?xf64>, %[[dx:.+]]: memref<?xf64>, %[[y:.+]]: memref<?xf64>, %[[dy:.+]]: memref<?xf64>, %[[n:.+]]: index) {
// CHECK-NEXT:    affine.parallel (%[[i:.+]]) = (0) to (symbol(%[[n]])) {
// CHECK-NEXT:      %[[dv:.+]] = affine.load %[[dx]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:      %[[v:.+]] = affine.load %[[x]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:      %[[l:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK-NEXT:      %[[r:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK-NEXT:      %[[ds:.+]] = arith.addf %[[l]], %[[r]] fastmath<fast> : f64
// CHECK-NEXT:      %[[s:.+]] = arith.mulf %[[v]], %[[v]] : f64
// CHECK-NEXT:      affine.store %[[ds]], %[[dy]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:      affine.store %[[s]], %[[y]][%[[i]]] : memref<?xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
