// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops %s | FileCheck %s

// The index is read out of memory that is being differentiated, so the load is
// reached, but an i64 is nothing to differentiate and no shadow was ever made
// for it. The memory-identity handler set a shadow on every result all the
// same, and setDiffe went looking for the one that was not there.

module {
  func.func @f(%idx: memref<?xi64>, %data: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %i = memref.load %idx[%c0] : memref<?xi64>
    %j = arith.index_cast %i : i64 to index
    %v = memref.load %data[%j] : memref<?xf64>
    %s = arith.mulf %v, %v : f64
    memref.store %s, %data[%j] : memref<?xf64>
    return
  }

  func.func @df(%idx: memref<?xi64>, %didx: memref<?xi64>, %data: memref<?xf64>, %ddata: memref<?xf64>) {
    enzyme.fwddiff @f(%idx, %didx, %data, %ddata) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>], ret_activity=[] } : (memref<?xi64>, memref<?xi64>, memref<?xf64>, memref<?xf64>) -> ()
    return
  }
}

// The index is read once, from the primal, and the tangent of v*v is 2*v*dv.

// CHECK: func.func private @fwddiffef(%[[idx:.+]]: memref<?xi64>, %[[didx:.+]]: memref<?xi64>, %[[data:.+]]: memref<?xf64>, %[[ddata:.+]]: memref<?xf64>)
// CHECK:         %[[i:.+]] = memref.load %[[idx]]
// CHECK-NEXT:    %[[j:.+]] = arith.index_cast %[[i]] : i64 to index
// CHECK-NEXT:    %[[dv:.+]] = memref.load %[[ddata]]{{\[}}%[[j]]] : memref<?xf64>
// CHECK-NEXT:    %[[v:.+]] = memref.load %[[data]]{{\[}}%[[j]]] : memref<?xf64>
// CHECK-NEXT:    %[[a:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK-NEXT:    %[[b:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f64
// CHECK-NEXT:    %[[ds:.+]] = arith.addf %[[a]], %[[b]] fastmath<fast> : f64
// CHECK-NEXT:    %[[s:.+]] = arith.mulf %[[v]], %[[v]] : f64
// CHECK-NEXT:    memref.store %[[ds]], %[[ddata]]{{\[}}%[[j]]]
// CHECK-NEXT:    memref.store %[[s]], %[[data]]{{\[}}%[[j]]]
