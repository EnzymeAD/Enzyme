// RUN: %eopt --enzyme %s | FileCheck %s

// An async gpu.launch carries a token result that its gpu.terminator does not
// forward (the terminator has no operands at all): the terminator handler must
// only pair the results a terminator operand exists for, not assert on the
// count mismatch.

module {
  func.func @square(%x : memref<?xf32>, %y : memref<?xf32>, %n : index) {
    %c1 = arith.constant 1 : index
    %t = gpu.launch async blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
        threads(%tx, %ty, %tz) in (%sx = %n, %sy = %c1, %sz = %c1) {
      %v = memref.load %x[%tx] : memref<?xf32>
      %m = arith.mulf %v, %v : f32
      memref.store %m, %y[%tx] : memref<?xf32>
      gpu.terminator
    }
    return
  }
  func.func @dsquare(%x : memref<?xf32>, %dx : memref<?xf32>, %y : memref<?xf32>, %dy : memref<?xf32>, %n : index) {
    enzyme.fwddiff @square(%x, %dx, %y, %dy, %n) { activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>], ret_activity=[] } : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()
    return
  }
}

// CHECK: func.func private @fwddiffesquare(%[[x:.+]]: memref<?xf32>, %[[dx:.+]]: memref<?xf32>, %[[y:.+]]: memref<?xf32>, %[[dy:.+]]: memref<?xf32>, %[[n:.+]]: index) {
// CHECK-NEXT:   %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:   %[[token:.+]] = gpu.launch async blocks(%[[bx:.+]], %[[by:.+]], %[[bz:.+]]) in (%[[gx:.+]] = %[[c1]], %[[gy:.+]] = %[[c1]], %[[gz:.+]] = %[[c1]]) threads(%[[tx:.+]], %[[ty:.+]], %[[tz:.+]]) in (%[[sx:.+]] = %[[n]], %[[sy:.+]] = %[[c1]], %[[sz:.+]] = %[[c1]]) {
// CHECK-NEXT:     %[[dv:.+]] = memref.load %[[dx]][%[[tx]]] : memref<?xf32>
// CHECK-NEXT:     %[[v:.+]] = memref.load %[[x]][%[[tx]]] : memref<?xf32>
// CHECK-NEXT:     %[[p0:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f32
// CHECK-NEXT:     %[[p1:.+]] = arith.mulf %[[dv]], %[[v]] fastmath<fast> : f32
// CHECK-NEXT:     %[[dm:.+]] = arith.addf %[[p0]], %[[p1]] fastmath<fast> : f32
// CHECK-NEXT:     %[[m:.+]] = arith.mulf %[[v]], %[[v]] : f32
// CHECK-NEXT:     memref.store %[[dm]], %[[dy]][%[[tx]]] : memref<?xf32>
// CHECK-NEXT:     memref.store %[[m]], %[[y]][%[[tx]]] : memref<?xf32>
// CHECK-NEXT:     gpu.terminator
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
