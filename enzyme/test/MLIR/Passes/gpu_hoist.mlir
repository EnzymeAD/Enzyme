// RUN: %eopt --remove-unnecessary-enzyme-ops %s | FileCheck %s

func.func @test_gpu_hoist(%ub_outer: index, %ub_inner: index, %val: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %cache = "enzyme.init"() : () -> !enzyme.Cache<memref<?xf32, 1>>

  scf.for %iv2 = %c0 to %ub_outer step %c1 {
    %mem = gpu.alloc (%ub_inner) : memref<?xf32, 1>
    memref.store %val, %mem[%c0] : memref<?xf32, 1>
    "enzyme.push"(%cache, %mem) : (!enzyme.Cache<memref<?xf32, 1>>, memref<?xf32, 1>) -> ()
  }

  scf.for %iv_rev = %c0 to %ub_outer step %c1 {
    %pop_mem = "enzyme.pop"(%cache) : (!enzyme.Cache<memref<?xf32, 1>>) -> memref<?xf32, 1>
    %ld = memref.load %pop_mem[%c0] : memref<?xf32, 1>
    gpu.dealloc %pop_mem : memref<?xf32, 1>
  }

  return
}

// CHECK: func.func @test_gpu_hoist(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[VAL:.+]]: f32) {
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[ALLOC:.+]] = gpu.alloc(%[[ARG0]], %[[ARG1]]) : memref<?x?xf32, 1>
// CHECK-NEXT:  scf.for %[[IV2:.+]] = %[[C0]] to %[[ARG0]] step %[[C1]] {
// CHECK-NEXT:    %[[SUBVIEW1:.+]] = memref.subview %[[ALLOC]][%[[IV2]], 0] [1, %[[ARG1]]] [1, 1] : memref<?x?xf32, 1> to memref<?xf32, strided<[1], offset: ?>, 1>
// CHECK-NEXT:    memref.store %[[VAL]], %[[SUBVIEW1]][%[[C0]]] : memref<?xf32, strided<[1], offset: ?>, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.for %[[IV_REV:.+]] = %[[C0]] to %[[ARG0]] step %[[C1]] {
// CHECK-NEXT:    %[[REV_IV:.+]] = arith.subi %[[ARG0]], %[[C1]] : index
// CHECK-NEXT:    %[[IDX:.+]] = arith.subi %[[REV_IV]], %[[IV_REV]] : index
// CHECK-NEXT:    %[[SUBVIEW2:.+]] = memref.subview %[[ALLOC]][%[[IDX]], 0] [1, %[[ARG1]]] [1, 1] : memref<?x?xf32, 1> to memref<?xf32, strided<[1], offset: ?>, 1>
// CHECK-NEXT:    %[[LD:.+]] = memref.load %[[SUBVIEW2]][%[[C0]]] : memref<?xf32, strided<[1], offset: ?>, 1>
// CHECK-NEXT:  }
// CHECK-NEXT:  gpu.dealloc %[[ALLOC]] : memref<?x?xf32, 1>
// CHECK-NEXT:  return
// CHECK-NEXT: }
