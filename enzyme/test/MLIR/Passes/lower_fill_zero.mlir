// RUN: %eopt --convert-enzyme-to-memref --split-input-file %s | FileCheck %s

func.func @scalar(%m: memref<f64>) {
  enzyme.fill_zero %m : memref<f64>
  return
}

// CHECK-LABEL: func.func @scalar(
// CHECK-SAME:      %[[M:.+]]: memref<f64>)
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    memref.store %[[ZERO]], %[[M]][] : memref<f64>
// CHECK-NEXT:    return

// -----

func.func @dynamic(%m: memref<?x3xf32>) {
  enzyme.fill_zero %m : memref<?x3xf32>
  return
}

// CHECK-LABEL: func.func @dynamic(
// CHECK-SAME:      %[[M:.+]]: memref<?x3xf32>)
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[D0:.+]] = memref.dim %[[M]], %{{.+}} : memref<?x3xf32>
// CHECK:         scf.for %[[I:.+]] = %{{.+}} to %[[D0]] step %{{.+}} {
// CHECK-NEXT:      scf.for %[[J:.+]] = %{{.+}} to %[[C3]] step %{{.+}} {
// CHECK-NEXT:        memref.store %[[ZERO]], %[[M]][%[[I]], %[[J]]] : memref<?x3xf32>
