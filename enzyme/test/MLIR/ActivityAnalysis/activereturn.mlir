// RUN: %eopt --print-activity-analysis --split-input-file %s 2>&1 | FileCheck %s

// CHECK-LABEL: @activereturn
// CHECK:         "y": Active
// CHECK:         "result": Active
func.func @activereturn(%x: memref<f64>) -> memref<f64> {
    %y = memref.alloca() {tag = "y"} : memref<memref<f64>>
    memref.store %x, %y[] : memref<memref<f64>>
    %u = memref.load %y[] {tag = "result"} : memref<memref<f64>>
    return %u : memref<f64>
}

// CHECK-LABEL: @constreturn
// CHECK:         "y": Constant
// CHECK:         "result": Constant
func.func @constreturn(%x: memref<f64>) -> memref<f64> {
    %y = memref.alloca() {tag = "y"} : memref<memref<f64>>
    %u = memref.load %y[] {tag = "result"} : memref<memref<f64>>
    return %u : memref<f64>
}
