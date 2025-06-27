// RUN: %eopt --print-activity-analysis='annotate' --split-input-file %s 2>&1 | FileCheck %s

// CHECK-LABEL: @activereturn
// CHECK:         "y": Active
// CHECK:         "z": Active
func.func @activereturn(%x: memref<f64> {enzyme.tag = "x"}) -> memref<f64> {
    %y = memref.alloca() {tag = "y"} : memref<memref<f64>>
    memref.store %x, %y[] : memref<memref<f64>>
    %z = memref.load %y[] {tag = "z"} : memref<memref<f64>>
    return %z : memref<f64>
}

// CHECK-LABEL: @constreturn
// CHECK:         "u": Constant
// CHECK:         "v": Constant
func.func @constreturn(%t: memref<f64>) -> memref<f64> {
    %u = memref.alloca() {tag = "u"} : memref<memref<f64>>
    %v = memref.load %u[] {tag = "v"} : memref<memref<f64>>
    return %v : memref<f64>
}
