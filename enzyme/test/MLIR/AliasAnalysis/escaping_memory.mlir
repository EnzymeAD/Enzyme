// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

// CHECK-DAG: "x" and "result": MayAlias
// CHECK-DAG: "x" and "y": NoAlias
// CHECK-DAG: "y" and "result": NoAlias
func.func @activereturn(%x: memref<f64> {enzyme.tag = "x"}) -> memref<f64> {
    %y = memref.alloca() {tag = "y"} : memref<memref<f64>>
    memref.store %x, %y[] : memref<memref<f64>>
    %u = memref.load %y[] {tag = "result"} : memref<memref<f64>>
    return %u : memref<f64>
}
