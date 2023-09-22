// RUN: %eopt --print-activity-analysis %s --split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: @ifthen:
// CHECK:         "x": Active
// CHECK:         "mem": Active
// CHECK:         "load0": Active
// CHECK:         "ifres": Active
func.func @ifthen(%x: f64 {enzyme.tag = "x"}, %p: i1) -> f64 {
    %m = memref.alloc() {tag = "mem"} : memref<f64>
    scf.if %p {
        memref.store %x, %m[] : memref<f64>
    }

    %res = scf.if %p -> f64 {
        %load = memref.load %m[] {tag = "load0"} : memref<f64>
        scf.yield %load : f64
    } else {
        %cst = arith.constant 0.0 : f64
        scf.yield %cst : f64
    } {tag = "ifres"}
    memref.dealloc %m : memref<f64>
    return %res : f64
}

// -----

// CHECK-LABEL: @forloop
// CHECK:         "x": Active
// CHECK:         "inner_res": Active
// CHECK:         "outer_res": Active
func.func @forloop(%x: memref<?xf64> {enzyme.tag = "x"}) -> f64 {
    %cst = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %c1 = arith.constant 1 : index
    %sum = scf.for %iv = %c0 to %dim step %c1 iter_args(%it = %cst) -> f64 {
        %xi = memref.load %x[%iv] : memref<?xf64>
        %inner_res = affine.for %jv = 0 to 3 iter_args(%j_it = %it) -> f64 {
            %j_it_next = arith.addf %j_it, %xi : f64
            affine.yield %j_it_next : f64
        } {tag = "inner_res"}
        scf.yield %inner_res : f64
    } {tag = "outer_res"}
    return %sum : f64
}

// CHECK-LABEL: @forloop_const
// CHECK:         "outer_res": Constant
func.func @forloop_const(%x: memref<?xf64> {enzyme.tag = "x"}) -> f64 {
    %cst = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %c1 = arith.constant 1 : index
    %sum = scf.for %iv = %c0 to %dim step %c1 iter_args(%it = %cst) -> f64 {
        %xi = memref.load %x[%iv] : memref<?xf64>
        %inner_res = affine.for %jv = 0 to 3 iter_args(%j_it = %it) -> f64 {
            %j_it_next = arith.addf %j_it, %xi : f64
            affine.yield %j_it_next : f64
        }
        scf.yield %cst : f64
    } {tag = "outer_res"}
    return %sum : f64
}
