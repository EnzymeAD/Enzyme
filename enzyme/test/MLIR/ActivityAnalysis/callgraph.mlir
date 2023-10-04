// RUN: %eopt --print-activity-analysis --split-input-file %s 2>&1 | FileCheck %s

// CHECK-LABEL: @callgraph_sparse:
// CHECK:         "callres": Active
func.func private @mysquare(%arg0: f64) -> f64 {
    %mul = arith.mulf %arg0, %arg0 : f64
    return %mul : f64
}

func.func @callgraph_sparse(%arg0: f64) -> f64 {
    %squared = call @mysquare(%arg0) {tag = "callres"} : (f64) -> f64
    return %squared : f64
}

// -----

func.func private @noop(%arg0: memref<f64>) {
    return
}

// CHECK-LABEL: @callgraph_dense:
// CHECK:         "sin": Active
func.func @callgraph_dense(%arg0: f64) -> f64 {
    %m = memref.alloc() : memref<f64>
    memref.store %arg0, %m[] : memref<f64>
    call @noop(%m) : (memref<f64>) -> ()
    %loaded = memref.load %m[] : memref<f64>
    %sin = math.sin %loaded {tag = "sin"} : f64
    return %sin : f64
}

// -----

func.func private @identity(%arg0: memref<f64>) -> memref<f64> {
    %space = memref.alloc() : memref<memref<f64>>
    memref.store %arg0, %space[] : memref<memref<f64>>
    %loaded = memref.load %space[] : memref<memref<f64>>
    memref.dealloc %space : memref<memref<f64>>
    return %loaded : memref<f64>
}

// CHECK-LABEL: @aliased_store:
// CHECK:         "retval": Active
func.func @aliased_store(%arg0: f64) -> f64 {
    %ptr = memref.alloc() : memref<f64>
    memref.store %arg0, %ptr[] : memref<f64>
    %new_ptr = call @identity(%ptr) : (memref<f64>) -> memref<f64>
    %val = memref.load %new_ptr[] {tag = "retval"} : memref<f64>
    return %val : f64
}

// -----

func.func private @write(%arg0: f64, %ptr: memref<f64>) {
    memref.store %arg0, %ptr[] : memref<f64>
    return
}

func.func private @read(%ptr: memref<f64>) -> f64 {
    %val = memref.load %ptr[] : memref<f64>
    return %val : f64
}

// CHECK-LABEL: @across_func_boundaries
// CHECK:         "retval": Active
func.func @across_func_boundaries(%arg0: f64) -> f64 {
    %ptr = memref.alloc() : memref<f64>
    call @write(%arg0, %ptr) : (f64, memref<f64>) -> ()
    %val = call @read(%ptr) {tag = "retval"} : (memref<f64>) -> f64
    return %val : f64
}

func.func private @no_write(%arg0: f64, %ptr: memref<f64>) {
    %cst = arith.constant 3.0 : f64
    memref.store %cst, %ptr[] : memref<f64>
    return
}
// CHECK-LABEL: @across_func_boundaries_const
// CHECK:         "retval": Constant
func.func @across_func_boundaries_const(%arg0: f64) -> f64 {
    %ptr = memref.alloc() : memref<f64>
    call @no_write(%arg0, %ptr) : (f64, memref<f64>) -> ()
    %val = call @read(%ptr) {tag = "retval"} : (memref<f64>) -> f64
    return %val : f64
}
