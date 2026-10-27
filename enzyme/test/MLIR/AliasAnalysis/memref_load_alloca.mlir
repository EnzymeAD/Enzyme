// RUN: %eopt --test-print-alias-analysis --split-input-file %s 2>&1 | FileCheck %s

// Distinct stack allocations and heap memory do not alias an argument or each
// other. Keep memref.alloc coverage when simplifying the alloca checks.
// CHECK-DAG: "argument" and "stack_a": NoAlias
// CHECK-DAG: "argument" and "stack_b": NoAlias
// CHECK-DAG: "argument" and "heap": NoAlias
// CHECK-DAG: "stack_a" and "stack_b": NoAlias
// CHECK-DAG: "stack_a" and "heap": NoAlias
// CHECK-DAG: "stack_b" and "heap": NoAlias
// CHECK-DAG: "argument" and "heap_b": NoAlias
// CHECK-DAG: "stack_a" and "heap_b": NoAlias
// CHECK-DAG: "stack_b" and "heap_b": NoAlias
// CHECK-DAG: "heap" and "heap_b": NoAlias
func.func @independent_allocations(%arg: memref<f64> {enzyme.tag = "argument"}) {
  %a = memref.alloca() {tag = "stack_a"} : memref<f64>
  %b = memref.alloca() {tag = "stack_b"} : memref<f64>
  %heap = memref.alloc() {tag = "heap"} : memref<f64>
  %heap_b = memref.alloc() {tag = "heap_b"} : memref<f64>
  memref.dealloc %heap : memref<f64>
  memref.dealloc %heap_b : memref<f64>
  return
}

// -----

// Loading a stored reference preserves its alias relationship with the
// argument, while the stack slot holding that reference remains independent.
// CHECK-DAG: "source" and "loaded": MayAlias
// CHECK-DAG: "source" and "slot": NoAlias
// CHECK-DAG: "slot" and "loaded": NoAlias
func.func @loaded_reference(%source: memref<f64> {enzyme.tag = "source"}) -> memref<f64> {
  %slot = memref.alloca() {tag = "slot"} : memref<memref<f64>>
  memref.store %source, %slot[] : memref<memref<f64>>
  %loaded = memref.load %slot[] {tag = "loaded"} : memref<memref<f64>>
  return %loaded : memref<f64>
}

// -----

// The same stored-reference relationship must hold for affine accesses after
// replacing explicit load/store checks with LoadLike/StoreLike interfaces.
// CHECK-DAG: "affine_source" and "affine_loaded": MayAlias
// CHECK-DAG: "affine_source" and "affine_slot": NoAlias
// CHECK-DAG: "affine_slot" and "affine_loaded": NoAlias
func.func @affine_loaded_reference(%source: memref<f64> {enzyme.tag = "affine_source"}) -> memref<f64> {
  %slot = memref.alloca() {tag = "affine_slot"} : memref<1xmemref<f64>>
  affine.store %source, %slot[0] : memref<1xmemref<f64>>
  %loaded = affine.load %slot[0] {tag = "affine_loaded"} : memref<1xmemref<f64>>
  return %loaded : memref<f64>
}
