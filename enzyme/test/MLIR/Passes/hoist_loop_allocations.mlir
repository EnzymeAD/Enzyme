// RUN: %eopt %s --hoist-loop-allocations --split-input-file | FileCheck %s
// RUN: %eopt %s --hoist-loop-allocations=max-hoisted-bytes=64 --split-input-file | FileCheck %s --check-prefix=BUDGET

// A scratch buffer allocated and freed within one iteration is the same buffer
// every time around, so allocating it once is observationally identical.

func.func @hoists_scratch(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    %buf = memref.alloc() : memref<32xf32>
    memref.store %cst, %buf[%c0] : memref<32xf32>
    memref.dealloc %buf : memref<32xf32>
  }
  return
}

// CHECK-LABEL: func.func @hoists_scratch
// CHECK:         %[[BUF:.+]] = memref.alloc()
// CHECK-NEXT:    scf.for
// CHECK-NEXT:      memref.store %{{.+}}, %[[BUF]]
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[BUF]]

// 32xf32 is 128 bytes, over the budget.
// BUDGET-LABEL: func.func @hoists_scratch
// BUDGET:         scf.for
// BUDGET-NEXT:      memref.alloc()

// -----

func.func @hoists_affine(%n: index, %arg: memref<32xf32>) {
  %cst = arith.constant 1.0 : f32
  affine.for %i = 0 to 32 {
    %buf = memref.alloc() : memref<32xf32>
    affine.store %cst, %buf[%i] : memref<32xf32>
    %v = affine.load %buf[%i] : memref<32xf32>
    affine.store %v, %arg[%i] : memref<32xf32>
    memref.dealloc %buf : memref<32xf32>
  }
  return
}

// CHECK-LABEL: func.func @hoists_affine
// CHECK:         %[[BUF:.+]] = memref.alloc()
// CHECK-NEXT:    affine.for
// CHECK:         }
// CHECK-NEXT:    memref.dealloc %[[BUF]]

// -----

// A view stays in the body -- it is re-derived from the one hoisted buffer --
// but it does not block the hoist.

func.func @hoists_through_view(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    %buf = memref.alloc() : memref<32xf32>
    %view = memref.subview %buf[0] [16] [1] : memref<32xf32> to memref<16xf32>
    memref.store %cst, %view[%c0] : memref<16xf32>
    memref.dealloc %buf : memref<32xf32>
  }
  return
}

// CHECK-LABEL: func.func @hoists_through_view
// CHECK:         %[[BUF:.+]] = memref.alloc()
// CHECK-NEXT:    scf.for
// CHECK-NEXT:      memref.subview %[[BUF]]
// CHECK:         }
// CHECK-NEXT:    memref.dealloc %[[BUF]]

// -----

// A nest drains bottom-up: the inner hoist puts the pair in the outer body,
// where the outer loop's turn lifts it the rest of the way.

func.func @hoists_out_of_nest(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %buf = memref.alloc() : memref<32xf32>
      memref.store %cst, %buf[%c0] : memref<32xf32>
      memref.dealloc %buf : memref<32xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @hoists_out_of_nest
// CHECK:         %[[BUF:.+]] = memref.alloc()
// CHECK-NEXT:    scf.for
// CHECK-NEXT:      scf.for
// CHECK-NEXT:        memref.store %{{.+}}, %[[BUF]]
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.dealloc %[[BUF]]

// -----

// The iterations of a parallel loop would share the hoisted buffer.

func.func @skips_parallel(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %buf = memref.alloc() : memref<32xf32>
    memref.store %cst, %buf[%c0] : memref<32xf32>
    memref.dealloc %buf : memref<32xf32>
  }
  return
}

// CHECK-LABEL: func.func @skips_parallel
// CHECK:         scf.parallel
// CHECK-NEXT:      memref.alloc()

// -----

// The buffer is carried to the next iteration, so each iteration needs its own.

func.func @skips_escaping_buffer(%n: index, %init: memref<32xf32>) -> memref<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> memref<32xf32> {
    %buf = memref.alloc() : memref<32xf32>
    memref.copy %acc, %buf : memref<32xf32> to memref<32xf32>
    scf.yield %buf : memref<32xf32>
  }
  return %r : memref<32xf32>
}

// CHECK-LABEL: func.func @skips_escaping_buffer
// CHECK:         scf.for
// CHECK-NEXT:      memref.alloc()

// -----

// A dynamic size computed from the induction variable is not the same
// allocation before the loop as it was inside it.

func.func @skips_iv_dependent_size(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    %buf = memref.alloc(%i) : memref<?xf32>
    memref.store %cst, %buf[%c0] : memref<?xf32>
    memref.dealloc %buf : memref<?xf32>
  }
  return
}

// CHECK-LABEL: func.func @skips_iv_dependent_size
// CHECK:         scf.for
// CHECK-NEXT:      memref.alloc(

// -----

// Freed on only one path: hoisting would turn N allocations and one free into
// one allocation and one free, which is a different program.

func.func @skips_conditional_dealloc(%n: index, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    %buf = memref.alloc() : memref<32xf32>
    memref.store %cst, %buf[%c0] : memref<32xf32>
    scf.if %c {
      memref.dealloc %buf : memref<32xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @skips_conditional_dealloc
// CHECK:         scf.for
// CHECK-NEXT:      memref.alloc()

// -----

// A call could stash the pointer anywhere.

func.func private @capture(memref<32xf32>)

func.func @skips_call_use(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %buf = memref.alloc() : memref<32xf32>
    func.call @capture(%buf) : (memref<32xf32>) -> ()
    memref.dealloc %buf : memref<32xf32>
  }
  return
}

// CHECK-LABEL: func.func @skips_call_use
// CHECK:         scf.for
// CHECK-NEXT:      memref.alloc()
