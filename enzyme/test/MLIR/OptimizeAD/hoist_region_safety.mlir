// RUN: %eopt --split-input-file --hoist-enzyme-regions %s | FileCheck %s

// Unknown effects must reject motion even when every operand dominates.
// They must also prevent a following load from moving across the call.
func.func private @unknown()
func.func @unknown_effects(%x: f64, %dx: f64, %mem: memref<1xf64>) -> f64 {
  %zero = arith.constant 0 : index
  %d = enzyme.fwddiff_region(%x, %dx) {
  ^bb0(%y: f64):
    func.call @unknown() : () -> ()
    %v = memref.load %mem[%zero] : memref<1xf64>
    %r = arith.mulf %y, %v : f64
    enzyme.yield %r : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @unknown_effects
// CHECK: enzyme.fwddiff_region
// CHECK: func.call @unknown
// CHECK: memref.load
// CHECK: enzyme.yield

// -----

// The capture and active block argument share the same incoming value, but
// only the block argument is active. Hoisting must preserve this distinction
// without relying on region outlining or differentiation.
func.func @capture_reverse(%x: f64, %seed: f64) -> f64 {
  %d = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %three = arith.constant 3.0 : f64
    %scaled = arith.mulf %y, %three : f64
    %product = arith.mulf %scaled, %x : f64
    enzyme.yield %product : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @capture_reverse(
// CHECK-SAME: %[[R_X:.*]]: f64, %[[R_SEED:.*]]: f64)
// CHECK: %[[THREE:.*]] = arith.constant 3.000000e+00 : f64
// CHECK: enzyme.autodiff_region(%[[R_X]], %[[R_SEED]]) {
// CHECK: ^bb0(%[[R_Y:.*]]: f64):
// CHECK: %[[SCALED:.*]] = arith.mulf %[[R_Y]], %[[THREE]] : f64
// CHECK: %[[PRODUCT:.*]] = arith.mulf %[[SCALED]], %[[R_X]] : f64
// CHECK: enzyme.yield %[[PRODUCT]] : f64

// -----

// Repeated incoming values retain their explicit duplicated/constant roles.
// The constant block argument maps to its primal when the coefficient moves,
// while the two duplicated block arguments and the capture remain distinct.
func.func @capture_forward_repeated(%x: f64, %dy: f64, %dz: f64) -> f64 {
  %d = enzyme.fwddiff_region(%x, %dy, %x, %x, %dz) {
  ^bb0(%y: f64, %c: f64, %z: f64):
    %coefficient = arith.addf %c, %x : f64
    %left = arith.mulf %y, %x : f64
    %right = arith.mulf %z, %coefficient : f64
    %sum = arith.addf %left, %right : f64
    enzyme.yield %sum : f64
  } <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64, f64, f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @capture_forward_repeated(
// CHECK-SAME: %[[F_X:.*]]: f64, %[[F_DY:.*]]: f64, %[[F_DZ:.*]]: f64)
// CHECK: %[[COEFFICIENT:.*]] = arith.addf %[[F_X]], %[[F_X]] : f64
// CHECK: enzyme.fwddiff_region(%[[F_X]], %[[F_DY]], %[[F_X]], %[[F_X]], %[[F_DZ]]) {
// CHECK: ^bb0(%[[F_Y:.*]]: f64, %[[F_C:.*]]: f64, %[[F_Z:.*]]: f64):
// CHECK: %[[LEFT:.*]] = arith.mulf %[[F_Y]], %[[F_X]] : f64
// CHECK: %[[RIGHT:.*]] = arith.mulf %[[F_Z]], %[[COEFFICIENT]] : f64
// CHECK: %[[SUM:.*]] = arith.addf %[[LEFT]], %[[RIGHT]] : f64
// CHECK: enzyme.yield %[[SUM]] : f64
// CHECK: activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>]

// -----

// A nested region with no free values must not overwrite the rejection of
// the loop's active initial value. This previously produced invalid SSA.
func.func @active_loop_operand(%x: f64, %dx: f64, %n: index) -> f64 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %d = enzyme.fwddiff_region(%x, %dx) {
  ^bb0(%y: f64):
    %r = scf.for %i = %zero to %n step %one iter_args(%a = %y) -> (f64) {
      %v = arith.mulf %a, %a : f64
      scf.yield %v : f64
    }
    enzyme.yield %r : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @active_loop_operand
// CHECK: enzyme.fwddiff_region
// CHECK: ^bb0(%[[ACTIVE:.*]]: f64):
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[ACTIVE]])
// CHECK: enzyme.yield

// -----

// Nested captures are checked separately from the loop's own operands.
func.func @active_loop_capture(%x: f64, %seed: f64, %n: index) -> f64 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %initial = arith.constant 1.0 : f64
  %d = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%y: f64):
    %r = scf.for %i = %zero to %n step %one iter_args(%a = %initial) -> (f64) {
      %v = arith.mulf %a, %y : f64
      scf.yield %v : f64
    }
    enzyme.yield %r : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @active_loop_capture
// CHECK: enzyme.autodiff_region
// CHECK: ^bb0(%[[ACTIVE:.*]]: f64):
// CHECK: scf.for
// CHECK: arith.mulf %{{.*}}, %[[ACTIVE]]
// CHECK: enzyme.yield

// -----

// Recursively collected unknown effects must remain a rejection after the
// nested capture check succeeds.
func.func private @nested_unknown()
func.func @unknown_nested_effects(%x: f64, %seed: f64, %n: index) -> f64 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %d = enzyme.autodiff_region(%x, %seed) {
  ^bb0(%y: f64):
    scf.for %i = %zero to %n step %one {
      func.call @nested_unknown() : () -> ()
    }
    %r = arith.mulf %y, %y : f64
    enzyme.yield %r : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @unknown_nested_effects
// CHECK: enzyme.autodiff_region
// CHECK: scf.for
// CHECK: func.call @nested_unknown
// CHECK: enzyme.yield

// -----

// The view aliases the destination of a stationary active store.
func.func @conflicting_alias(%x: f64, %dx: f64, %mem: memref<2xf64>) -> f64 {
  %zero = arith.constant 0 : index
  %view = memref.subview %mem[0] [1] [1] : memref<2xf64> to memref<1xf64, strided<[1]>>
  %d = enzyme.fwddiff_region(%x, %dx) {
  ^bb0(%y: f64):
    memref.store %y, %mem[%zero] : memref<2xf64>
    %v = memref.load %view[%zero] : memref<1xf64, strided<[1]>>
    %r = arith.mulf %y, %v : f64
    enzyme.yield %r : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @conflicting_alias
// CHECK: enzyme.fwddiff_region
// CHECK: memref.store
// CHECK: memref.load
// CHECK: enzyme.yield

// -----

// A free effect conflicts with a stationary store to the same buffer.
// The buffer dominates the region, so operand checks alone permit motion.
func.func @dealloc_after_active_store(%x: f64, %dx: f64, %mem: memref<1xf64>) -> f64 {
  %zero = arith.constant 0 : index
  %d = enzyme.fwddiff_region(%x, %dx) {
  ^bb0(%y: f64):
    memref.store %y, %mem[%zero] : memref<1xf64>
    memref.dealloc %mem : memref<1xf64>
    enzyme.yield %y : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @dealloc_after_active_store(
// CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64, %[[MEM:.*]]: memref<1xf64>)
// CHECK: enzyme.fwddiff_region
// CHECK: ^bb0(%[[ACTIVE:.*]]: f64):
// CHECK: memref.store %[[ACTIVE]], %[[MEM]][%{{.*}}] : memref<1xf64>
// CHECK-NEXT: memref.dealloc %[[MEM]] : memref<1xf64>
// CHECK-NEXT: enzyme.yield %[[ACTIVE]] : f64

// -----

// Supported effectful motion: an independent entry-block load still moves.
func.func @independent_load(%x: f64, %dx: f64, %mem: memref<1xf64>) -> f64 {
  %zero = arith.constant 0 : index
  %d = enzyme.fwddiff_region(%x, %dx) {
  ^bb0(%y: f64):
    %v = memref.load %mem[%zero] : memref<1xf64>
    %r = arith.mulf %y, %v : f64
    enzyme.yield %r : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @independent_load
// CHECK: %[[LOAD:.*]] = memref.load
// CHECK: enzyme.fwddiff_region
// CHECK: arith.mulf %{{.*}}, %[[LOAD]]
// CHECK: enzyme.yield

// -----

// Hoisting an alloca changes its automatic allocation scope from the AD region
// to the function, extending its lifetime if the region executes repeatedly.
func.func @automatic_allocation_scope(%x: f64, %dx: f64) -> f64 {
  %zero = arith.constant 0 : index
  %d = enzyme.fwddiff_region(%x, %dx) {
  ^bb0(%y: f64):
    %mem = memref.alloca() : memref<1xf64>
    memref.store %y, %mem[%zero] : memref<1xf64>
    %r = memref.load %mem[%zero] : memref<1xf64>
    enzyme.yield %r : f64
  } <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
  return %d : f64
}
// CHECK-LABEL: func.func @automatic_allocation_scope
// CHECK: enzyme.fwddiff_region
// CHECK: memref.alloca
// CHECK: memref.store
// CHECK: memref.load
// CHECK: enzyme.yield
