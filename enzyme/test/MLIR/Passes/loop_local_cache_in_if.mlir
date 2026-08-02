// RUN: %eopt %s --remove-unnecessary-enzyme-ops --split-input-file | FileCheck %s

// A cache whose init is inside the loop is that iteration's own: it is pushed
// and popped within one iteration, never carried to a reverse loop. The scf.if
// holding it must therefore keep removing it itself, rather than standing down
// (`preserve_cache`) in favour of the enclosing loop's remover, which pairs a
// forward loop with a reverse one and so has nothing to pair this with. The
// shape comes up when a differentiated function is inlined into a loop body.

module {
  func.func @loop_local(%cond: i1, %val: f32, %n: index, %out: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.parallel (%i) = (%c0) to (%n) step (%c1) {
      %cache = "enzyme.init"() : () -> !enzyme.Cache<f32>
      scf.if %cond {
        "enzyme.push"(%cache, %val) : (!enzyme.Cache<f32>, f32) -> ()
      }
      scf.if %cond {
        %p = "enzyme.pop"(%cache) : (!enzyme.Cache<f32>) -> f32
        memref.store %p, %out[%i] : memref<?xf32>
      }
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @loop_local(
// CHECK-NOT:     enzyme.init
// CHECK-NOT:     enzyme.push
// CHECK-NOT:     enzyme.pop
// CHECK-NOT:     preserve_cache

// -----

// The same cache initialized outside the loop is carried across it, so the ifs
// do stand down and leave it to the loop's remover.

module {
  func.func @loop_carried(%cond: i1, %val: f32, %n: index, %out: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cache = "enzyme.init"() : () -> !enzyme.Cache<f32>

    scf.parallel (%i) = (%c0) to (%n) step (%c1) {
      scf.if %cond {
        "enzyme.push"(%cache, %val) : (!enzyme.Cache<f32>, f32) -> ()
      }
      scf.reduce
    }
    scf.parallel (%i) = (%c0) to (%n) step (%c1) {
      scf.if %cond {
        %p = "enzyme.pop"(%cache) : (!enzyme.Cache<f32>) -> f32
        memref.store %p, %out[%i] : memref<?xf32>
      }
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @loop_carried(
// CHECK:         scf.if %{{.*}} {
// CHECK:         } {preserve_cache}

// -----

// A cache local to a *nested* loop is that loop's business: it must not count
// against annotating the if that happens to contain the loop, whose own cache
// (%outer) is carried across the parallel. Everything here is removable.

module {
  func.func @nested_loop(%cond: i1, %val: f32, %n: index, %out: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %outer = "enzyme.init"() : () -> !enzyme.Cache<f32>

    scf.parallel (%i) = (%c0) to (%n) step (%c1) {
      scf.if %cond {
        %inner = "enzyme.init"() : () -> !enzyme.Cache<f32>
        scf.for %j = %c0 to %n step %c1 {
          "enzyme.push"(%inner, %val) : (!enzyme.Cache<f32>, f32) -> ()
        }
        scf.for %j = %c0 to %n step %c1 {
          %p = "enzyme.pop"(%inner) : (!enzyme.Cache<f32>) -> f32
          memref.store %p, %out[%j] : memref<?xf32>
        }
        "enzyme.push"(%outer, %val) : (!enzyme.Cache<f32>, f32) -> ()
      }
      scf.reduce
    }
    scf.parallel (%i) = (%c0) to (%n) step (%c1) {
      scf.if %cond {
        %q = "enzyme.pop"(%outer) : (!enzyme.Cache<f32>) -> f32
        memref.store %q, %out[%i] : memref<?xf32>
      }
      scf.reduce
    }
    return
  }
}

// CHECK-LABEL: func.func @nested_loop(
// CHECK-NOT:     enzyme.init
// CHECK-NOT:     enzyme.push
// CHECK-NOT:     enzyme.pop
