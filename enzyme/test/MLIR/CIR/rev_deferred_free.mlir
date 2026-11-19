// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s
// RUN: %eopt --enzyme --canonicalize --convert-enzyme-to-cir %s | FileCheck %s --check-prefix=LOWER

// The reverse sweep still uses the shadow of what the forward sweep frees,
// until it gets back to the allocation. When that allocation cannot be found
// from the free -- @make allocates what @f frees, and @g allocates what
// @consume frees -- the shadow goes on a list instead, which the outermost
// derivative frees once its reverse sweep is done. A split derivative is
// handed its caller's list as part of its tape.

!u64i = !cir.int<u, 64>
!void = !cir.void
module {
  cir.func private @malloc(!u64i) -> !cir.ptr<!void>
  cir.func private @free(!cir.ptr<!void>)

  cir.func @make(%x: !cir.double) -> !cir.ptr<!cir.double> {
    %n = cir.const #cir.int<8> : !u64i
    %m = cir.call @malloc(%n) : (!u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %m : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    cir.return %t : !cir.ptr<!cir.double>
  }
  cir.func @f(%x: !cir.double) -> !cir.double {
    %t = cir.call @make(%x) : (!cir.double) -> !cir.ptr<!cir.double>
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    %v = cir.cast bitcast %t : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    cir.call @free(%v) : (!cir.ptr<!void>) -> ()
    cir.return %y : !cir.double
  }

  cir.func @consume(%t: !cir.ptr<!cir.double>) -> !cir.double {
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    %v = cir.cast bitcast %t : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    cir.call @free(%v) : (!cir.ptr<!void>) -> ()
    cir.return %y : !cir.double
  }
  cir.func @g(%x: !cir.double) -> !cir.double {
    %n = cir.const #cir.int<8> : !u64i
    %m = cir.call @malloc(%n) : (!u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %m : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %r = cir.call @consume(%t) : (!cir.ptr<!cir.double>) -> !cir.double
    cir.return %r : !cir.double
  }

  cir.func @rev_f(%x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @f(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev_g(%x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @g(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// @f frees what @make allocated: the shadow goes on @f's own list before the
// primal free, as null if it is the primal itself.
// CHECK-LABEL: cir.func private @diffef(
// CHECK:         %[[LIST:.+]] = "enzyme.init"() {enzyme.deferred_free = "free"} : () -> !enzyme.Cache<!cir.ptr<!void>>
// CHECK:         %[[DV:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.double> -> !cir.ptr<!void>
// CHECK:         %[[V:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.double> -> !cir.ptr<!void>
// CHECK:         %[[SAME:.+]] = cir.cmp eq %[[DV]], %[[V]] : !cir.ptr<!void>
// CHECK:         %[[TOFREE:.+]] = cir.select if %[[SAME]] then %{{.+}} else %[[DV]]
// CHECK:         "enzyme.push"(%[[LIST]], %[[TOFREE]])
// CHECK-NEXT:    cir.call @free(%[[V]])

// @make's shadow is not freed where it is allocated: it is returned.
// CHECK-LABEL: cir.func private @augmented_make(
// CHECK-NOT:     cir.call @free
// CHECK:         cir.return

// @g hands @consume its own list as the last tape argument.
// CHECK-LABEL: cir.func private @diffeg(
// CHECK:         %[[GLIST:.+]] = "enzyme.init"() {enzyme.deferred_free = "free"} : () -> !enzyme.Cache<!cir.ptr<!void>>
// CHECK-NEXT:    %[[GTAPE:.+]] = builtin.unrealized_conversion_cast %[[GLIST]]
// CHECK:         cir.call @augmented_consume(%{{.+}}, %{{.+}}, %{{.+}}, %[[GTAPE]])

// @consume pushes onto it before its primal free.
// CHECK-LABEL: cir.func private @augmented_consume(
// CHECK-SAME:    %[[CTAPE:[^:]+]]: !cir.ptr<!void>) -> !cir.double
// CHECK:         %[[CLIST:.+]] = builtin.unrealized_conversion_cast %[[CTAPE]] : !cir.ptr<!void> to !enzyme.Cache<!cir.ptr<!void>>
// CHECK:         "enzyme.push"(%[[CLIST]], %{{.+}})
// CHECK-NEXT:    cir.call @free(

// The list's owner frees what is on it once its reverse sweep is done, then
// the list.
// LOWER-LABEL: cir.func internal private @__enzyme_cir_free_each(
// LOWER:         cir.call @free(
// LOWER-LABEL: cir.func private @diffef(
// LOWER:         cir.call @__enzyme_cir_free_each(%[[D:[^,]+]], %[[S:[^,]+]], %[[C:[^)]+]])
// LOWER-NEXT:    %[[B:.+]] = cir.load %[[D]]
// LOWER-NEXT:    %[[V:.+]] = cir.cast bitcast %[[B]]
// LOWER-NEXT:    cir.call @free(%[[V]])
// LOWER-NEXT:    cir.return
// LOWER-LABEL: cir.func private @augmented_consume(
// LOWER-NOT:     @__enzyme_cir_free_each
// LOWER:         cir.return
