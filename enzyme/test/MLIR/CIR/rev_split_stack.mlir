// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// A split derivative's forward sweep runs in @augmented_<f>, whose frame is
// gone by the time @diffe<f> runs. @sq's stack slot %t, and its shadow, are
// still used then: the reverse sweep accumulates into the shadow through the
// pointer it cached. Both are allocated on the heap instead, and freed when
// the reverse sweep gets back to the slot.

module {
  cir.func @sq(%x: !cir.ptr<!cir.double>, %out: !cir.ptr<!cir.double>) {
    %t = cir.alloca "t" align(8) : !cir.ptr<!cir.double>
    %v = cir.load %x : !cir.ptr<!cir.double>, !cir.double
    %m = cir.fmul %v, %v : !cir.double
    cir.store %m, %t : !cir.double, !cir.ptr<!cir.double>
    %r = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.store %r, %out : !cir.double, !cir.ptr<!cir.double>
    cir.return
  }
  cir.func @f(%x: !cir.ptr<!cir.double>) -> !cir.double {
    %o = cir.alloca "o" align(8) : !cir.ptr<!cir.double>
    cir.call @sq(%x, %o) : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>) -> ()
    %r = cir.load %o : !cir.ptr<!cir.double>, !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev(%x: !cir.ptr<!cir.double>, %dx: !cir.ptr<!cir.double>, %seed: !cir.double) {
    enzyme.autodiff @f(%x, %dx, %seed) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.double) -> ()
    cir.return
  }
}

// The slot and its shadow, zeroed, come from malloc; both go on the tape.
// CHECK-LABEL: cir.func private @augmented_sq(
// CHECK-NOT:     cir.alloca
// CHECK:         %[[T:.+]] = cir.call @malloc(%[[N:.+]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:         "enzyme.push"(%{{.+}}, %[[T]])
// CHECK:         %[[DT:.+]] = cir.call @malloc(%[[N]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:         cir.libc.memset %[[N]] bytes at %[[DT]]
// CHECK:         "enzyme.push"(%{{.+}}, %[[DT]])
// CHECK-NOT:     cir.call @free
// CHECK:         cir.return

// The reverse sweep frees them once it is done with the slot.
// CHECK-LABEL: cir.func private @diffesq(
// CHECK-NOT:     cir.call @free
// CHECK:         cir.store %{{.+}}, %{{.+}} : !cir.double, !cir.ptr<!cir.double>
// CHECK:         %[[TP:.+]] = "enzyme.pop"(%{{.+}}) : (!enzyme.Cache<!cir.ptr<!void>>) -> !cir.ptr<!void>
// CHECK-NEXT:    cir.call @free(%[[TP]])
// CHECK-NEXT:    %[[DTP:.+]] = "enzyme.pop"(%{{.+}}) : (!enzyme.Cache<!cir.ptr<!void>>) -> !cir.ptr<!void>
// CHECK-NEXT:    cir.call @free(%[[DTP]])
// CHECK-NEXT:    cir.return
