// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s
// RUN: %eopt --enzyme --canonicalize --convert-enzyme-to-cir %s | FileCheck %s --check-prefix=LOWER

// A deferred free of what an aligned operator new allocated is made with the
// unsized aligned operator delete, which needs the alignment again: it is
// part of the list's key, and passed when the list is freed. The sized
// delete @consume calls frees with the same alignment.

!u64i = !cir.int<u, 64>
!void = !cir.void
module {
  cir.func private @_ZnwmSt11align_val_t(!u64i, !u64i) -> !cir.ptr<!void>
  cir.func private @_ZdlPvmSt11align_val_t(!cir.ptr<!void>, !u64i, !u64i)

  cir.func @consume(%t: !cir.ptr<!cir.double>) -> !cir.double {
    %n = cir.const #cir.int<64> : !u64i
    %al = cir.const #cir.int<64> : !u64i
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    %v = cir.cast bitcast %t : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    cir.call @_ZdlPvmSt11align_val_t(%v, %n, %al) : (!cir.ptr<!void>, !u64i, !u64i) -> ()
    cir.return %y : !cir.double
  }
  cir.func @g(%x: !cir.double) -> !cir.double {
    %n = cir.const #cir.int<64> : !u64i
    %al = cir.const #cir.int<64> : !u64i
    %m = cir.call @_ZnwmSt11align_val_t(%n, %al) : (!u64i, !u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %m : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %r = cir.call @consume(%t) : (!cir.ptr<!cir.double>) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev_g(%x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @g(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL: cir.func private @diffeg(
// CHECK:         %[[LIST:.+]] = "enzyme.init"() {enzyme.deferred_free = "_ZdlPvSt11align_val_t:64"} : () -> !enzyme.Cache<!cir.ptr<!void>>
// CHECK:         %[[TAPE:.+]] = builtin.unrealized_conversion_cast %[[LIST]]
// CHECK:         cir.call @augmented_consume(%{{.+}}, %{{.+}}, %{{.+}}, %[[TAPE]])
// CHECK-LABEL: cir.func private @augmented_consume(
// CHECK-SAME:    %[[LTAPE:[^:]+]]: !cir.ptr<!void>) -> !cir.double
// CHECK:         %[[CLIST:.+]] = builtin.unrealized_conversion_cast %[[LTAPE]] : !cir.ptr<!void> to !enzyme.Cache<!cir.ptr<!void>>
// CHECK:         "enzyme.push"(%[[CLIST]], %{{.+}})
// CHECK-NEXT:    cir.call @_ZdlPvmSt11align_val_t(

// LOWER-LABEL:   cir.func internal private @__enzyme_cir_free_each(
// LOWER:           %[[AL:.+]] = cir.const #cir.int<64> : !u64i
// LOWER:           cir.call @_ZdlPvSt11align_val_t(%{{.+}}, %[[AL]]) : (!cir.ptr<!void>, !u64i) -> ()
// LOWER:         cir.func private @_ZdlPvSt11align_val_t(!cir.ptr<!void>, !u64i)
// LOWER-LABEL:   cir.func private @diffeg(
// LOWER:           cir.call @__enzyme_cir_free_each(
// LOWER:           cir.return
