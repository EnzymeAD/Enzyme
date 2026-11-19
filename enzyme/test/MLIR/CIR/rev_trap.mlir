// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// A path that ends in cir.trap never returns, so it never gets to the reverse
// sweep either: its block stays a trap. Only the returns lead into the
// reverse sweep, in a derivative that is split too.

!s64i = !cir.int<s, 64>
module {
  cir.func @checked(%x: !cir.double, %n: !s64i) -> !cir.double {
    %max = cir.const #cir.int<1000> : !s64i
    %big = cir.cmp gt %n, %max : !s64i
    cir.brcond %big ^trap, ^ok
  ^trap:
    cir.trap
  ^ok:
    %y = cir.fmul %x, %x : !cir.double
    cir.return %y : !cir.double
  }
  cir.func @fill(%p: !cir.ptr<!cir.double>, %x: !cir.double, %n: !s64i) {
    %max = cir.const #cir.int<1000> : !s64i
    %big = cir.cmp gt %n, %max : !s64i
    cir.brcond %big ^trap, ^ok
  ^trap:
    cir.trap
  ^ok:
    %y = cir.fmul %x, %x : !cir.double
    cir.store %y, %p : !cir.double, !cir.ptr<!cir.double>
    cir.return
  }
  cir.func @g(%x: !cir.double, %n: !s64i) -> !cir.double {
    %t = cir.alloca "t" align(8) : !cir.ptr<!cir.double>
    cir.call @fill(%t, %x, %n) : (!cir.ptr<!cir.double>, !cir.double, !s64i) -> ()
    %r = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev(%x: !cir.double, %n: !s64i, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @checked(%x, %n, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !s64i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev_g(%x: !cir.double, %n: !s64i, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @g(%x, %n, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !s64i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL: cir.func private @diffechecked(
// CHECK:         cir.brcond %{{.+}} ^[[TRAP:.+]], ^[[OK:[^ ]+]]
// CHECK-NEXT:  ^[[TRAP]]:
// CHECK-NEXT:    cir.trap
// CHECK-NEXT:  ^[[OK]]:
// CHECK-NOT:     cir.trap
// CHECK:         cir.return

// @fill's forward sweep has one exit, its return; the trap stays a trap.
// CHECK-LABEL: cir.func private @augmented_fill(
// CHECK:         cir.brcond %{{.+}} ^[[FTRAP:.+]], ^[[FOK:[^ ]+]]
// CHECK-NEXT:  ^[[FTRAP]]:
// CHECK-NEXT:    cir.trap
// CHECK-NEXT:  ^[[FOK]]:
// CHECK-NOT:     cir.trap
// CHECK:         cir.store
// CHECK-NEXT:    cir.return

// CHECK-LABEL: cir.func private @diffefill(
// CHECK-NOT:     cir.trap
// CHECK:         cir.return
