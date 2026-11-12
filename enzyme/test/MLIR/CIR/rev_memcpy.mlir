// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize %s | FileCheck %s

// The adjoint of memcpy(dst, src, n) of doubles is dsrc[i] += ddst[i];
// ddst[i] = 0 over n bytes. The forward sweep must not copy the shadows: ddst
// holds the adjoint seed by then. Pointer bytes have no adjoint; their
// shadows are copied in the forward sweep instead.

!u64i = !cir.int<u, 64>
!void = !cir.void
module {
  cir.func @copy(%dst: !cir.ptr<!cir.double>, %src: !cir.ptr<!cir.double>, %n: !u64i) {
    %d = cir.cast bitcast %dst : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    %s = cir.cast bitcast %src : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    cir.libc.memcpy %n bytes from %s to %d : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
    cir.return
  }
  cir.func @copy_ptrs(%dst: !cir.ptr<!cir.ptr<!cir.double>>, %src: !cir.ptr<!cir.ptr<!cir.double>>, %n: !u64i) {
    %d = cir.cast bitcast %dst : !cir.ptr<!cir.ptr<!cir.double>> -> !cir.ptr<!void>
    %s = cir.cast bitcast %src : !cir.ptr<!cir.ptr<!cir.double>> -> !cir.ptr<!void>
    cir.libc.memcpy %n bytes from %s to %d : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
    cir.return
  }
  cir.func @rev(%dst: !cir.ptr<!cir.double>, %ddst: !cir.ptr<!cir.double>, %src: !cir.ptr<!cir.double>, %dsrc: !cir.ptr<!cir.double>, %n: !u64i) {
    enzyme.autodiff @copy(%dst, %ddst, %src, %dsrc, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !u64i) -> ()
    cir.return
  }
  cir.func @rev_const_src(%dst: !cir.ptr<!cir.double>, %ddst: !cir.ptr<!cir.double>, %src: !cir.ptr<!cir.double>, %n: !u64i) {
    enzyme.autodiff @copy(%dst, %ddst, %src, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !u64i) -> ()
    cir.return
  }
  cir.func @rev_ptrs(%dst: !cir.ptr<!cir.ptr<!cir.double>>, %ddst: !cir.ptr<!cir.ptr<!cir.double>>, %src: !cir.ptr<!cir.ptr<!cir.double>>, %dsrc: !cir.ptr<!cir.ptr<!cir.double>>, %n: !u64i) {
    enzyme.autodiff @copy_ptrs(%dst, %ddst, %src, %dsrc, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.ptr<!cir.double>>, !u64i) -> ()
    cir.return
  }
}

// CHECK-LABEL: cir.func private @diffecopy(
// CHECK-SAME:      %[[DST:[^:]+]]: !cir.ptr<!cir.double>, %[[DDST:[^:]+]]: !cir.ptr<!cir.double>, %[[SRC:[^:]+]]: !cir.ptr<!cir.double>, %[[DSRC:[^:]+]]: !cir.ptr<!cir.double>, %[[N:[^:]+]]: !u64i)
// CHECK-DAG:     %[[ZERO:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CHECK-DAG:     %[[EIGHT:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-DAG:     %[[IDX:.*]] = cir.alloca "memcpy.idx"
// CHECK:         cir.libc.memcpy %[[N]] bytes
// CHECK-NOT:     cir.libc.memcpy
// CHECK:         %[[COUNT:.*]] = cir.div %[[N]], %[[EIGHT]] : !u64i
// CHECK:         cir.for : cond {
// CHECK:           %[[I:.*]] = cir.load %[[IDX]]
// CHECK:           cir.cmp lt %[[I]], %[[COUNT]]
// CHECK:         } body {
// CHECK:           %[[D:.*]] = cir.ptr_stride
// CHECK:           %[[S:.*]] = cir.ptr_stride
// CHECK:           %[[DADJ:.*]] = cir.load %[[D]]
// CHECK:           %[[SADJ:.*]] = cir.load %[[S]]
// CHECK:           %[[SUM:.*]] = cir.fadd %[[SADJ]], %[[DADJ]]
// CHECK:           cir.store %[[SUM]], %[[S]]
// CHECK:           cir.store %[[ZERO]], %[[D]]
// CHECK:         } step {

// CHECK-LABEL: cir.func private @diffecopy_0(
// CHECK-SAME:      %[[DST:[^:]+]]: !cir.ptr<!cir.double>, %[[DDST:[^:]+]]: !cir.ptr<!cir.double>, %[[SRC:[^:]+]]: !cir.ptr<!cir.double>, %[[N:[^:]+]]: !u64i)
// CHECK-DAG:     %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CHECK-DAG:     %[[DD:.*]] = cir.cast bitcast %[[DDST]]
// CHECK:         cir.libc.memset %[[N]] bytes at %[[DD]] to %[[ZERO]]
// CHECK-NOT:     cir.for

// CHECK-LABEL: cir.func private @diffecopy_ptrs(
// CHECK-SAME:      %[[DST:[^:]+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[DDST:[^:]+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[SRC:[^:]+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[DSRC:[^:]+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[N:[^:]+]]: !u64i)
// CHECK-DAG:     %[[DD:.*]] = cir.cast bitcast %[[DDST]]
// CHECK-DAG:     %[[DS:.*]] = cir.cast bitcast %[[DSRC]]
// CHECK:         cir.libc.memcpy %[[N]] bytes from %[[DS]] to %[[DD]]
// CHECK-NOT:     cir.for
// CHECK:         cir.return
