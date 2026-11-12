// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize %s | FileCheck %s

// memcpy(dst, src, n) of doubles and of double pointers, as clang emits it
// through void pointers. Tangents and shadow pointers are bytes alike, so a
// copy of the shadows carries both; from an undifferentiated source, float
// bytes get a zero tangent and pointer bytes keep the primal.

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
  cir.func @fwd(%dst: !cir.ptr<!cir.double>, %ddst: !cir.ptr<!cir.double>, %src: !cir.ptr<!cir.double>, %dsrc: !cir.ptr<!cir.double>, %n: !u64i) {
    enzyme.fwddiff @copy(%dst, %ddst, %src, %dsrc, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !u64i) -> ()
    cir.return
  }
  cir.func @fwd_const_src(%dst: !cir.ptr<!cir.double>, %ddst: !cir.ptr<!cir.double>, %src: !cir.ptr<!cir.double>, %n: !u64i) {
    enzyme.fwddiff @copy(%dst, %ddst, %src, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !u64i) -> ()
    cir.return
  }
  cir.func @fwd_ptrs_const_src(%dst: !cir.ptr<!cir.ptr<!cir.double>>, %ddst: !cir.ptr<!cir.ptr<!cir.double>>, %src: !cir.ptr<!cir.ptr<!cir.double>>, %n: !u64i) {
    enzyme.fwddiff @copy_ptrs(%dst, %ddst, %src, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity=[] }> : (!cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.ptr<!cir.double>>, !cir.ptr<!cir.ptr<!cir.double>>, !u64i) -> ()
    cir.return
  }
}

// CHECK-LABEL: cir.func private @fwddiffecopy(
// CHECK-SAME:      %[[DST:[^:]+]]: !cir.ptr<!cir.double>, %[[DDST:[^:]+]]: !cir.ptr<!cir.double>, %[[SRC:[^:]+]]: !cir.ptr<!cir.double>, %[[DSRC:[^:]+]]: !cir.ptr<!cir.double>, %[[N:[^:]+]]: !u64i)
// CHECK-DAG:     %[[DD:.*]] = cir.cast bitcast %[[DDST]]
// CHECK-DAG:     %[[DS:.*]] = cir.cast bitcast %[[DSRC]]
// CHECK:         cir.libc.memcpy %[[N]] bytes from %[[DS]] to %[[DD]]

// CHECK-LABEL: cir.func private @fwddiffecopy_0(
// CHECK-SAME:      %[[DST:[^:]+]]: !cir.ptr<!cir.double>, %[[DDST:[^:]+]]: !cir.ptr<!cir.double>, %[[SRC:[^:]+]]: !cir.ptr<!cir.double>, %[[N:[^:]+]]: !u64i)
// CHECK-DAG:     %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CHECK-DAG:     %[[DD:.*]] = cir.cast bitcast %[[DDST]]
// CHECK:         cir.libc.memset %[[N]] bytes at %[[DD]] to %[[ZERO]]

// CHECK-LABEL: cir.func private @fwddiffecopy_ptrs(
// CHECK-SAME:      %[[DST:[^:]+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[DDST:[^:]+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[SRC:[^:]+]]: !cir.ptr<!cir.ptr<!cir.double>>, %[[N:[^:]+]]: !u64i)
// CHECK-DAG:     %[[DD:.*]] = cir.cast bitcast %[[DDST]]
// CHECK-DAG:     %[[S:.*]] = cir.cast bitcast %[[SRC]]
// CHECK:         cir.libc.memcpy %[[N]] bytes from %[[S]] to %[[DD]]
