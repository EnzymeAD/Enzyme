// RUN: %eopt --print-activity-analysis='dataflow=false annotate=true' %s 2>&1 | FileCheck %s

// A copy from memory nothing differentiates transfers no derivative, whether
// it is a cir.copy or a cir.libc.memcpy through void pointers.

!u64i = !cir.int<u, 64>
!void = !cir.void
!rec_S = !cir.struct<"S" {data !cir.double, data !cir.double}>
module {
  // CHECK-LABEL: cir.func @copy_const_src
  // CHECK:         cir.copy %{{.*}} to %{{.*}} {enzyme.ici = true} : !cir.ptr<!cir.double>
  cir.func @copy_const_src(%p: !cir.ptr<!cir.double>, %c: !cir.ptr<!cir.double> {enzyme.const}) {
    cir.copy %c to %p : !cir.ptr<!cir.double>
    cir.return
  }

  // CHECK-LABEL: cir.func @copy_struct_const_src
  // CHECK:         cir.copy %{{.*}} to %{{.*}} {enzyme.ici = true} : !cir.ptr<{{.*}}>
  cir.func @copy_struct_const_src(%p: !cir.ptr<!rec_S>, %c: !cir.ptr<!rec_S> {enzyme.const}) {
    cir.copy %c to %p : !cir.ptr<!rec_S>
    cir.return
  }

  // CHECK-LABEL: cir.func @memcpy_const_src
  // CHECK:         cir.libc.memcpy %{{.*}} bytes from %{{.*}} to %{{.*}} {enzyme.ici = true}
  cir.func @memcpy_const_src(%p: !cir.ptr<!cir.double>, %c: !cir.ptr<!cir.double> {enzyme.const}) {
    %len = cir.const #cir.int<8> : !u64i
    %d = cir.cast bitcast %p : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    %s = cir.cast bitcast %c : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    cir.libc.memcpy %len bytes from %s to %d : !u64i, !cir.ptr<!void> -> !cir.ptr<!void>
    cir.return
  }

  // Both sides active: the copy carries a derivative.
  // CHECK-LABEL: cir.func @copy_active
  // CHECK:         cir.copy %{{.*}} to %{{.*}} {enzyme.ici = false} : !cir.ptr<!cir.double>
  cir.func @copy_active(%p: !cir.ptr<!cir.double>, %q: !cir.ptr<!cir.double>) {
    cir.copy %q to %p : !cir.ptr<!cir.double>
    cir.return
  }
}
