// RUN: %eopt --enzyme %s | FileCheck %s

// After mem2reg a loop that walks a pointer carries it as a block argument.
// In reverse mode only a pointer gets a shadow block argument, and the
// forward pass has to pass that shadow along every edge into the loop.
!s32i = !cir.int<s, 32>
module {
  cir.func @walk(%p: !cir.ptr<!cir.double>, %n: !s32i) -> !cir.double {
    %zero = cir.const #cir.int<0> : !s32i
    %one = cir.const #cir.int<1> : !s32i
    %fz = cir.const #cir.fp<0.000000e+00> : !cir.double
    cir.br ^bb1(%fz, %p, %zero : !cir.double, !cir.ptr<!cir.double>, !s32i)
  ^bb1(%s: !cir.double, %q: !cir.ptr<!cir.double>, %i: !s32i):
    %c = cir.cmp lt %i, %n : !s32i
    cir.brcond %c ^bb2, ^bb3
  ^bb2:
    %v = cir.load %q : !cir.ptr<!cir.double>, !cir.double
    %s2 = cir.fadd %s, %v : !cir.double
    %q2 = cir.ptr_stride %q, %one : (!cir.ptr<!cir.double>, !s32i) -> !cir.ptr<!cir.double>
    %i2 = cir.inc %i : !s32i
    cir.br ^bb1(%s2, %q2, %i2 : !cir.double, !cir.ptr<!cir.double>, !s32i)
  ^bb3:
    cir.return %s : !cir.double
  }
  cir.func @dwalk(%p: !cir.ptr<!cir.double>, %dp: !cir.ptr<!cir.double>, %n: !s32i, %seed: !cir.double) {
    enzyme.autodiff @walk(%p, %dp, %n, %seed) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !s32i, !cir.double) -> ()
    cir.return
  }
}

// The float sum gets no shadow argument; the pointer's shadow follows it, and
// both edges into the loop pass the shadow: the entry passes %dp, the back
// edge the shadow of %q2.
// CHECK-LABEL: cir.func private @diffewalk(
// CHECK-SAME:      %[[P:[a-zA-Z0-9$._-]+]]: !cir.ptr<!cir.double>, %[[DP:[a-zA-Z0-9$._-]+]]: !cir.ptr<!cir.double>,
// CHECK:         cir.br ^bb1(%{{.+}}, %[[P]], %[[DP]], %{{.+}} : !cir.double, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !s32i)
// CHECK:       ^bb1(%{{.+}}: !cir.double, %[[Q:.+]]: !cir.ptr<!cir.double>, %[[DQ:.+]]: !cir.ptr<!cir.double>, %{{.+}}: !s32i):
// CHECK:         %[[DQ2:.+]] = cir.ptr_stride %[[DQ]],
// CHECK:         %[[Q2:.+]] = cir.ptr_stride %[[Q]],
// CHECK:         cir.br ^bb1(%{{.+}}, %[[Q2]], %[[DQ2]], %{{.+}} : !cir.double, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !s32i)
