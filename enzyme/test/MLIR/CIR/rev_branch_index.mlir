// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// The reverse sweep's control flow is built in CIR: the branch index is a
// !s32i, and the derivative branches on it with cir.switch.flat, so it lowers
// with the rest of the function. The adjoint of the block argument flows back
// along the edge control came in on, selected by the index that edge's
// predecessor recorded: d/dx is 2 through ^a and 3 through ^b.

!s32i = !cir.int<s, 32>
module {
  cir.func @f(%x: !cir.double, %c: !cir.bool) -> !cir.double {
    cir.brcond %c ^a, ^b
  ^a:
    %two = cir.const #cir.fp<2.0> : !cir.double
    %xa = cir.fmul %x, %two : !cir.double
    cir.br ^join(%xa : !cir.double)
  ^b:
    %three = cir.const #cir.fp<3.0> : !cir.double
    %xb = cir.fmul %x, %three : !cir.double
    cir.br ^join(%xb : !cir.double)
  ^join(%v: !cir.double):
    cir.return %v : !cir.double
  }
  cir.func @df(%x: !cir.double, %c: !cir.bool, %dr: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @f(%x, %c, %dr) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.bool, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL:   cir.func private @diffef(
// CHECK-SAME:      %[[X:.+]]: !cir.double, %[[C:.+]]: !cir.bool, %[[DR:.+]]: !cir.double) -> !cir.double {
// CHECK-NOT:       {{(arith|cf)\.}}
// CHECK-DAG:       %[[IDX_A:.+]] = cir.const #cir.int<0> : !s32i
// CHECK-DAG:       %[[IDX_B:.+]] = cir.const #cir.int<-1> : !s32i
// CHECK-DAG:       %[[ZERO:.+]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CHECK:           %[[FLAGS:.+]] = "enzyme.init"() : () -> !enzyme.Cache<!s32i>
// CHECK:           cir.brcond %[[C]] ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           "enzyme.push"(%[[FLAGS]], %[[IDX_A]]) : (!enzyme.Cache<!s32i>, !s32i) -> ()
// CHECK-NEXT:      cir.br ^bb3
// CHECK:         ^bb2:
// CHECK:           "enzyme.push"(%[[FLAGS]], %[[IDX_B]]) : (!enzyme.Cache<!s32i>, !s32i) -> ()
// CHECK-NEXT:      cir.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[FLAG:.+]] = "enzyme.pop"(%[[FLAGS]]) : (!enzyme.Cache<!s32i>) -> !s32i
// CHECK:           %[[DV:.+]] = "enzyme.get"
// CHECK:           %[[IS_B:.+]] = cir.cmp eq %[[FLAG]], %[[IDX_B]] : !s32i
// CHECK-NEXT:      %[[D_B:.+]] = cir.select if %[[IS_B]] then %[[DV]] else %[[ZERO]]
// CHECK:           %[[IS_A:.+]] = cir.cmp eq %[[FLAG]], %[[IDX_A]] : !s32i
// CHECK-NEXT:      %[[D_A:.+]] = cir.select if %[[IS_A]] then %[[DV]] else %[[ZERO]]
// CHECK:           cir.switch.flat %[[FLAG]] : !s32i, ^bb4 [
// CHECK-NEXT:        0: ^bb5
// CHECK-NEXT:      ]
// CHECK:         ^bb4:
// CHECK:           "enzyme.pop"
// CHECK:           cir.fmul
// CHECK:           cir.br ^bb6
// CHECK:         ^bb5:
// CHECK:           "enzyme.pop"
// CHECK:           cir.fmul
// CHECK:           cir.br ^bb6
// CHECK:         ^bb6:
// CHECK:           cir.return
// CHECK-NOT:       {{(arith|cf)\.}}
