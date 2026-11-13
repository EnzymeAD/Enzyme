// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// The adjoint of a floating cast passes the result's adjoint on to the source
// and then clears it. In a loop the same adjoint is reused by every iteration
// of the reverse sweep, so one left uncleared is passed on again by each later
// iteration.

!s32i = !cir.int<s, 32>
module {
  cir.func @g(%x: !cir.float) -> !cir.double {
    %c0 = cir.const #cir.int<0> : !s32i
    %c1 = cir.const #cir.int<1> : !s32i
    %c3 = cir.const #cir.int<3> : !s32i
    %zero = cir.const #cir.fp<0.0> : !cir.double
    cir.br ^h(%zero, %c0 : !cir.double, !s32i)
  ^h(%s: !cir.double, %i: !s32i):
    %cond = cir.cmp lt %i, %c3 : !s32i
    cir.brcond %cond ^body, ^exit
  ^body:
    %xd = cir.cast floating %x : !cir.float -> !cir.double
    %t = cir.sin %xd : !cir.double
    %s2 = cir.fadd %s, %t : !cir.double
    %i2 = cir.add %i, %c1 : !s32i
    cir.br ^h(%s2, %i2 : !cir.double, !s32i)
  ^exit:
    cir.return %s : !cir.double
  }
  cir.func @dg(%x: !cir.float, %dr: !cir.double) -> !cir.float {
    %r = enzyme.autodiff @g(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.float, !cir.double) -> !cir.float
    cir.return %r : !cir.float
  }
}

// CHECK-LABEL:   cir.func private @diffeg(
// CHECK-DAG:       %[[ZERO:.+]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// The reverse of the loop body ends with the adjoint of %xd = cir.cast.
// CHECK:           cir.cos
// CHECK:           "enzyme.set"
// CHECK-NEXT:      %[[DXD:.+]] = "enzyme.get"(%[[GXD:.+]]) : (!enzyme.Gradient<!cir.double>) -> !cir.double
// CHECK-NEXT:      "enzyme.set"(%[[GXD]], %[[ZERO]]) : (!enzyme.Gradient<!cir.double>, !cir.double) -> ()
// CHECK-NEXT:      %{{.+}} = cir.cast floating %[[DXD]] : !cir.double -> !cir.float
