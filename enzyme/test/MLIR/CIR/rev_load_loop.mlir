// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// The adjoint of a load adds the loaded value's adjoint into shadow memory and
// then clears it. In a loop the same adjoint is reused by every iteration of
// the reverse sweep, so one left uncleared is added again by each later
// iteration.

!s32i = !cir.int<s, 32>
module {
  cir.func @g(%x: !cir.double) -> !cir.double {
    %sp = cir.alloca "s" align(8) : !cir.ptr<!cir.double>
    %c0 = cir.const #cir.int<0> : !s32i
    %c1 = cir.const #cir.int<1> : !s32i
    %c3 = cir.const #cir.int<3> : !s32i
    %one = cir.const #cir.fp<1.0> : !cir.double
    cir.store %one, %sp : !cir.double, !cir.ptr<!cir.double>
    cir.br ^h(%c0 : !s32i)
  ^h(%i: !s32i):
    %cond = cir.cmp lt %i, %c3 : !s32i
    cir.brcond %cond ^body, ^exit
  ^body:
    %s = cir.load %sp : !cir.ptr<!cir.double>, !cir.double
    %sx = cir.sin %x : !cir.double
    %t = cir.fmul %s, %sx : !cir.double
    %s2 = cir.fadd %t, %x : !cir.double
    cir.store %s2, %sp : !cir.double, !cir.ptr<!cir.double>
    %i2 = cir.add %i, %c1 : !s32i
    cir.br ^h(%i2 : !s32i)
  ^exit:
    %r = cir.load %sp : !cir.ptr<!cir.double>, !cir.double
    cir.return %r : !cir.double
  }
  cir.func @dg(%x: !cir.double, %dr: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @g(%x, %dr) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL:   cir.func private @diffeg(
// CHECK:           %[[ZERO:.+]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// The reverse of the loop body ends with the adjoint of %s = cir.load %sp.
// CHECK:           cir.cos
// CHECK:           "enzyme.set"
// CHECK-NEXT:      %[[DS:.+]] = "enzyme.get"(%[[GS:.+]]) : (!enzyme.Gradient<!cir.double>) -> !cir.double
// CHECK-NEXT:      "enzyme.set"(%[[GS]], %[[ZERO]]) : (!enzyme.Gradient<!cir.double>, !cir.double) -> ()
// CHECK-NEXT:      %[[SHADOW:.+]] = "enzyme.pop"(%{{.+}}) : (!enzyme.Cache<!cir.ptr<!cir.double>>) -> !cir.ptr<!cir.double>
// CHECK-NEXT:      %[[OLD:.+]] = cir.load %[[SHADOW]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT:      %[[NEW:.+]] = cir.fadd %[[OLD]], %[[DS]] : !cir.double
// CHECK-NEXT:      cir.store %[[NEW]], %[[SHADOW]] : !cir.double, !cir.ptr<!cir.double>
