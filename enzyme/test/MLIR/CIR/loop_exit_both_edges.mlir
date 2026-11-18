// RUN: %eopt --enzyme %s | FileCheck %s

// What mem2reg makes of
//   do { if (j == 7) { s = 0; break; } s = s * x; j++; } while (j < n);
//   return s;
// The latch passes %s2 along both edges, into the loop and out of it. Only
// the first was followed, so %x looked as if it never reached the return:
// the shadow block arguments came out inconsistent and the result failed to
// verify.
!s32i = !cir.int<s, 32>
module {
  cir.func @f(%x: !cir.double, %n: !s32i) -> !cir.double {
    %zero = cir.const #cir.int<0> : !s32i
    %seven = cir.const #cir.int<7> : !s32i
    %fz = cir.const #cir.fp<0.000000e+00> : !cir.double
    %fone = cir.const #cir.fp<1.000000e+00> : !cir.double
    cir.br ^loop(%fone, %zero : !cir.double, !s32i)
  ^loop(%s: !cir.double, %j: !s32i):
    %brk = cir.cmp eq %j, %seven : !s32i
    cir.brcond %brk ^break, ^latch
  ^break:
    cir.br ^exit(%fz : !cir.double)
  ^latch:
    %s2 = cir.fmul %s, %x : !cir.double
    %j2 = cir.inc nsw %j : !s32i
    %go = cir.cmp lt %j2, %n : !s32i
    cir.brcond %go ^loop(%s2, %j2 : !cir.double, !s32i), ^exit(%s2 : !cir.double)
  ^exit(%r: !cir.double):
    cir.return %r : !cir.double
  }
  cir.func @fwd(%x: !cir.double, %dx: !cir.double, %n: !s32i) -> !cir.double {
    %r = enzyme.fwddiff @f(%x, %dx, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!cir.double, !cir.double, !s32i) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev(%x: !cir.double, %n: !s32i, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @f(%x, %n, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !s32i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// Forward mode: both edges of the latch pass the shadow of %s2, and the exit
// returns it.
// CHECK-LABEL: cir.func private @fwddiffef(
// CHECK-SAME:      %[[X:[a-zA-Z0-9$._-]+]]: !cir.double, %[[DX:[a-zA-Z0-9$._-]+]]: !cir.double,
// CHECK:       ^bb1(%[[S:.+]]: !cir.double, %[[DS:.+]]: !cir.double, %{{.+}}: !s32i):
// CHECK:         %[[A:.+]] = cir.fmul %[[DS]], %[[X]]
// CHECK-NEXT:    %[[B:.+]] = cir.fmul %[[DX]], %[[S]]
// CHECK-NEXT:    %[[DS2:.+]] = cir.fadd %[[A]], %[[B]]
// CHECK-NEXT:    %[[S2:.+]] = cir.fmul %[[S]], %[[X]]
// CHECK:         cir.brcond %{{.+}} ^bb1(%[[S2]], %[[DS2]], %{{.+}} : !cir.double, !cir.double, !s32i), ^bb4(%[[S2]], %[[DS2]] : !cir.double, !cir.double)
// CHECK:       ^bb4(%{{.+}}: !cir.double, %[[DR:.+]]: !cir.double):
// CHECK-NEXT:    cir.return %[[DR]]

// Reverse mode: the gradient of %x is the adjoint of %s2 times the cached %s,
// summed over the iterations, and returned.
// CHECK-LABEL: cir.func private @diffef(
// CHECK:         %[[SP:.+]] = "enzyme.pop"(%{{.+}}) : (!enzyme.Cache<!cir.double>) -> !cir.double
// CHECK-NEXT:    %[[XP:.+]] = "enzyme.pop"(%{{.+}}) : (!enzyme.Cache<!cir.double>) -> !cir.double
// CHECK-NEXT:    cir.fmul %[[D:.+]], %[[XP]]
// CHECK:         %[[G:.+]] = cir.fmul %[[D]], %[[SP]]
// CHECK-NEXT:    %[[OLD:.+]] = "enzyme.get"(%[[GX:.+]])
// CHECK-NEXT:    cir.fadd %[[OLD]], %[[G]]
// CHECK:         %[[RES:.+]] = "enzyme.get"(%[[GX]])
// CHECK-NEXT:    cir.return %[[RES]]
