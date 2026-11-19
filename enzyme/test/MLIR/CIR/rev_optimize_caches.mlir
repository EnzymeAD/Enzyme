// RUN: %eopt --enzyme=optimizeCaches=true --canonicalize %s | FileCheck %s
// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s --check-prefix=OFF

// With optimizeCaches, what the reverse sweep needs of the forward sweep is
// not cached once per use:
//  - %x, an argument, is there in the reverse sweep as it is;
//  - %s, carried around the loop, is pushed once an iteration where it is
//    defined, read by the fmul's adjoint, and popped once the reverse sweep
//    is back at the top of the iteration;
//  - &p[i], an address, is computed again from p's shadow and i.

!s32i = !cir.int<s, 32>
module {
  cir.func @f(%x: !cir.double, %p: !cir.ptr<!cir.double>, %n: !s32i) -> !cir.double {
    %zero = cir.const #cir.int<0> : !s32i
    %one = cir.const #cir.fp<1.0> : !cir.double
    cir.br ^loop(%one, %zero : !cir.double, !s32i)
  ^loop(%s: !cir.double, %i: !s32i):
    %a = cir.ptr_stride %p, %i : (!cir.ptr<!cir.double>, !s32i) -> !cir.ptr<!cir.double>
    %v = cir.load %a : !cir.ptr<!cir.double>, !cir.double
    %t = cir.fmul %s, %x : !cir.double
    %u = cir.fmul %t, %v : !cir.double
    %next = cir.inc %i : !s32i
    %more = cir.cmp lt %next, %n : !s32i
    cir.brcond %more ^loop(%u, %next : !cir.double, !s32i), ^done
  ^done:
    cir.return %u : !cir.double
  }
  cir.func @rev(%x: !cir.double, %p: !cir.ptr<!cir.double>, %dp: !cir.ptr<!cir.double>, %n: !s32i, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @f(%x, %p, %dp, %n, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !s32i, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL: cir.func private @diffef(
// CHECK-SAME:    %[[X:[^:]+]]: !cir.double, %[[P:[^:]+]]: !cir.ptr<!cir.double>, %[[DP:[^:]+]]: !cir.ptr<!cir.double>
// CHECK-NOT:     "enzyme.push"(%{{.+}}, %[[X]])
// CHECK:       ^bb1(%[[S:[^:]+]]: !cir.double, %[[I:[^:]+]]: !s32i,
// CHECK:         "enzyme.push"(%[[ICACHE:.+]], %[[I]]) : (!enzyme.Cache<!s32i>, !s32i) -> ()
// CHECK-NEXT:    "enzyme.push"(%[[SCACHE:.+]], %[[S]]) : (!enzyme.Cache<!cir.double>, !cir.double) -> ()
// CHECK-NOT:     "enzyme.push"(%{{.+}}) : (!enzyme.Cache<!cir.ptr
// CHECK-NOT:     "enzyme.push"(%{{.+}}, %[[X]])
// CHECK:         cir.brcond
// CHECK:         %[[RS:.+]] = "enzyme.pop"(%[[SCACHE]])
// CHECK:         cir.fmul %{{.+}}, %[[RS]]
// CHECK:         %[[RI:.+]] = "enzyme.pop"(%[[ICACHE]])
// CHECK-NEXT:    %[[RA:.+]] = cir.ptr_stride %[[DP]], %[[RI]]
// CHECK-NEXT:    %[[OLD:.+]] = cir.load %[[RA]]
// CHECK-NEXT:    %[[NEW:.+]] = cir.fadd %[[OLD]]
// CHECK-NEXT:    cir.store %[[NEW]], %[[RA]]

// Without it, the address and %x are cached by each op that needs them.
// OFF-LABEL:   cir.func private @diffef(
// OFF-SAME:      %[[X:[^:]+]]: !cir.double,
// OFF:           "enzyme.push"(%{{.+}}, %{{.+}}) : (!enzyme.Cache<!cir.ptr<!cir.double>>, !cir.ptr<!cir.double>) -> ()
// OFF:           "enzyme.push"(%{{.+}}, %[[X]])
