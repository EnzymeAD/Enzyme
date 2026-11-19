// RUN: %eopt --enzyme=runtimeActivity=true --canonicalize %s | FileCheck %s
// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s --check-prefix=OFF

// An inactive pointer stored into differentiated memory has itself for its
// shadow, so a pointer that was passed in, or loaded from memory, can too.
// With runtime activity, the reverse sweep compares such a shadow with its
// primal before it accumulates through it: when they are the same, the
// adjoint belongs to nothing, and goes to scratch memory instead of into the
// primal. A pointer into a local slot, whose shadow is allocated with it,
// needs no check.

module {
  // q[0] = p[0] * p[0], through a local slot
  cir.func @sq(%p: !cir.ptr<!cir.double>, %q: !cir.ptr<!cir.double>) {
    %t = cir.alloca "t" align(8) : !cir.ptr<!cir.double>
    %v = cir.load %p : !cir.ptr<!cir.double>, !cir.double
    %m = cir.fmul %v, %v : !cir.double
    cir.store %m, %t : !cir.double, !cir.ptr<!cir.double>
    %r = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.store %r, %q : !cir.double, !cir.ptr<!cir.double>
    cir.return
  }
  cir.func @rev(%p: !cir.ptr<!cir.double>, %dp: !cir.ptr<!cir.double>, %q: !cir.ptr<!cir.double>, %dq: !cir.ptr<!cir.double>) {
    enzyme.autodiff @sq(%p, %dp, %q, %dq) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[] }> : (!cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>, !cir.ptr<!cir.double>) -> ()
    cir.return
  }
}

// CHECK:       cir.global "private" internal @[[SCRATCH:__enzyme_cir_scratch[0-9]*]] = #cir.fp<0.000000e+00> : !cir.double
// CHECK-LABEL: cir.func private @diffesq(
// CHECK-SAME:    %[[P:[^:]+]]: !cir.ptr<!cir.double>, %[[DP:[^:]+]]: !cir.ptr<!cir.double>, %[[Q:[^:]+]]: !cir.ptr<!cir.double>, %[[DQ:[^:]+]]: !cir.ptr<!cir.double>
// CHECK:         %[[DT:.+]] = cir.alloca "t"
// CHECK:         "enzyme.push"(%[[DPC:.+]], %[[DP]])
// CHECK-NEXT:    "enzyme.push"(%[[PC:.+]], %[[P]])
// CHECK:         "enzyme.push"(%[[DQC:.+]], %[[DQ]])
// CHECK-NEXT:    "enzyme.push"(%[[QC:.+]], %[[Q]])

// The store into q: what dq holds is the adjoint unless dq is q, and it is
// zeroed in scratch memory then.
// CHECK:         %[[RDQ:.+]] = "enzyme.pop"(%[[DQC]])
// CHECK-NEXT:    %[[RQ:.+]] = "enzyme.pop"(%[[QC]])
// CHECK-NEXT:    %[[QSAME:.+]] = cir.cmp eq %[[RDQ]], %[[RQ]] : !cir.ptr<!cir.double>
// CHECK-NEXT:    %[[QADJ:.+]] = cir.load %[[RDQ]]
// CHECK-NEXT:    cir.select if %[[QSAME]] then %{{.+}} else %[[QADJ]]
// CHECK:         %[[QS:.+]] = cir.get_global @[[SCRATCH]]
// CHECK-NEXT:    %[[QZ:.+]] = cir.select if %[[QSAME]] then %[[QS]] else %[[RDQ]]
// CHECK-NEXT:    cir.store %{{.+}}, %[[QZ]]

// The slot's shadow is accumulated into directly.
// CHECK:         %[[RDT:.+]] = "enzyme.pop"
// CHECK-NEXT:    cir.load %[[RDT]]
// CHECK-NEXT:    cir.fadd
// CHECK-NEXT:    cir.store %{{.+}}, %[[RDT]]

// The load from p accumulates into dp, or into scratch memory if dp is p.
// CHECK:         %[[RDP:.+]] = "enzyme.pop"(%[[DPC]])
// CHECK-NEXT:    %[[RP:.+]] = "enzyme.pop"(%[[PC]])
// CHECK-NEXT:    %[[PSAME:.+]] = cir.cmp eq %[[RDP]], %[[RP]] : !cir.ptr<!cir.double>
// CHECK-NEXT:    %[[PS:.+]] = cir.get_global @[[SCRATCH]]
// CHECK-NEXT:    %[[PADDR:.+]] = cir.select if %[[PSAME]] then %[[PS]] else %[[RDP]]
// CHECK-NEXT:    %[[PADJ:.+]] = cir.load %[[PADDR]]
// CHECK-NEXT:    %[[PSUM:.+]] = cir.fadd %[[PADJ]]
// CHECK-NEXT:    cir.store %[[PSUM]], %[[PADDR]]
// CHECK-NEXT:    cir.return

// Without runtime activity, nothing is compared.
// OFF-NOT:       __enzyme_cir_scratch
// OFF-LABEL:   cir.func private @diffesq(
// OFF-NOT:       cir.cmp eq
// OFF:           cir.return
