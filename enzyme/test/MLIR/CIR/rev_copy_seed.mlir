// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize %s | FileCheck %s

// *p = *q for a struct of doubles. While the forward sweep runs, the shadow of
// p holds its adjoint seed, so the shadows must not be copied then; the
// reverse sweep does dq += dp; dp = 0.

!rec_S = !cir.struct<"S" {data !cir.double, data !cir.double}>
module {
  cir.func @assign(%p: !cir.ptr<!rec_S>, %q: !cir.ptr<!rec_S>) {
    cir.copy %q to %p : !cir.ptr<!rec_S>
    cir.return
  }
  cir.func @rev(%p: !cir.ptr<!rec_S>, %dp: !cir.ptr<!rec_S>, %q: !cir.ptr<!rec_S>, %dq: !cir.ptr<!rec_S>) {
    enzyme.autodiff @assign(%p, %dp, %q, %dq) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity=[] }> : (!cir.ptr<!rec_S>, !cir.ptr<!rec_S>, !cir.ptr<!rec_S>, !cir.ptr<!rec_S>) -> ()
    cir.return
  }
}

// CHECK-LABEL: cir.func private @diffeassign(
// CHECK-SAME:      %[[P:[^:]+]]: !cir.ptr<!rec_S>, %[[DP:[^:]+]]: !cir.ptr<!rec_S>, %[[Q:[^:]+]]: !cir.ptr<!rec_S>, %[[DQ:[^:]+]]: !cir.ptr<!rec_S>)
// CHECK:         cir.copy %[[Q]] to %[[P]]
// CHECK-NOT:     cir.copy
// CHECK:         %[[DPADJ:.*]] = cir.load %[[DP]]
// CHECK:         %[[DQADJ:.*]] = cir.load %[[DQ]]
// CHECK:         cir.store %{{.*}}, %[[DQ]]
// CHECK:         cir.store %{{.*}}, %[[DP]]
// CHECK:         cir.return
