; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; Regression test for Enzyme.jl #3912:
; In a loop with coupled state PHIs (%s0..%s3) feeding a carried active value
; (%prev), querying %s3 in DifferentialUseAnalysis::is_value_needed_in_reverse
; must not permanently poison seen[%s1]/seen[%s2]/seen[%s3.next] with tentative
; false entries encountered along cycle edges before reaching the active sink.

declare i64 @llvm.fshl.i64(i64, i64, i64)

define double @run_loop(ptr noalias nocapture %rng, double %prev_init, double %alpha) noinline {
entry:
  %s0.pre = load i64, ptr %rng, align 8
  %s1.ptr = getelementptr inbounds i64, ptr %rng, i64 1
  %s1.pre = load i64, ptr %s1.ptr, align 8
  %s2.ptr = getelementptr inbounds i64, ptr %rng, i64 2
  %s2.pre = load i64, ptr %s2.ptr, align 8
  %s3.ptr = getelementptr inbounds i64, ptr %rng, i64 3
  %s3.pre = load i64, ptr %s3.ptr, align 8
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %s3 = phi i64 [ %s3.pre, %entry ], [ %s3.next, %loop ]
  %s2 = phi i64 [ %s2.pre, %entry ], [ %s2.next, %loop ]
  %s1 = phi i64 [ %s1.pre, %entry ], [ %s1.next, %loop ]
  %s0 = phi i64 [ %s0.pre, %entry ], [ %s0.next, %loop ]
  %prev = phi double [ %prev_init, %entry ], [ %obs, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %prod = fmul double %prev, %alpha
  %acc.next = fadd double %acc, %prod
  %t12 = shl i64 %s1, 17
  %t13 = xor i64 %s0, %s2
  %t14 = xor i64 %s1, %s3
  %t15 = xor i64 %t13, %s1
  %t16 = xor i64 %s0, %t14
  %t17 = xor i64 %t13, %t12
  %t18 = call i64 @llvm.fshl.i64(i64 %t14, i64 %t14, i64 45)
  %t19 = add i64 %t16, %t18
  %t20 = call i64 @llvm.fshl.i64(i64 %t19, i64 %t19, i64 23)
  %t21 = add i64 %t20, %t16
  %t22 = shl i64 %t15, 17
  %t23 = xor i64 %t17, %t16
  %t24 = xor i64 %t15, %t18
  %s1.next = xor i64 %t23, %t15
  %s0.next = xor i64 %t24, %t16
  %s2.next = xor i64 %t23, %t22
  %s3.next = call i64 @llvm.fshl.i64(i64 %t24, i64 %t24, i64 45)
  %obs = uitofp i64 %t21 to double
  %done = icmp eq i64 %iv.next, 8
  br i1 %done, label %exit, label %loop

exit:
  store i64 %s0.next, ptr %rng, align 8
  store i64 %s1.next, ptr %s1.ptr, align 8
  store i64 %s2.next, ptr %s2.ptr, align 8
  store i64 %s3.next, ptr %s3.ptr, align 8
  ret double %acc.next
}

define double @outer(ptr noalias nocapture %rng, double %alpha) {
entry:
  %r = call double @run_loop(ptr %rng, double 0.500000e+00, double %alpha)
  ret double %r
}

define double @caller(ptr %rng, double %alpha) {
entry:
  %dalpha = call double (...) @__enzyme_autodiff(ptr @outer, metadata !"enzyme_const", ptr %rng, double %alpha)
  ret double %dalpha
}

declare double @__enzyme_autodiff(...)

; CHECK-LABEL: define internal { double } @differun_loop(
; CHECK: entry:
; CHECK-DAG:   %s0_malloccache = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) ptr @malloc(i64 64)
; CHECK-DAG:   %t14_malloccache = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) ptr @malloc(i64 64)
; CHECK: invertloop_phirc:
; CHECK-NEXT:   %[[idx:.+]] = sub nuw i64 %"iv1'ac.0", 1
; CHECK-NEXT:   %[[s0gep:.+]] = getelementptr inbounds{{( nuw)?}} i64, ptr %s0_malloccache, i64 %[[idx]]
; CHECK-NEXT:   %[[s0val:.+]] = load i64, ptr %[[s0gep]], align 8
; CHECK-NEXT:   %[[t14gep:.+]] = getelementptr inbounds{{( nuw)?}} i64, ptr %t14_malloccache, i64 %[[idx]]
; CHECK-NEXT:   %[[t14val:.+]] = load i64, ptr %[[t14gep]], align 8
; CHECK-NEXT:   %t16_unwrap = xor i64 %[[s0val]], %[[t14val]]
; CHECK-NEXT:   %[[t18val:.+]] = call i64 @llvm.fshl.i64(i64 %[[t14val]], i64 %[[t14val]], i64 45)
; CHECK-NEXT:   %t19_unwrap = add i64 %t16_unwrap, %[[t18val]]
; CHECK-NEXT:   %[[t20val:.+]] = call i64 @llvm.fshl.i64(i64 %t19_unwrap, i64 %t19_unwrap, i64 23)
; CHECK-NEXT:   %t21_unwrap = add i64 %[[t20val]], %t16_unwrap
; CHECK-NEXT:   %obs_unwrap = uitofp i64 %t21_unwrap to double
