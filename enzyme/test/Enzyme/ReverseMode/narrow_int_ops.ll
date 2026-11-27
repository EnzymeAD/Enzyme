; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,instsimplify,simplifycfg)" -S 2>&1 | FileCheck %s; fi

target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"

declare void @__enzyme_autodiff(...)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.abs.i32(i32, i1)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

define void @step_ops(ptr nocapture %state, ptr nocapture writeonly %out) {
entry:
  %p = getelementptr inbounds { double, i32 }, ptr %state, i64 0, i32 1
  %nfe = load i32, ptr %p, align 4
  %q = sdiv i32 100, %nfe
  %m1 = call i32 @llvm.smax.i32(i32 %q, i32 0)
  %m2 = call i32 @llvm.smin.i32(i32 %m1, i32 50)
  %m3 = call i32 @llvm.umax.i32(i32 %m2, i32 1)
  %m4 = call i32 @llvm.umin.i32(i32 %m3, i32 49)
  %ab = call i32 @llvm.abs.i32(i32 %m4, i1 false)
  %ov = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %ab, i32 1)
  %v = extractvalue { i32, i1 } %ov, 0
  store i32 %v, ptr %p, align 4
  %x = load double, ptr %state, align 8
  %y = fmul double %x, %x
  store double %y, ptr %out, align 8
  ret void
}

define void @uncond_loop(ptr nocapture %state, ptr nocapture writeonly %out) #0 {
entry:
  %p = getelementptr inbounds { double, i32 }, ptr %state, i64 0, i32 1
  %nfe0 = load i32, ptr %p, align 4
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %cnt = phi i32 [ %nfe0, %entry ], [ %inc, %loop ]
  %inc = add i32 %cnt, 1
  %inext = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inext, 3
  br i1 %done, label %exit, label %loop

exit:
  store i32 %inc, ptr %p, align 4
  %x = load double, ptr %state, align 8
  %y = fmul double %x, %x
  store double %y, ptr %out, align 8
  ret void
}

define void @loop_caller(ptr %s, ptr %o) {
  call void @uncond_loop(ptr %s, ptr %o)
  ret void
}

define void @rev_step(ptr %s, ptr %ds, ptr %o, ptr %do) {
  call void (...) @__enzyme_autodiff(ptr @step_ops, ptr %s, ptr %ds, ptr %o, ptr %do)
  ret void
}

define void @rev_loop(ptr %s, ptr %ds, ptr %o, ptr %do) {
  call void (...) @__enzyme_autodiff(ptr @loop_caller, ptr %s, ptr %ds, ptr %o, ptr %do)
  ret void
}

attributes #0 = { noinline }

; CHECK-NOT: Mismatched activity

; CHECK-LABEL: define internal void @diffestep_ops(
; CHECK:   %nfe = load i32, ptr %p, align 4
; CHECK-NEXT:   %q = sdiv i32 100, %nfe
; CHECK-NOT: sdiv i32
; CHECK:   %[[DOUT:.+]] = load double, ptr %"out'", align 8
; CHECK:   store double 0.000000e+00, ptr %"out'", align 8
; CHECK:   %[[MUL:.+]] = fmul fast double %[[DOUT]], %x
; CHECK:   %[[ADD:.+]] = fadd fast double %[[MUL]], %[[MUL]]
; CHECK:   %[[DS:.+]] = load double, ptr %"state'", align 8
; CHECK:   %[[NS:.+]] = fadd fast double %[[DS]], %[[ADD]]
; CHECK:   store double %[[NS]], ptr %"state'", align 8
; CHECK:   ret void

; CHECK-LABEL: define internal void @diffeuncond_loop(
; CHECK:   %[[DOUT2:.+]] = load double, ptr %"out'", align 8
; CHECK:   store double 0.000000e+00, ptr %"out'", align 8
; CHECK:   %[[MUL2:.+]] = fmul fast double %[[DOUT2]], %x
; CHECK:   %[[ADD2:.+]] = fadd fast double %[[MUL2]], %[[MUL2]]
; CHECK:   %[[DS2:.+]] = load double, ptr %"state'", align 8
; CHECK:   %[[NS2:.+]] = fadd fast double %[[DS2]], %[[ADD2]]
; CHECK:   store double %[[NS2]], ptr %"state'", align 8
; CHECK:   ret void
