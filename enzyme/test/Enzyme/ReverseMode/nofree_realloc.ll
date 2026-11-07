; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli -

declare ptr @malloc(i64)
declare ptr @realloc(ptr, i64)
declare void @free(ptr)
declare void @abort()
declare double @llvm.fabs.f64(double)
declare double @__enzyme_autodiff(...)

define internal i64 @helper_with_realloc(ptr %keep) noinline "enzyme_inactive" {
entry:
  %seen = load double, ptr %keep, align 8
  %pos = fcmp ogt double %seen, 0.0
  %addend = zext i1 %pos to i64
  %buf0 = call ptr @malloc(i64 8)
  store i64 7, ptr %buf0, align 8
  %buf1 = call ptr @realloc(ptr %buf0, i64 16)
  %gep1 = getelementptr inbounds i64, ptr %buf1, i64 1
  store i64 10, ptr %gep1, align 8
  %v0 = load i64, ptr %buf1, align 8
  %v1 = load i64, ptr %gep1, align 8
  %sum0 = add i64 %v0, %v1
  %sum = add i64 %sum0, %addend
  call void @free(ptr %buf1)
  ret i64 %sum
}

define double @caller_with_inactive_realloc(double %x) {
entry:
  %slot = call ptr @malloc(i64 8)
  store double %x, ptr %slot, align 8
  %cnt = call i64 @helper_with_realloc(ptr %slot)
  %scale = sitofp i64 %cnt to double
  %val = load double, ptr %slot, align 8
  %sq = fmul double %val, %val
  %res = fmul double %sq, %scale
  call void @free(ptr %slot)
  ret double %res
}

define double @inner_dyn(ptr %arr, double %seed) noinline {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %body ]
  %acc = phi double [ %seed, %entry ], [ %acc.next, %body ]
  %elem.ptr = getelementptr inbounds double, ptr %arr, i64 %iv
  %elem = load double, ptr %elem.ptr, align 8
  %cont = fcmp ogt double %elem, 0.0
  br i1 %cont, label %body, label %exit

body:
  %sq = fmul double %acc, %acc
  %acc.next = fadd double %sq, %elem
  %iv.next = add nuw nsw i64 %iv, 1
  br label %loop

exit:
  ret double %acc
}

define double @nested_outer(double %x) {
entry:
  %arr = call ptr @malloc(i64 32)
  store double 1.0, ptr %arr, align 8
  %p1 = getelementptr inbounds double, ptr %arr, i64 1
  store double 2.0, ptr %p1, align 8
  %p2 = getelementptr inbounds double, ptr %arr, i64 2
  store double 3.0, ptr %p2, align 8
  %p3 = getelementptr inbounds double, ptr %arr, i64 3
  store double -1.0, ptr %p3, align 8
  ; inner_dyn: i=0: 2^2+1=5, i=1: 5^2+2=27, i=2: 27^2+3=732, i=3: stop
  ; d/dseed = 2*2 * 2*5 * 2*27 = 2160
  %inner_grad = call double (...) @__enzyme_autodiff(ptr @inner_dyn, metadata !"enzyme_const", ptr %arr, double 2.0)
  %a1 = load double, ptr %p1, align 8
  %prod = fmul double %a1, %x
  %term = fmul double %inner_grad, %prod
  call void @free(ptr %arr)
  ret double %term
}

define i32 @main() {
entry:
  ; caller_with_inactive_realloc(3.0) = 18.0 * 9.0, d/dx = 18.0 * 6.0 = 108.0
  %d1 = call double (...) @__enzyme_autodiff(ptr @caller_with_inactive_realloc, double 3.0)
  %err1 = fsub double %d1, 108.0
  %abs1 = call double @llvm.fabs.f64(double %err1)
  %bad1 = fcmp ogt double %abs1, 1.0e-9
  br i1 %bad1, label %fail, label %check2

check2:
  ; nested_outer(x) = 2160.0 * 2.0 * x = 4320.0 * x, d/dx = 4320.0
  %d2 = call double (...) @__enzyme_autodiff(ptr @nested_outer, double 5.0)
  %err2 = fsub double %d2, 4320.0
  %abs2 = call double @llvm.fabs.f64(double %err2)
  %bad2 = fcmp ogt double %abs2, 1.0e-9
  br i1 %bad2, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret i32 0
}

; CHECK-LABEL: define internal i64 @nofree_helper_with_realloc(
; CHECK-NOT: call ptr @realloc(
; CHECK-NOT: call void @free(
; CHECK: ret i64
; CHECK: resize{{.*}}:
; CHECK-NOT: call ptr @realloc(
; CHECK-NOT: call void @free(
; CHECK: %[[NEWBUF:.+]] = call ptr @malloc(i64 16)
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr %[[NEWBUF]], ptr %buf0, i64 8, i1 false)
; CHECK-NOT: call void @free(
; CHECK: }

; CHECK-LABEL: define internal { double } @nofree_diffeinner_dyn(
; CHECK-NOT: call ptr @realloc(
; CHECK-NOT: call void @free(
; CHECK: ret { double }
; CHECK: resize{{.*}}:
; CHECK-NOT: call ptr @realloc(
; CHECK-NOT: call void @free(
; CHECK: %[[NEWCACHE:.+]] = call ptr @malloc(i64 %[[NEWSZ:.+]])
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr %[[NEWCACHE]], ptr %[[OLDCACHE:.+]], i64 %[[OLDSZ:.+]], i1 false)
; CHECK-NOT: call void @free(
; CHECK: }
