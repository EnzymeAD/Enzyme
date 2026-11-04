; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg),verify" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

@.str = private unnamed_addr constant [5 x i8] c"%.0f\00", align 1
@.sep = private unnamed_addr constant [2 x i8] c" \00", align 1
@.nl = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

declare void @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

; With K == 4, outer iteration i + 1 stores into [4*(i+1), 4*(i+1)+4), which
; overwrites the range [4*i+4, 4*i+8) read by iteration i even though the load
; and store use disjoint constant offsets from the loop-varying %base pointer.
define void @nested_loop(ptr %A, i64 %K, i64 %N) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %off = mul nsw i64 %i, %K
  %base = getelementptr inbounds double, ptr %A, i64 %off
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %ld.idx = add nuw nsw i64 %j, 4
  %ld.ptr = getelementptr inbounds double, ptr %base, i64 %ld.idx
  %val = load double, ptr %ld.ptr, align 8
  %sq = fmul double %val, %val
  %st.ptr = getelementptr inbounds double, ptr %base, i64 %j
  store double %sq, ptr %st.ptr, align 8
  %j.next = add nuw nsw i64 %j, 1
  %inner.done = icmp eq i64 %j.next, 4
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.done = icmp eq i64 %i.next, %N
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

; Unrolled single-loop form where %K has unknown sign, so same-iteration SCEV
; subtraction must not cancel the %K stride across outer iterations.
define void @unrolled_inner(ptr %A, i64 %K, i64 %N) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %off = mul nsw i64 %i, %K
  %base = getelementptr inbounds double, ptr %A, i64 %off
  %ld.ptr = getelementptr inbounds double, ptr %base, i64 4
  %val = load double, ptr %ld.ptr, align 8
  %sq = fmul double %val, %val
  store double %sq, ptr %base, align 8
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %N
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define i32 @main() {
entry:
  %A = alloca [16 x double], align 8
  %dA = alloca [16 x double], align 8
  br label %init

init:
  %k = phi i64 [ 0, %entry ], [ %k.next, %init ]
  %kp1 = add nuw nsw i64 %k, 1
  %val = sitofp i64 %kp1 to double
  %ap = getelementptr inbounds [16 x double], ptr %A, i64 0, i64 %k
  %dap = getelementptr inbounds [16 x double], ptr %dA, i64 0, i64 %k
  store double %val, ptr %ap, align 8
  store double 1.000000e+00, ptr %dap, align 8
  %k.next = add nuw nsw i64 %k, 1
  %init.done = icmp eq i64 %k.next, 16
  br i1 %init.done, label %run1, label %init

run1:
  call void (...) @__enzyme_autodiff(ptr @nested_loop, ptr %A, ptr %dA, i64 4, i64 3)
  br label %print1

print1:
  %p1 = phi i64 [ 0, %run1 ], [ %p1.next, %print1 ]
  %dap1 = getelementptr inbounds [16 x double], ptr %dA, i64 0, i64 %p1
  %dv1 = load double, ptr %dap1, align 8
  %c1 = call i32 (ptr, ...) @printf(ptr @.str, double %dv1)
  %p1.next = add nuw nsw i64 %p1, 1
  %p1.done = icmp eq i64 %p1.next, 16
  %sep1 = select i1 %p1.done, ptr @.nl, ptr @.sep
  %c2 = call i32 (ptr, ...) @printf(ptr %sep1)
  br i1 %p1.done, label %run2, label %print1

run2:
  %A2 = alloca [9 x double], align 8
  %dA2 = alloca [9 x double], align 8
  br label %init2

init2:
  %m = phi i64 [ 0, %run2 ], [ %m.next, %init2 ]
  %mp1 = add nuw nsw i64 %m, 1
  %mval = sitofp i64 %mp1 to double
  %ap2 = getelementptr inbounds [9 x double], ptr %A2, i64 0, i64 %m
  %dap2 = getelementptr inbounds [9 x double], ptr %dA2, i64 0, i64 %m
  store double %mval, ptr %ap2, align 8
  store double 1.000000e+00, ptr %dap2, align 8
  %m.next = add nuw nsw i64 %m, 1
  %init2.done = icmp eq i64 %m.next, 9
  br i1 %init2.done, label %call2, label %init2

call2:
  call void (...) @__enzyme_autodiff(ptr @unrolled_inner, ptr %A2, ptr %dA2, i64 4, i64 2)
  %r0p = getelementptr inbounds [9 x double], ptr %dA2, i64 0, i64 4
  %r0 = load double, ptr %r0p, align 8
  %r1p = getelementptr inbounds [9 x double], ptr %dA2, i64 0, i64 8
  %r1 = load double, ptr %r1p, align 8
  %c3 = call i32 (ptr, ...) @printf(ptr @.str, double %r0)
  %c4 = call i32 (ptr, ...) @printf(ptr @.sep)
  %c5 = call i32 (ptr, ...) @printf(ptr @.str, double %r1)
  %c6 = call i32 (ptr, ...) @printf(ptr @.nl)
  ret i32 0
}

; EVAL: 0 0 0 0 10 12 14 16 18 20 22 24 27 29 31 33
; EVAL-NEXT: 10 19

; CHECK-LABEL: define internal void @diffenested_loop(
; CHECK-SAME: ptr {{.*}}%A, ptr {{.*}}%"A'", i64 %K, i64 %N)
; CHECK: entry:
; CHECK:   %[[CACHE1:.+]] = tail call noalias nonnull ptr @malloc(i64 %mallocsize), !enzyme_cache_alloc
; CHECK: inner:
; CHECK:   %val = load double, ptr %ld.ptr, align 8
; CHECK:   %[[SLOT1:.+]] = getelementptr inbounds double, ptr %[[CACHE1]], i64 %{{.+}}
; CHECK:   store double %val, ptr %[[SLOT1]], align 8
; CHECK: invertentry:
; CHECK:   tail call void @free(ptr nonnull %[[CACHE1]]), !enzyme_cache_free
; CHECK: invertinner:
; CHECK:   %[[REVSLOT1:.+]] = getelementptr inbounds double, ptr %[[CACHE1]], i64 %{{.+}}
; CHECK:   %[[CACHED1:.+]] = load double, ptr %[[REVSLOT1]], align 8

; CHECK-LABEL: define internal void @diffeunrolled_inner(
; CHECK-SAME: ptr {{.*}}%A, ptr {{.*}}%"A'", i64 %K, i64 %N)
; CHECK: entry:
; CHECK:   %[[CACHE2:.+]] = tail call noalias nonnull ptr @malloc(i64 %mallocsize), !enzyme_cache_alloc
; CHECK: loop:
; CHECK:   %val = load double, ptr %ld.ptr, align 8
; CHECK:   %[[SLOT2:.+]] = getelementptr inbounds double, ptr %[[CACHE2]], i64 %iv
; CHECK:   store double %val, ptr %[[SLOT2]], align 8
; CHECK: invertentry:
; CHECK:   tail call void @free(ptr nonnull %[[CACHE2]]), !enzyme_cache_free
; CHECK: invertloop:
; CHECK:   %[[REVSLOT2:.+]] = getelementptr inbounds double, ptr %[[CACHE2]], i64 %"iv'ac.0"
; CHECK:   %[[CACHED2:.+]] = load double, ptr %[[REVSLOT2]], align 8
