; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-dedup-loads=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=NODEDUP; fi

; Two identical loads of x[i] (as in `x[i] * x[i]` before CSE). Since x is
; overwritten after the loop, the loaded values must be cached, but both loads
; read the same address in the same iteration with no intervening write, so
; they must share a single cache.

define double @sumsq(ptr %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi double [ 0.000000e+00, %entry ], [ %s.next, %loop ]
  %gep1 = getelementptr inbounds double, ptr %x, i64 %i
  %a = load double, ptr %gep1, align 8
  %gep2 = getelementptr inbounds double, ptr %x, i64 %i
  %b = load double, ptr %gep2, align 8
  %m = fmul double %a, %b
  %s.next = fadd double %s, %m
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop

exit:
  store double 0.000000e+00, ptr %x, align 8
  ret double %s.next
}

; Same with an indirect index, so the value is cached per iteration.
define double @sumsq_ind(ptr %x, ptr %idx, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi double [ 0.000000e+00, %entry ], [ %s.next, %loop ]
  %ip = getelementptr inbounds i64, ptr %idx, i64 %i
  %j = load i64, ptr %ip, align 8
  %gep1 = getelementptr inbounds double, ptr %x, i64 %j
  %a = load double, ptr %gep1, align 8
  %gep2 = getelementptr inbounds double, ptr %x, i64 %j
  %b = load double, ptr %gep2, align 8
  %m = fmul double %a, %b
  %s.next = fadd double %s, %m
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop

exit:
  store double 0.000000e+00, ptr %x, align 8
  ret double %s.next
}

; A store to x[i] between the two loads: they must not share a cache.
define double @sumsq_clob(ptr %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi double [ 0.000000e+00, %entry ], [ %s.next, %loop ]
  %gep1 = getelementptr inbounds double, ptr %x, i64 %i
  %a = load double, ptr %gep1, align 8
  %a2 = fmul double %a, 2.000000e+00
  store double %a2, ptr %gep1, align 8
  %gep2 = getelementptr inbounds double, ptr %x, i64 %i
  %b = load double, ptr %gep2, align 8
  %m = fmul double %a, %b
  %s.next = fadd double %s, %m
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop

exit:
  store double 0.000000e+00, ptr %x, align 8
  ret double %s.next
}

; The same loop in a function called from a loop, so it is differentiated in
; split mode and the loaded value is stored in the tape of the augmented
; primal: only one of the two loads may be stored there.
@g = global double 0.000000e+00

define double @sumsq_callee(ptr %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi double [ 0.000000e+00, %entry ], [ %s.next, %loop ]
  %gep1 = getelementptr inbounds double, ptr %x, i64 %i
  %a = load double, ptr %gep1, align 8
  %gep2 = getelementptr inbounds double, ptr %x, i64 %i
  %b = load double, ptr %gep2, align 8
  %m = fmul double %a, %b
  %s.next = fadd double %s, %m
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop

exit:
  store double %s.next, ptr @g, align 8
  ret double %s.next
}

define void @outer(ptr noalias %out, ptr %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %r = call double @sumsq_callee(ptr %x, i64 %n)
  %g = getelementptr inbounds double, ptr %out, i64 %i
  store double %r, ptr %g, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}

declare double @__enzyme_autodiff(ptr, ...)

define void @test(ptr %x, ptr %dx, ptr %idx, ptr %out, ptr %dout, i64 %n) {
entry:
  %r1 = call double (ptr, ...) @__enzyme_autodiff(ptr @sumsq, ptr %x, ptr %dx, i64 %n)
  %r2 = call double (ptr, ...) @__enzyme_autodiff(ptr @sumsq_ind, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %idx, i64 %n)
  %r3 = call double (ptr, ...) @__enzyme_autodiff(ptr @sumsq_clob, ptr %x, ptr %dx, i64 %n)
  %r4 = call double (ptr, ...) @__enzyme_autodiff(ptr @outer, ptr %out, ptr %dout, ptr %x, ptr %dx, i64 %n)
  ret void
}

; Without deduplication each of the two loads gets its own cache.
; NODEDUP-LABEL: define internal void @diffesumsq(
; NODEDUP: %b_malloccache = tail call noalias nonnull ptr @malloc
; NODEDUP: %a_malloccache = tail call noalias nonnull ptr @malloc
; NODEDUP-LABEL: define internal { { ptr, ptr }, double } @augmented_sumsq_callee(
; NODEDUP: %a_malloccache = tail call noalias nonnull ptr @malloc
; NODEDUP: %b_malloccache = tail call noalias nonnull ptr @malloc

; CHECK: define internal void @diffesumsq(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %n, -1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %a_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK-NEXT:   %1 = mul nuw nsw i64 8, %n
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %a_malloccache, ptr nonnull align 8 %x, i64 %1, i1 false)
; CHECK-NEXT:   br label %loop
; CHECK-NOT: @malloc(
; CHECK: invertentry:
; CHECK-NEXT:   tail call void @free(ptr nonnull %a_malloccache)
; CHECK-NEXT:   ret void
; CHECK: invertloop:
; CHECK-NEXT:   %"s.next'de.0" = phi double [ %differeturn, %exit ], [ %10, %incinvertloop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %0, %exit ], [ %11, %incinvertloop ]
; CHECK-NEXT:   %2 = getelementptr inbounds double, ptr %a_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %3 = load double, ptr %2, align 8
; CHECK-NEXT:   %4 = fmul fast double %"s.next'de.0", %3
; CHECK-NEXT:   %5 = fmul fast double %"s.next'de.0", %3
; CHECK-NEXT:   %6 = fadd fast double %4, %5
; CHECK-NEXT:   %"gep1'ipg_unwrap" = getelementptr inbounds double, ptr %"x'", i64 %"iv'ac.0"
; CHECK-NEXT:   %7 = load double, ptr %"gep1'ipg_unwrap", align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %6
; CHECK-NEXT:   store double %8, ptr %"gep1'ipg_unwrap", align 8
; CHECK-NOT: @malloc(
; CHECK: }

; CHECK: define internal void @diffesumsq_ind(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%idx, i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %n, -1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %a_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK-NEXT:   %mallocsize8 = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %j_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize8)
; CHECK-NOT: @malloc(
; CHECK: loop:
; CHECK:   %a = load double, ptr %gep1, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds double, ptr %a_malloccache, i64 %iv
; CHECK-NEXT:   store double %a, ptr %2, align 8
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next, %n
; CHECK: invertloop:
; CHECK-NEXT:   %"s.next'de.0" = phi double [ %differeturn, %exit ], [ %13, %incinvertloop ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %0, %exit ], [ %14, %incinvertloop ]
; CHECK-NEXT:   %3 = getelementptr inbounds double, ptr %a_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %4 = load double, ptr %3, align 8
; CHECK-NEXT:   %5 = fmul fast double %"s.next'de.0", %4
; CHECK-NEXT:   %6 = fmul fast double %"s.next'de.0", %4
; CHECK-NEXT:   %7 = fadd fast double %5, %6
; CHECK-NOT: @malloc(
; CHECK: }

; The store to x[i] between the loads keeps them apart: two caches.
; CHECK: define internal void @diffesumsq_clob(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = add i64 %n, -1
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %b_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK-NEXT:   %mallocsize8 = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %a_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize8)
; CHECK: loop:
; CHECK:   %a = load double, ptr %gep1, align 8
; CHECK:   store double %a2, ptr %gep1, align 8
; CHECK:   store double %a, ptr %1, align 8
; CHECK:   %b = load double, ptr %gep2, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds double, ptr %b_malloccache, i64 %iv
; CHECK-NEXT:   store double %b, ptr %2, align 8

; Split mode: the tape of the augmented primal holds a single cache.
; CHECK: define internal { ptr, double } @augmented_sumsq_callee(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { ptr, double }, align 8
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %n, 8
; CHECK-NEXT:   %a_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK-NEXT:   store ptr %a_malloccache, ptr %0, align 8
; CHECK-NEXT:   br label %loop
; CHECK: loop:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %s = phi double [ 0.000000e+00, %entry ], [ %s.next, %loop ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %gep1 = getelementptr inbounds double, ptr %x, i64 %iv
; CHECK-NEXT:   %a = load double, ptr %gep1, align 8
; CHECK-NEXT:   %1 = getelementptr inbounds double, ptr %a_malloccache, i64 %iv
; CHECK-NEXT:   store double %a, ptr %1, align 8
; CHECK-NEXT:   %m = fmul double %a, %a

; CHECK: define internal void @diffesumsq_callee(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %n, double %differeturn, ptr %tapeArg)
; CHECK: invertloop:
; CHECK:   %1 = getelementptr inbounds double, ptr %tapeArg, i64 %"iv'ac.0"
; CHECK-NEXT:   %2 = load double, ptr %1, align 8
; CHECK-NEXT:   %3 = fmul fast double %"s.next'de.0", %2
; CHECK-NEXT:   %4 = fmul fast double %"s.next'de.0", %2
; CHECK-NEXT:   %5 = fadd fast double %3, %4
