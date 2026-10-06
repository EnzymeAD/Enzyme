; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; for (i = 0; i < N; i++)        // N, M >= 1
;   for (j = 0; j < M; j++) {
;     v = A[i*K + M + j];
;     A[i*K + j] = v * v;
;   }
;
; With K == M, outer iteration i+1 overwrites exactly what iteration i read,
; so %v must be cached rather than reloaded from %A in the reverse pass.
; Integration/ReverseMode/overwrite_outer_loop.c checks the gradient.

declare void @__enzyme_autodiff(...)

define void @f(ptr %A, i64 %K, i64 %N, i64 %M) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %off = mul i64 %i, %K
  %base = getelementptr double, ptr %A, i64 %off
  %hi = getelementptr double, ptr %base, i64 %M
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %ldp = getelementptr double, ptr %hi, i64 %j
  %v = load double, ptr %ldp, align 8
  %sq = fmul double %v, %v
  %stp = getelementptr double, ptr %base, i64 %j
  store double %sq, ptr %stp, align 8
  %j.next = add nuw nsw i64 %j, 1
  %inner.done = icmp eq i64 %j.next, %M
  br i1 %inner.done, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.done = icmp eq i64 %i.next, %N
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

define void @test(ptr %A, ptr %dA, i64 %K, i64 %N, i64 %M) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, ptr %A, ptr %dA, i64 %K, i64 %N, i64 %M)
  ret void
}

; %v is cached in the forward pass and read back from the cache in the reverse
; pass.

; CHECK: define internal void @diffef(ptr {{.*}}%A, ptr {{.*}}%"A'", i64 %K, i64 %N, i64 %M)

; CHECK: entry:
; CHECK-NEXT:   %[[R0:[0-9]+]] = add i64 %N, -1
; CHECK-NEXT:   %[[R1:[0-9]+]] = add i64 %M, -1
; CHECK-NEXT:   %[[R2:[0-9]+]] = mul nuw nsw i64 %M, %N
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[R2]], 8
; CHECK-NEXT:   %[[VCACHE:[A-Za-z0-9_.]+]] = tail call noalias nonnull ptr @malloc(i64 %mallocsize), !enzyme_cache_alloc !0
; CHECK-NEXT:   br label %outer

; CHECK: outer:
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %outer.latch ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %off = mul i64 %iv, %K
; CHECK-NEXT:   %base = getelementptr double, ptr %A, i64 %off
; CHECK-NEXT:   %hi = getelementptr double, ptr %base, i64 %M
; CHECK-NEXT:   br label %inner

; CHECK: inner:
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %inner ], [ 0, %outer ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %ldp = getelementptr double, ptr %hi, i64 %iv1
; CHECK-NEXT:   %v = load double, ptr %ldp, align 8, !alias.scope !2, !noalias !5
; CHECK-NEXT:   %sq = fmul double %v, %v
; CHECK-NEXT:   %stp = getelementptr double, ptr %base, i64 %iv1
; CHECK-NEXT:   store double %sq, ptr %stp, align 8, !alias.scope !2, !noalias !5
; CHECK-NEXT:   %[[R3:[0-9]+]] = mul nuw nsw i64 %iv, %M
; CHECK-NEXT:   %[[R4:[0-9]+]] = add nuw nsw i64 %iv1, %[[R3]]
; CHECK-NEXT:   %[[R5:[0-9]+]] = getelementptr inbounds double, ptr %[[VCACHE]], i64 %[[R4]]
; CHECK-NEXT:   store double %v, ptr %[[R5]], align 8, !invariant.group !7
; CHECK-NEXT:   %inner.done = icmp eq i64 %iv.next2, %M
; CHECK-NEXT:   br i1 %inner.done, label %outer.latch, label %inner

; CHECK: outer.latch:
; CHECK-NEXT:   %outer.done = icmp eq i64 %iv.next, %N
; CHECK-NEXT:   br i1 %outer.done, label %invertouter.latch, label %outer

; CHECK: invertentry:
; CHECK-NEXT:   tail call void @free(ptr nonnull %[[VCACHE]]), !enzyme_cache_free !0
; CHECK-NEXT:   ret void

; CHECK: invertouter:
; CHECK-NEXT:   %[[R6:[0-9]+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[R6]], label %invertentry, label %incinvertouter

; CHECK: incinvertouter:
; CHECK-NEXT:   %[[R7:[0-9]+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertouter.latch

; CHECK: invertinner:
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %[[R1]], %invertouter.latch ], [ %[[R19:[0-9]+]], %incinvertinner ]
; CHECK-NEXT:   %off_unwrap = mul i64 %"iv'ac.0", %K
; CHECK-NEXT:   %"base'ipg_unwrap" = getelementptr double, ptr %"A'", i64 %off_unwrap
; CHECK-NEXT:   %"stp'ipg_unwrap" = getelementptr double, ptr %"base'ipg_unwrap", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[R8:[0-9]+]] = load double, ptr %"stp'ipg_unwrap", align 8, !alias.scope !5, !noalias !2
; CHECK-NEXT:   store double 0.000000e+00, ptr %"stp'ipg_unwrap", align 8, !alias.scope !5, !noalias !2
; CHECK-NEXT:   %[[R9:[0-9]+]] = mul nuw nsw i64 %"iv'ac.0", %M
; CHECK-NEXT:   %[[R10:[0-9]+]] = add nuw nsw i64 %"iv1'ac.0", %[[R9]]
; CHECK-NEXT:   %[[R11:[0-9]+]] = getelementptr inbounds double, ptr %[[VCACHE]], i64 %[[R10]]
; CHECK-NEXT:   %[[R12:[0-9]+]] = load double, ptr %[[R11]], align 8, !invariant.group !7, !enzyme_type !8
; CHECK-NEXT:   %[[R13:[0-9]+]] = fmul fast double %[[R8]], %[[R12]]
; CHECK-NEXT:   %[[R14:[0-9]+]] = fmul fast double %[[R8]], %[[R12]]
; CHECK-NEXT:   %[[R15:[0-9]+]] = fadd fast double %[[R13]], %[[R14]]
; CHECK-NEXT:   %"hi'ipg_unwrap" = getelementptr double, ptr %"base'ipg_unwrap", i64 %M
; CHECK-NEXT:   %"ldp'ipg_unwrap" = getelementptr double, ptr %"hi'ipg_unwrap", i64 %"iv1'ac.0"
; CHECK-NEXT:   %[[R16:[0-9]+]] = load double, ptr %"ldp'ipg_unwrap", align 8, !alias.scope !5, !noalias !2
; CHECK-NEXT:   %[[R17:[0-9]+]] = fadd fast double %[[R16]], %[[R15]]
; CHECK-NEXT:   store double %[[R17]], ptr %"ldp'ipg_unwrap", align 8, !alias.scope !5, !noalias !2
; CHECK-NEXT:   %[[R18:[0-9]+]] = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %[[R18]], label %invertouter, label %incinvertinner

; CHECK: incinvertinner:
; CHECK-NEXT:   %[[R19]] = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertinner

; CHECK: invertouter.latch:
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[R7]], %incinvertouter ], [ %[[R0]], %outer.latch ]
; CHECK-NEXT:   br label %invertinner
; CHECK-NEXT: }
