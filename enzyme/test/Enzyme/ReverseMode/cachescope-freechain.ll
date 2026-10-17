; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; In the reverse pass of @g, %p is needed for the derivative of %q. Unwrapping
; %p first unwraps the branch condition %cond, whose phi %cphi moves the
; builder into a new block. Unwrapping %p then gives up, erasing the blocks it
; made, including the lookups of %x from the cache that the reverse pass had
; just created for it. When %x is later restored from the tape,
; cacheForReverse removes that cache: its stores and frees. The free was the
; last user of the cache allocation, so erasing the operands of the free also
; erased the allocation, while scopeMap still held it in an AssertingVH
; ("An asserting value handle still pointed to this value!" on a build with
; LLVM_ENABLE_ABI_BREAKING_CHECKS, a use after free otherwise).

declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)

define void @g(ptr noalias %a, ptr noalias %out, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %par = and i64 %i, 1
  %c0 = icmp eq i64 %par, 0
  %pa = getelementptr inbounds double, ptr %a, i64 %i
  %x = load double, ptr %pa, align 8
  br i1 %c0, label %T, label %J

T:
  %v = load double, ptr %pa, align 8
  %vv = fmul double %v, %v
  %pb = getelementptr inbounds double, ptr %out, i64 %n
  %po2 = getelementptr inbounds double, ptr %pb, i64 %i
  store double %vv, ptr %po2, align 8
  br label %J

J:
  %cphi = phi double [ %v, %T ], [ 1.000000e+00, %loop ]
  %cond = fcmp ogt double %cphi, 0.000000e+00
  br i1 %cond, label %merge, label %Q

Q:
  %x2 = fmul double %x, 3.000000e+00
  br label %merge

merge:
  %p = phi double [ %x, %J ], [ %x2, %Q ]
  %q = fmul double %p, %p
  store double 0.000000e+00, ptr %pa, align 8
  %po = getelementptr inbounds double, ptr %out, i64 %i
  store double %q, ptr %po, align 8
  br label %latch

latch:
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @dg(ptr %a, ptr %da, ptr %out, ptr %dout, i64 %n) {
entry:
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @g, metadata !"enzyme_dup", ptr %a, ptr %da, metadata !"enzyme_dup", ptr %out, ptr %dout, i64 %n)
  call void (...) @__enzyme_reverse(ptr @g, metadata !"enzyme_dup", ptr %a, ptr %da, metadata !"enzyme_dup", ptr %out, ptr %dout, i64 %n, ptr %tape)
  ret void
}

; CHECK: define internal void @diffeg(ptr noalias {{.*}}%a, ptr {{.*}}%"a'", ptr noalias {{.*}}%out, ptr {{.*}}%"out'", i64 %n, ptr %tapeArg)
; CHECK: entry:
; CHECK-NEXT:   %truetape = load { ptr, ptr }, ptr %tapeArg
; CHECK-NEXT:   tail call void @free(ptr nonnull %tapeArg)
; CHECK-DAG:   %[[xcache:.+]] = extractvalue { ptr, ptr } %truetape, 0
; CHECK-DAG:   %[[vcache:.+]] = extractvalue { ptr, ptr } %truetape, 1

; CHECK: loop:
; CHECK:   %[[xptr:.+]] = getelementptr inbounds double, ptr %[[xcache]], i64 %iv
; CHECK-NEXT:   %x = load double, ptr %[[xptr]]
; CHECK-NEXT:   br i1 %c0, label %T, label %J

; CHECK: T:
; CHECK-NEXT:   %[[vptr:.+]] = getelementptr inbounds double, ptr %[[vcache]], i64 %iv
; CHECK-NEXT:   %v = load double, ptr %[[vptr]]
; CHECK-NEXT:   br label %J

; CHECK: invertentry:
; CHECK-DAG:   tail call void @free(ptr nonnull %[[xcache]])
; CHECK-DAG:   tail call void @free(ptr nonnull %[[vcache]])
; CHECK:   ret void
