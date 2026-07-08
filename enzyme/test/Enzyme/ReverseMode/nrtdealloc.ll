; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Regression test: the Numba runtime's reference-count release, NRT_decref,
; must be recognized as a deallocation function (isDeallocationFunction),
; the same way swift_release / __rust_dealloc / _mlir_memref_to_llvm_free are.
;
; Numba frees a heap allocation with a paired NRT_decref. Unless Enzyme knows
; NRT_decref frees its argument, it cannot pair the release with the allocation
; and cannot manage the allocation's lifetime across the augmented/reverse
; split (downstream, where NRT_decref is additionally marked enzyme_inactive,
; the unrecognized release was instead duplicated across both passes, separately
; freeing the allocation and corrupting the heap).
;
; This test isolates the deallocation recognition using a built-in-recognized
; allocation (malloc) whose only release is NRT_decref -- the malloc/free-nouse
; pattern of alloctomallocnouse.ll, with free replaced by NRT_decref. With the
; recognition, Enzyme treats the pair exactly as malloc/free: the value is dead
; in the forward pass, so the augmented function is empty and the reverse pass
; re-allocates a shadow and frees it once. (Enzyme emits malloc's canonical
; deallocator, @free, for its own shadow allocation; NRT_decref itself is a
; Numba allocation's deallocator and is paired downstream, so it does not
; appear in this self-contained module -- what is tested here is that the
; source NRT_decref is understood as a free.)
;
; Before recognizing NRT_decref, differentiating @sum fails outright with
; "Enzyme: No augmented forward pass found for NRT_decref" (the release has no
; known derivative and is not a deallocation); with the recognition it compiles
; to the IR checked below.

declare i8* @malloc(i64)
declare void @NRT_decref(i8*)

define dso_local double @subsum(double* nocapture readonly %x, i64 %n) {
entry:
  %m = call i8* @malloc(i64 8)
  %v = bitcast i8* %m to double*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %res = load double, double* %v
  call void @NRT_decref(i8* %m)
  ret double %res

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %total.07 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %add = fadd fast double %0, %total.07
  store double %add, double* %v
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define dso_local double @sum(double* nocapture readonly %x, i64 %n) {
entry:
  %res = call double @subsum(double* %x, i64 %n)
  store double 0.000000e+00, double* %x
  ret double %res
}

define dso_local void @dsum(double* %x, double* %xp, i64 %n) {
entry:
  %0 = tail call double (double (double*, i64)*, ...) @__enzyme_autodiff(double (double*, i64)* nonnull @sum, double* %x, double* %xp, i64 %n)
  ret void
}

declare double @__enzyme_autodiff(double (double*, i64)*, ...)

; The recognized malloc/NRT_decref pair is dead in the forward pass, so the
; augmented function caches nothing and is empty -- impossible unless the
; NRT_decref is understood to free the malloc.
; CHECK: define internal void @augmented_subsum(ptr nocapture readonly %x, ptr %"x'", i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; The reverse pass re-allocates a shadow and frees it exactly once (via
; malloc's canonical deallocator).  NRT_decref is never duplicated into the
; derivative.
; CHECK: define internal void @diffesubsum(ptr nocapture readonly %x, ptr %"x'", i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m'mi" = call {{.*}}ptr @malloc(i64 8)
; CHECK: call void @free(ptr nonnull %"m'mi")
; CHECK-NOT: NRT_decref
