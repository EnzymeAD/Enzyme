; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Regression test: the Numba runtime's reference-count release, NRT_decref,
; must be recognized as a deallocation function (isDeallocationFunction),
; the same way swift_release / __rust_dealloc / _mlir_memref_to_llvm_free are.
;
; This uses the actual Numba NRT functions: an array is allocated with
; NRT_MemInfo_alloc_aligned and released with NRT_decref (the IR Numba emits).
; The allocation is registered with __enzyme_allocation_like so Enzyme knows
; how to shadow it self-contained; its shadow-free is registered as @free, NOT
; NRT_decref -- so the recognition of the SOURCE NRT_decref as a deallocation
; depends solely on isDeallocationFunction (the change under test), not on the
; allocation/free pairing.
;
; With the recognition, the allocation/NRT_decref pair is dead in the forward
; pass, so the augmented function is empty and the reverse pass re-allocates a
; shadow (via NRT_MemInfo_alloc_aligned) and frees it once, with no NRT_decref
; duplicated into the derivative.  Before recognizing NRT_decref, differentiating
; @sum fails outright with "Enzyme: No augmented forward pass found for
; NRT_decref".  Pointer spellings are matched loosely so the test passes under
; both the typed-pointer (LLVM <= 16) and opaque-pointer (LLVM >= 17) forms.

@.dealloc = private unnamed_addr constant [3 x i8] c"-1\00"
@__enzyme_allocation_like = global [4 x i8*] [
  i8* bitcast (i8* (i64, i32)* @NRT_MemInfo_alloc_aligned to i8*),
  i8* inttoptr (i64 0 to i8*),
  i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.dealloc, i64 0, i64 0),
  i8* bitcast (void (i8*)* @free to i8*)
]

declare noalias i8* @NRT_MemInfo_alloc_aligned(i64, i32)
declare void @NRT_decref(i8*)
declare void @free(i8*)

define dso_local double @subsum(double* nocapture readonly %x, i64 %n) {
entry:
  %m = call i8* @NRT_MemInfo_alloc_aligned(i64 8, i32 32)
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

; The recognized allocation/NRT_decref pair is dead in the forward pass, so the
; augmented function caches nothing and is empty -- impossible unless the
; NRT_decref is understood to free the allocation.
; CHECK: define internal void @augmented_subsum(
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
; CHECK-NEXT: }

; The reverse pass re-allocates a shadow with the real NRT allocator and frees
; it exactly once.  NRT_decref is never duplicated into the derivative.
; CHECK: define internal void @diffesubsum(
; CHECK: %"m'mi" = call {{.*}}@NRT_MemInfo_alloc_aligned(i64 8, i32 32)
; CHECK: call void @free({{.*}}%"m'mi")
; CHECK-NOT: NRT_decref
