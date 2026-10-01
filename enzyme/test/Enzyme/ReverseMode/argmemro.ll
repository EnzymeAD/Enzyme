; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; A call whose memory(...) attribute leaves argument memory read-only does
; not write the active slot it is handed, even when the function may write
; elsewhere -- so an inactive observer between the store and the load needs
; no derivative, and the reverse pass may recompute the load from the
; primal instead of demanding one.

declare void @observe(double* nocapture) "enzyme_inactive" nofree memory(argmem: read, inaccessiblemem: readwrite)

define double @sumsq(double* %arr, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sum2, %loop ]
  %g = getelementptr inbounds double, double* %arr, i64 %i
  call void @observe(double* %g)
  %v = load double, double* %g, align 8
  %sq = fmul double %v, %v
  %sum2 = fadd double %sum, %sq
  %inext = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %inext, %n
  br i1 %cmp, label %exit, label %loop

exit:
  ret double %sum2
}

declare double @__enzyme_autodiff(...)

define double @dsumsq(double* %arr, double* %darr, i64 %n) {
entry:
  %r = call double (...) @__enzyme_autodiff(double (double*, i64)* @sumsq, metadata !"enzyme_dup", double* %arr, double* %darr, i64 %n)
  ret double %r
}

; A second observer whose unlisted "other" memory may be written: even though
; argument memory itself is read-only, the written other memory can alias the
; bytes the argument points to, so the loaded value cannot be recomputed in
; the reverse pass -- it must be cached from the forward pass instead.

; enzyme_no_escaping_allocation keeps the check about caching: without it,
; opaque-pointer IR cannot rule out an allocation being stored into the
; argument (typed IR can, from the pointee type), and the inactive fallback
; is skipped altogether on newer LLVM.
declare void @observe2(double* nocapture) "enzyme_inactive" "enzyme_no_escaping_allocation" nofree memory(readwrite, argmem: read)

define double @sumsq2(double* %arr, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sum2, %loop ]
  %g = getelementptr inbounds double, double* %arr, i64 %i
  call void @observe2(double* %g)
  %v = load double, double* %g, align 8
  %sq = fmul double %v, %v
  %sum2 = fadd double %sum, %sq
  %inext = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %inext, %n
  br i1 %cmp, label %exit, label %loop

exit:
  ret double %sum2
}

define double @dsumsq2(double* %arr, double* %darr, i64 %n) {
entry:
  %r = call double (...) @__enzyme_autodiff(double (double*, i64)* @sumsq2, metadata !"enzyme_dup", double* %arr, double* %darr, i64 %n)
  ret double %r
}

; The attribute spelling of the arguments differs between typed-pointer
; (LLVM 16, "double* nocapture readonly") and opaque-pointer IR
; ("ptr readonly captures(none)"), so the arguments are matched loosely.

; CHECK: define internal void @diffesumsq({{.*}} %arr, {{.*}} %"arr'", i64 %n, double %differeturn)
; CHECK: %v_unwrap = load double, {{(double\*|ptr)}} %g_unwrap
; CHECK: %"g'ipg_unwrap" = getelementptr inbounds double, {{(double\*|ptr)}} %"arr'"

; The observe2 version caches %v in the forward loop rather than reloading it
; in the reverse.
; CHECK: define internal void @diffesumsq2({{.*}} %arr, {{.*}} %"arr'", i64 %n, double %differeturn)
; CHECK: @malloc(
; CHECK: %[[v2:.+]] = load double, {{(double\*|ptr)}} %g
; CHECK: store double %[[v2]], {{(double\*|ptr)}} %
