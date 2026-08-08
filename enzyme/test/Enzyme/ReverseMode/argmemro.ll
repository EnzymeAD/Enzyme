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

; CHECK: define internal void @diffesumsq(ptr readonly captures(none) %arr, ptr captures(none) %"arr'", i64 %n, double %differeturn)
; CHECK: %v_unwrap = load double, ptr %g_unwrap
; CHECK: %"g'ipg_unwrap" = getelementptr inbounds double, ptr %"arr'"
