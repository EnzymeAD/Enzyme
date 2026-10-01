; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-strict-aliasing=0 -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-strict-aliasing=0 -S | FileCheck %s

; The loopy reduction phi %acc has a constant (rather than instruction) incoming
; value from the preheader. Such a value is always available and must not be
; added to the min-cut recompute graph, which only contains instructions.

define double @foo(double* %p) {
entry:
  br label %loop

exit:
  %res = call double @llvm.log.f64(double %sel)
  ret double %res

loop:
  %acc = phi double [ 0.000000e+00, %entry ], [ %sel, %loop ]
  %cmp = fcmp ogt double %acc, 0.000000e+00
  %ld = load double, double* %p, align 8
  %sel = select i1 %cmp, double %acc, double %ld
  %exitcond = icmp eq i64 0, 0
  br i1 %exitcond, label %exit, label %loop
}

define void @dtarget(double* %p, double* %dp) {
entry:
  %r = call double (...) @__enzyme_autodiff(double (double*)* @foo, double* %p, double* %dp)
  ret void
}

declare double @__enzyme_autodiff(...)

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
declare double @llvm.log.f64(double) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn }

; CHECK: define internal {{(dso_local )?}}void @diffefoo({{i8\*|ptr|double\*}}
