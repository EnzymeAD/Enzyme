;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define double @wrapper(double* %x) {
entry:
  %call = tail call double @cblas_dnrm2(i32 3, double* %x, i32 1)
  %mul = fmul double %call, %call
  ret double %mul
}

declare double @cblas_dnrm2(i32, double*, i32)

define void @caller(double* %x, double* %_x) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (double (double*)* @wrapper to i8*), double* %x, double* %_x)
  ret void
}

declare double @__enzyme_autodiff(i8*, double*, double*)

;CHECK:define internal void @diffewrapper(double* %x, double* %"x'", double %differeturn) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  %call = tail call double @cblas_dnrm2(i32 3, double* %x, i32 1)
;CHECK-NEXT:  %m0diffecall = fmul fast double %differeturn, %call
;CHECK-NEXT:  %m1diffecall = fmul fast double %differeturn, %call
;CHECK-NEXT:  %0 = fadd fast double %m0diffecall, %m1diffecall
;CHECK-NEXT:  %1 = fdiv fast double %0, %call
;CHECK-NEXT:  call void @cblas_daxpy(i32 3, double %1, double* %x, i32 1, double* %"x'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
