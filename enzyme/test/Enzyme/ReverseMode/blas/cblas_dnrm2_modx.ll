;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define double @wrapper(i32 %n, double* %x, i32 %incx) {
entry:
  %call = tail call double @cblas_dnrm2(i32 %n, double* %x, i32 %incx)
  store double 0.000000e+00, double* %x, align 8
  ret double %call
}

declare double @cblas_dnrm2(i32, double*, i32)

define void @caller(i32 %n, double* %x, double* %_x, i32 %incx) {
entry:
  %call = tail call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (i32, double*, i32)* @wrapper to i8*), i32 %n, double* %x, double* %_x, i32 %incx)
  ret void
}

declare double @__enzyme_autodiff(i8*, ...)

;CHECK:define internal void @diffewrapper(i32 %n, double* %x, double* %"x'", i32 %incx, double %differeturn) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = zext i32 %n to i64
;CHECK-NEXT:  %mallocsize = mul i64 %0, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %1 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %x, i32 %n, i32 %incx)
;CHECK-NEXT:  %call = tail call double @cblas_dnrm2(i32 %n, double* %x, i32 %incx)
;CHECK-NEXT:  store double 0.000000e+00, double* %x, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"x'", align 8
;CHECK-NEXT:  %2 = fdiv fast double %differeturn, %call
;CHECK-NEXT:  call void @cblas_daxpy(i32 %n, double %2, double* %1, i32 1, double* %"x'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
