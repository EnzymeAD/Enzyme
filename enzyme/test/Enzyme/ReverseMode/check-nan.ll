; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-check-nan -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-check-nan -S | FileCheck %s

define double @f(double %x) {
entry:
  %m = fmul double %x, %x
  ret double %m
}

define double @d_f(double %x) {
entry:
  %r = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @f, double %x)
  ret double %r
}

declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double } @diffef(double %x, double %differeturn)
; CHECK: invert
; CHECK:   call void @__enzyme_sanitize_nan_double(double %differeturn, i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{.*}}, i32 0, i32 0))
; CHECK:   %[[a0:.+]] = load double, double* %"m'de"
; CHECK:   call void @__enzyme_sanitize_nan_double(double 0.000000e+00, i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{.*}}, i32 0, i32 0))
; CHECK:   %[[a1:.+]] = fmul fast double %[[a0]], %x
; CHECK:   %[[a2:.+]] = load double, double* %"x'de"
; CHECK:   %[[add1:.+]] = fadd fast double %[[a2]], %[[a1]]
; CHECK-NEXT:   call void @__enzyme_sanitize_nan_double(double %[[add1]], i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{.*}}, i32 0, i32 0))
; CHECK:   %[[a3:.+]] = fmul fast double %[[a0]], %x
; CHECK:   %[[a4:.+]] = load double, double* %"x'de"
; CHECK:   %[[add2:.+]] = fadd fast double %[[a4]], %[[a3]]
; CHECK-NEXT:   call void @__enzyme_sanitize_nan_double(double %[[add2]], i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{.*}}, i32 0, i32 0))
; CHECK:   store double %[[add2]], double* %"x'de"
; CHECK:   %[[res_load:.+]] = load double, double* %"x'de"
; CHECK:   %[[res:.+]] = insertvalue { double } {{(undef|poison)}}, double %[[res_load]], 0
; CHECK-NEXT:   ret { double } %[[res]]

; CHECK: define internal void @__enzyme_sanitize_nan_double(double %0, i8* %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = fcmp uno double %0, %0
; CHECK-NEXT:   br i1 %2, label %bad, label %good

; CHECK: good:                                             ; preds = %entry
; CHECK-NEXT:   ret void

; CHECK: bad:                                              ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @puts(i8* %1)
; CHECK-NEXT:   call void @exit(i32 1)
; CHECK-NEXT:   unreachable
