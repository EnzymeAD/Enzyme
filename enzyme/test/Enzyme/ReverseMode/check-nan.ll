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

; CHECK: define internal { double } @diffef(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"m'de", align 8
; CHECK-NEXT:   %"x'de" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"x'de", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   call void @__enzyme_sanitize_nan_double(double %differeturn, i8* getelementptr inbounds ([75 x i8], [75 x i8]* @.str, i32 0, i32 0))
; CHECK-NEXT:   store double %differeturn, double* %"m'de", align 8
; CHECK-NEXT:   %0 = load double, double* %"m'de", align 8
; CHECK-NEXT:   call void @__enzyme_sanitize_nan_double(double 0.000000e+00, i8* getelementptr inbounds ([75 x i8], [75 x i8]* @.str.1, i32 0, i32 0))
; CHECK-NEXT:   store double 0.000000e+00, double* %"m'de", align 8
; CHECK-NEXT:   %1 = fmul fast double %0, %x
; CHECK-NEXT:   %2 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %1
; CHECK-NEXT:   call void @__enzyme_sanitize_nan_double(double %3, i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.2, i32 0, i32 0))
; CHECK-NEXT:   store double %3, double* %"x'de", align 8
; CHECK-NEXT:   %4 = fmul fast double %0, %x
; CHECK-NEXT:   %5 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %6 = fadd fast double %5, %4
; CHECK-NEXT:   call void @__enzyme_sanitize_nan_double(double %6, i8* getelementptr inbounds ([59 x i8], [59 x i8]* @.str.3, i32 0, i32 0))
; CHECK-NEXT:   store double %6, double* %"x'de", align 8
; CHECK-NEXT:   %7 = load double, double* %"x'de", align 8
; CHECK-NEXT:   %8 = insertvalue { double } undef, double %7, 0
; CHECK-NEXT:   ret { double } %8
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_sanitize_nan_double(double %0, i8* %1) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = fcmp uno double %0, %0
; CHECK-NEXT:   br i1 %2, label %bad, label %good

; CHECK: good:                                             ; preds = %entry
; CHECK-NEXT:   ret void

; CHECK: bad:                                              ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @puts(i8* %1)
; CHECK-NEXT:   call void @exit(i32 1)
; CHECK-NEXT:   unreachable
; CHECK-NEXT: }
