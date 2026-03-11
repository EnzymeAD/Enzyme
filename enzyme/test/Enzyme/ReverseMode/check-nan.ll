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

; CHECK: define internal {{(dso_local )?}}void @diffef(double %x, double %differeturn)
; CHECK: entry:
; CHECK:   %m = fmul double %x, %x
; CHECK:   %0 = fmul fast double %differeturn, %x
; CHECK:   %1 = fmul fast double %differeturn, %x
; CHECK:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   call void @__enzyme_sanitize_nan_double(double %2, i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{.*}}, i32 0, i32 0))
; CHECK-NEXT:   ret void

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
