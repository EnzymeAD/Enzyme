; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

define void @f(double* %x) {
  %y = load double, double* %x
  %m = fmul double %y, %y
  store double %m, double* %x
  ret void
}

declare void (double*)* @__enzyme_truncate(...)

define void @tester(double* %data) {
entry:
  %ptr = call void (double*)* (...) @__enzyme_truncate(void (double*)* @f, i64 64, i64 32)
  call void %ptr(double* %data)
  ret void
}

; CHECK: define void @tester(double* %data)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @trunc_64_32f(double* %data)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @trunc_64_32f(double* %x)
; CHECK-NEXT:   %1 = alloca double, align 8
; CHECK-NEXT:   %y = load double, double* %x, align 8
; CHECK-NEXT:   store double %y, double* %1, align 8
; CHECK-NEXT:   %2 = bitcast double* %1 to float*
; CHECK-NEXT:   %3 = load float, float* %2, align 4
; CHECK-NEXT:   store double %y, double* %1, align 8
; CHECK-NEXT:   %4 = bitcast double* %1 to float*
; CHECK-NEXT:   %5 = load float, float* %4, align 4
; CHECK-NEXT:   %m = fmul float %5, %3
; CHECK-NEXT:   %6 = bitcast double* %1 to i64*
; CHECK-NEXT:   store i64 0, i64* %6, align 4
; CHECK-NEXT:   %7 = bitcast double* %1 to float*
; CHECK-NEXT:   store float %m, float* %7, align 4
; CHECK-NEXT:   %8 = load double, double* %1, align 8
; CHECK-NEXT:   store double %8, double* %x, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }