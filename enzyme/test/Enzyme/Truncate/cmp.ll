; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

define i1 @f(double %x, double %y) {
  %res = fcmp olt double %x, %y
  ret i1 %res
}

declare i1 (double, double)* @__enzyme_truncate(...)

define i1 @tester(double %x, double %y) {
entry:
  %ptr = call i1 (double, double)* (...) @__enzyme_truncate(i1 (double, double)* @f, i64 64, i64 32)
  %res = call i1 %ptr(double %x, double %y)
  ret i1 %res
}

; CHECK: define i1 @tester(double %x, double %y) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res = call i1 @trunc_64_32f(double %x, double %y)
; CHECK-NEXT:   ret i1 %res
; CHECK-NEXT: }

; CHECK: define internal i1 @trunc_64_32f(double %x, double %y) {
; CHECK-NEXT:   %1 = alloca double, align 8
; CHECK-NEXT:   store double %y, double* %1, align 8
; CHECK-NEXT:   %2 = bitcast double* %1 to float*
; CHECK-NEXT:   %3 = load float, float* %2, align 4
; CHECK-NEXT:   store double %x, double* %1, align 8
; CHECK-NEXT:   %4 = bitcast double* %1 to float*
; CHECK-NEXT:   %5 = load float, float* %4, align 4
; CHECK-NEXT:   %res = fcmp olt float %5, %3
; CHECK-NEXT:   ret i1 %res
; CHECK-NEXT: }
