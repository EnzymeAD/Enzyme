; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define { double, i64 } @tester(double %x) {
entry:
  %a = call { double, i64 } @logabsgamma(double %x)
  ret { double, i64 } %a
}

define { double, i64 } @test_derivative(double %x, double %dx) {
entry:
  %0 = tail call { double, i64 } (...) @__enzyme_fwddiff({ double, i64 } (double)* nonnull @tester, double %x, double %dx)
  ret { double, i64 } %0
}

declare { double, i64 } @logabsgamma(double)

; Function Attrs: nounwind
declare { double, i64 } @__enzyme_fwddiff(...)

; CHECK: define internal { double, i64 } @fwddiffetester(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @digamma(double %x)
; CHECK-NEXT:   %1 = fmul fast double %0, %"x'"
; CHECK-NEXT:   %2 = insertvalue { double, i64 } undef, double %1, 0
; CHECK-NEXT:   ret { double, i64 } %2
; CHECK-NEXT: }
