; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

define { double, i64 } @julia_logabsgamma_3264_inner.1(double %x, i64 %z) {
entry:
  %iadd = add i64 %z, 1
  %.fca.0.insert = insertvalue { double, i64 } undef, double %x, 0
  %.fca.1.insert = insertvalue { double, i64 } %.fca.0.insert, i64 %iadd, 1
  ret { double, i64 } %.fca.1.insert
}

declare { double, i64 } @__enzyme_fwddiff(...)

define { double, i64 } @ad(double %x, double %dx) {
  %m = call  { double, i64 } (...) @__enzyme_fwddiff({ double, i64 } (double, i64)* @julia_logabsgamma_3264_inner.1, double %x, double %dx, i64 1)
  ret { double, i64 } %m
}

; CHECK: define internal { double, i64 } @fwddiffejulia_logabsgamma_3264_inner.1(double %x, double %"x'", i64 %z) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %iadd = add i64 %z, 1
; CHECK-NEXT:   %".fca.0.insert'ipiv" = insertvalue { double, i64 } zeroinitializer, double %"x'", 0
; CHECK-NEXT:   %".fca.1.insert'ipiv" = insertvalue { double, i64 } %".fca.0.insert'ipiv", i64 %iadd, 1
; CHECK-NEXT:   ret { double, i64 } %".fca.1.insert'ipiv"
; CHECK-NEXT: }