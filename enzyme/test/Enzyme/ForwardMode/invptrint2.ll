; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

define { { double*, i64, double }, double } @z0({ double*, i64, double } %const, double %act) {
bb:
  %e = extractvalue { double*, i64, double } %const, 0
  %i = extractvalue { double*, i64, double } %const, 1
  %q = extractvalue { double*, i64, double } %const, 2
  %g = getelementptr inbounds double, double* %e, i64 %i
  %ld = load double, double* %g, align 8
  %a = fadd double %ld, %act
  %b = fadd double %a, %q
  %res = insertvalue { { double*, i64, double }, double } undef, { double*, i64, double } %const, 0
  %res2 = insertvalue { { double*, i64, double }, double } %res, double %b, 1
  ret { { double*, i64, double }, double } %res2
}

declare { { double*, i64, double }, double } @__enzyme_fwddiff(...)

define { { double*, i64, double }, double } @dsquare() {
bb:
  %r = call { { double*, i64, double }, double } (...) @__enzyme_fwddiff({ { double*, i64, double }, double } ({ double*, i64, double }, double)* nonnull @z0, metadata !"enzyme_const", { double*, i64, double } undef, double 1.0, double 1.0)
  ret { { double*, i64, double }, double } %r
}

; CHECK: define internal { { double*, i64, double }, double } @fwddiffez0({ double*, i64, double } %const, double %act, double %"act'")
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = alloca { double*, i64, double }
; CHECK-NEXT:   store { double*, i64, double } %const, { double*, i64, double }* %0
; CHECK-NEXT:   %1 = bitcast { double*, i64, double }* %0 to [24 x i8]*
; CHECK-NEXT:   %2 = getelementptr inbounds [24 x i8], [24 x i8]* %1, i32 0, i32 16
; CHECK-NEXT:   %3 = bitcast i8* %2 to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %3
; CHECK-NEXT:   %4 = load { double*, i64, double }, { double*, i64, double }* %0
; CHECK-NEXT:   %"res'ipiv" = insertvalue { { double*, i64, double }, double } zeroinitializer, { double*, i64, double } %4, 0
; CHECK-NEXT:   %res = insertvalue { { double*, i64, double }, double } undef, { double*, i64, double } %const, 0
; CHECK-NEXT:   %"res2'ipiv" = insertvalue { { double*, i64, double }, double } %"res'ipiv", double %"act'", 1
; CHECK-NEXT:   ret { { double*, i64, double }, double } %"res2'ipiv"
; CHECK-NEXT: }
