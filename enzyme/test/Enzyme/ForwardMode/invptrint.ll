; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

define { { double*, i64 }, double } @z0({ double*, i64 } %const, double %act) {
bb:
  %e = extractvalue { double*, i64 } %const, 0
  %i = extractvalue { double*, i64 } %const, 1
  %g = getelementptr inbounds double, double* %e, i64 %i
  %ld = load double, double* %g, align 8
  %a = fadd double %ld, %act
  %res = insertvalue { { double*, i64 }, double } undef, { double*, i64 } %const, 0
  %res2 = insertvalue { { double*, i64 }, double } %res, double %a, 1
  ret { { double*, i64 }, double } %res2
}

declare { { double*, i64 }, double } @__enzyme_fwddiff(...)

define { { double*, i64 }, double } @dsquare() {
bb:
  %r = call { { double*, i64 }, double } (...) @__enzyme_fwddiff({ { double*, i64 }, double } ({ double*, i64 }, double)* nonnull @z0, metadata !"enzyme_const", { double*, i64 } undef, double 1.0, double 1.0)
  ret { { double*, i64 }, double } %r
}

; CHECK: define internal { { double*, i64 }, double } @fwddiffez0({ double*, i64 } %const, double %act, double %"act'")
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"res'ipiv" = insertvalue { { double*, i64 }, double } zeroinitializer, { double*, i64 } %const, 0
; CHECK-NEXT:   %res = insertvalue { { double*, i64 }, double } undef, { double*, i64 } %const, 0
; CHECK-NEXT:   %"res2'ipiv" = insertvalue { { double*, i64 }, double } %"res'ipiv", double %"act'", 1
; CHECK-NEXT:   ret { { double*, i64 }, double } %"res2'ipiv"
; CHECK-NEXT: }
