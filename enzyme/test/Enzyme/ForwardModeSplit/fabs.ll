; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %0 = tail call fast double @llvm.fabs.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwdsplit(double (double)* nonnull @tester, double %x, double 1.0, i8* null)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %[[differet:.+]], i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fcmp fast olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[i1:.+]] = select{{( fast)?}} i1 %[[i0]], double -1.000000e+00, double 1.000000e+00
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[differet]], %[[i1]]
; CHECK-NEXT:   ret double %[[i2]]
; CHECK-NEXT: }
