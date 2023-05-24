; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,instcombine,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fsub fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwdsplit(double (double, double)* nonnull @tester, double %x, double 1.0, double %y, double 0.0, i8* null)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwdsplit(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", double %y, double %"y'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fsub fast double %"x'", %"y'"
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
