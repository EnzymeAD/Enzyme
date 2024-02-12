; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false  -enzyme -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg)" -S | FileCheck %s

; Same structure as the div.ll test, only substituted double -> x86_fp80.

; Function Attrs: noinline nounwind readnone uwtable
define x86_fp80 @tester(x86_fp80 %x, x86_fp80 %y) {
entry:
  %0 = fdiv fast x86_fp80 %x, %y
  ret x86_fp80 %0
}

define x86_fp80 @test_derivative(x86_fp80 %x, x86_fp80 %y) {
entry:
  %0 = tail call x86_fp80 (x86_fp80 (x86_fp80, x86_fp80)*, ...) @__enzyme_autodiff(x86_fp80 (x86_fp80, x86_fp80)* nonnull @tester, x86_fp80 %x, x86_fp80 %y)
  ret x86_fp80 %0
}

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_autodiff(x86_fp80 (x86_fp80, x86_fp80)*, ...)

; CHECK: define internal {{(dso_local )?}}{ x86_fp80, x86_fp80 } @diffetester(x86_fp80 %x, x86_fp80 %y, x86_fp80 %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[diffex:.+]] = fdiv fast x86_fp80 %[[differet]], %y
; CHECK-NEXT:   %[[xdivy:.+]] = fdiv fast x86_fp80 %x, %y
; CHECK-NEXT:   %[[xdivydret:.+]] = fmul fast x86_fp80 %[[diffex]], %[[xdivy]]
; CHECK-NEXT:   %[[mxdivy2:.+]] = {{(fsub fast x86_fp80 0.000000e\+00,|fneg fast x86_fp80)}} %[[xdivydret]]
; CHECK-NEXT:   %[[res1:.+]] = insertvalue { x86_fp80, x86_fp80 } undef, x86_fp80 %[[diffex]], 0
; CHECK-NEXT:   %[[res2:.+]] = insertvalue { x86_fp80, x86_fp80 } %[[res1:.+]], x86_fp80 %[[mxdivy2]], 1
; CHECK-NEXT:   ret { x86_fp80, x86_fp80 } %[[res2]]
; CHECK-NEXT: }
