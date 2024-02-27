; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define x86_fp80 @tester(x86_fp80 %x) {
entry:
  %0 = tail call fast x86_fp80 @coshl(x86_fp80 %x)
  ret x86_fp80 %0
}

define x86_fp80 @test_derivative(x86_fp80 %x) {
entry:
  %0 = tail call x86_fp80 (x86_fp80 (x86_fp80)*, ...) @__enzyme_autodiff(x86_fp80 (x86_fp80)* nonnull @tester, x86_fp80 %x)
  ret x86_fp80 %0
}

; Function Attrs: nounwind readnone speculatable
declare x86_fp80 @coshl(x86_fp80)

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_autodiff(x86_fp80 (x86_fp80)*, ...)

; CHECK: define internal { x86_fp80 } @diffetester(x86_fp80 %x, x86_fp80 %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast x86_fp80 @sinhl(x86_fp80 %x)
; CHECK-NEXT:   %1 = fmul fast x86_fp80 %differeturn, %0
; CHECK-NEXT:   %2 = insertvalue { x86_fp80 } undef, x86_fp80 %1, 0
; CHECK-NEXT:   ret { x86_fp80 } %2
; CHECK-NEXT: }
