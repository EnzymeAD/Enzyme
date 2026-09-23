; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Same structure as the sincn.ll test, only substituted double -> x86_fp80 and f64 -> f80. The derivative multiplies by
; pi rounded to x86_fp80, 0xK4000C90FDAA22168C235.

; Function Attrs: nounwind readnone uwtable
define x86_fp80 @tester(x86_fp80 %x) {
entry:
  %0 = tail call fast x86_fp80 @sincnl(x86_fp80 %x)
  ret x86_fp80 %0
}

define x86_fp80 @test_derivative(x86_fp80 %x) {
entry:
  %0 = tail call x86_fp80 (x86_fp80 (x86_fp80)*, ...) @__enzyme_autodiff(x86_fp80 (x86_fp80)* nonnull @tester, x86_fp80 %x)
  ret x86_fp80 %0
}

; Function Attrs: nounwind readnone speculatable
declare x86_fp80 @sincnl(x86_fp80)

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_autodiff(x86_fp80 (x86_fp80)*, ...)

; CHECK: define internal { x86_fp80 } @diffetester(x86_fp80 %x, x86_fp80 %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[cmp:.+]] = fcmp fast oeq x86_fp80 %x, 0xK00000000000000000000
; CHECK-NEXT:   %[[a0:.+]] = fmul fast x86_fp80 0xK4000C90FDAA22168C235, %x
; CHECK-NEXT:   %[[i0:.+]] = call fast x86_fp80 @llvm.cos.f80(x86_fp80 %[[a0]])
; CHECK-NEXT:   %[[i1:.+]] = call fast x86_fp80 @sincnl(x86_fp80 %x)
; CHECK-NEXT:   %[[i2:.+]] = fsub fast x86_fp80 %[[i0]], %[[i1]]
; CHECK-NEXT:   %[[i3:.+]] = fdiv fast x86_fp80 %[[i2]], %x
; CHECK-NEXT:   %[[i4:.+]] = fmul fast x86_fp80 %differeturn, %[[i3]]
; CHECK-NEXT:   %[[sel:.+]] = select {{(fast )?}}i1 %[[cmp]], x86_fp80 0xK00000000000000000000, x86_fp80 %[[i4]]
; CHECK-NEXT:   %[[i5:.+]] = insertvalue { x86_fp80 } undef, x86_fp80 %[[sel]], 0
; CHECK-NEXT:   ret { x86_fp80 } %[[i5]]
; CHECK-NEXT: }
