; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg)" -S | FileCheck %s

; Same structure as the sincospi.ll test, only substituted double -> x86_fp80. The derivative multiplies by pi rounded to
; x86_fp80, 0xK4000C90FDAA22168C235.

; Function Attrs: nounwind readnone uwtable
define [2 x x86_fp80] @meta(x86_fp80 %x) {
entry:
  %0 = tail call [2 x x86_fp80] @sincospil(x86_fp80 %x)
  ret [2 x x86_fp80] %0
}

define x86_fp80 @tester(x86_fp80 %x) {
entry:
  %0 = tail call [2 x x86_fp80] @meta(x86_fp80 %x)
  %e = extractvalue [2 x x86_fp80] %0, 0
  ret x86_fp80 %e
}

define x86_fp80 @test_derivative(x86_fp80 %x) {
entry:
  %0 = tail call x86_fp80 (...) @__enzyme_autodiff(x86_fp80 (x86_fp80)* nonnull @tester, x86_fp80 %x)
  ret x86_fp80 %0
}

declare [2 x x86_fp80] @sincospil(x86_fp80)

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_autodiff(...)

; CHECK: define internal { x86_fp80 } @diffemeta(x86_fp80 %x, [2 x x86_fp80] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call {{(fast )?}}[2 x x86_fp80] @sincospil(x86_fp80 %x)
; CHECK-NEXT:   %1 = extractvalue [2 x x86_fp80] %0, 1
; CHECK-NEXT:   %2 = extractvalue [2 x x86_fp80] %differeturn, 0
; CHECK-NEXT:   %3 = fmul fast x86_fp80 %1, %2
; CHECK-NEXT:   %4 = fmul fast x86_fp80 0xK4000C90FDAA22168C235, %3
; CHECK-NEXT:   %5 = extractvalue [2 x x86_fp80] %0, 0
; CHECK-NEXT:   %6 = extractvalue [2 x x86_fp80] %differeturn, 1
; CHECK-NEXT:   %7 = fmul fast x86_fp80 %5, %6
; CHECK-NEXT:   %8 = {{(fsub fast x86_fp80 0xK80000000000000000000,|fneg fast x86_fp80)}} %7
; CHECK-NEXT:   %9 = fmul fast x86_fp80 0xK4000C90FDAA22168C235, %8
; CHECK-NEXT:   %10 = fadd fast x86_fp80 %4, %9
; CHECK-NEXT:   %11 = insertvalue { x86_fp80 } undef, x86_fp80 %10, 0
; CHECK-NEXT:   ret { x86_fp80 } %11
; CHECK-NEXT: }
