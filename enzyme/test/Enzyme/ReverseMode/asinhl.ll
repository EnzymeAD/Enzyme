; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define x86_fp80 @tester(x86_fp80 %x) {
entry:
  %0 = tail call fast x86_fp80 @asinhl(x86_fp80 %x)
  ret x86_fp80 %0
}

define x86_fp80 @test_derivative(x86_fp80 %x) {
entry:
  %0 = tail call x86_fp80 (x86_fp80 (x86_fp80)*, ...) @__enzyme_autodiff(x86_fp80 (x86_fp80)* nonnull @tester, x86_fp80 %x)
  ret x86_fp80 %0
}

; Function Attrs: nounwind readnone speculatable
declare x86_fp80 @asinhl(x86_fp80)

; Function Attrs: nounwind
declare x86_fp80 @__enzyme_autodiff(x86_fp80 (x86_fp80)*, ...)

; CHECK: define internal { x86_fp80 } @diffetester(x86_fp80 %x, x86_fp80 %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'de" = alloca x86_fp80, align 16
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %"'de", align 16
; CHECK-NEXT:   %"x'de" = alloca x86_fp80, align 16
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   store x86_fp80 %differeturn, x86_fp80* %"'de", align 16
; CHECK-NEXT:   %0 = load x86_fp80, x86_fp80* %"'de", align 16
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %"'de", align 16
; CHECK-NEXT:   %1 = fmul fast x86_fp80 %x, %x
; CHECK-NEXT:   %2 = fadd fast x86_fp80 %1, 0xK3FFF8000000000000000
; CHECK-NEXT:   %3 = call fast x86_fp80 @llvm.sqrt.f80(x86_fp80 %2)
; CHECK-NEXT:   %4 = fdiv fast x86_fp80 %0, %3
; CHECK-NEXT:   %5 = load x86_fp80, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   %6 = fadd fast x86_fp80 %5, %4
; CHECK-NEXT:   store x86_fp80 %6, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   %7 = load x86_fp80, x86_fp80* %"x'de", align 16
; CHECK-NEXT:   %8 = insertvalue { x86_fp80 } undef, x86_fp80 %7, 0
; CHECK-NEXT:   ret { x86_fp80 } %8
; CHECK-NEXT: }

