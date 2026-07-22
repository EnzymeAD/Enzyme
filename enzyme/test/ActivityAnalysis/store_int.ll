; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s

define double @f(double %x, i8* %ptr) {
entry:
  %a = alloca i8, align 1
  %val = load i8, i8* %ptr, align 1
  store i8 %val, i8* %a, align 1
  ret double %x
}

; CHECK: double %x: icv:0
; CHECK-NEXT: {{i8\*|ptr}} %ptr: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %a = alloca i8, align 1: icv:1 ici:1
; CHECK-NEXT:   %val = load i8, {{i8\*|ptr}} %ptr, align 1: icv:1 ici:1
; CHECK-NEXT:   store i8 %val, {{i8\*|ptr}} %a, align 1: icv:1 ici:1
; CHECK-NEXT:   ret double %x: icv:1 ici:1
