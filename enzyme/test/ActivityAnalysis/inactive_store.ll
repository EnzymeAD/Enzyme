; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s

define double @f(double %x) {
entry:
  %a = alloca i8, align 1
  %val = call i8 @get_inactive_val(), !enzyme_inactive !0
  store i8 %val, i8* %a, align 1
  ret double %x
}

declare i8 @get_inactive_val()

!0 = !{}

; CHECK: double %x: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %a = alloca i8, align 1: icv:1 ici:1
; CHECK-NEXT:   %val = call i8 @get_inactive_val(), !enzyme_inactive !0: icv:1 ici:1
; CHECK-NEXT:   store i8 %val, {{i8\*|ptr}} %a, align 1: icv:1 ici:1
; CHECK-NEXT:   ret double %x: icv:1 ici:1
