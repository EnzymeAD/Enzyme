; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s

define double @f(double %x) {
entry:
  %a = alloca i8, align 1
  store i8 17, i8* %a, align 1
  %c = call double @sub(double %x, i8* %a)
  ret double %c
}

declare double @sub(double, i8*) 

; CHECK: double %x: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %a = alloca i8, align 1: icv:1 ici:1
; CHECK-NEXT:   store i8 17, i8* %a, align 1: icv:1 ici:1
; CHECK-NEXT:   %c = call double @sub(double %x, i8* %a): icv:0 ici:0
; CHECK-NEXT:   ret double %c: icv:1 ici:1
