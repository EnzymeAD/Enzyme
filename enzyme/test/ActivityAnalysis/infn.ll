; RUN: %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=kernel_main -activity-analysis-inactive-args -o /dev/null | FileCheck %s

declare double* @mydata(double*)

declare double @loadstore(double*)

define double @kernel_main(double* %in) {
entry:
  %tmp11 = call double* @mydata(double* %in)
  %tmp12 = call double @loadstore(double* %tmp11)
  ret double %tmp12
}

; CHECK: double* %in: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   %tmp11 = call double* @mydata(double* %in): icv:1 ici:1
; CHECK-NEXT:   %tmp12 = call double @loadstore(double* %tmp11): icv:1 ici:1
; CHECK-NEXT:   ret double %tmp12: icv:1 ici:1
