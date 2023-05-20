; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -S | FileCheck %s

declare dso_local double @RatelLog1pSeries(double %x) "enzyme_math"="log1p"

define void @f(double %a6) {
entry:
  %a = call fast double @RatelLog1pSeries(double %a6)
  ret void
}

; CHECK: double %a6: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %a = call fast double @RatelLog1pSeries(double %a6): icv:1 ici:1
; CHECK-NEXT:   ret void: icv:1 ici:1
