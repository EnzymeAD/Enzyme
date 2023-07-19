; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=kernel_main -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=kernel_main -S | FileCheck %s

declare i8* @malloc(i64)

declare i8* @tuple(i8*)

declare float @use(i8*, float)

define float @kernel_main(float %tmp1) {
entry:
  %const = call i8* @malloc(i64 4) "enzyme_inactive"
  %tup = call i8* @tuple(i8* %const) "enzyme_immutable"
  %res = call float @use(i8* %tup, float %tmp1)
  ret float %res
}

; CHECK: entry
; CHECK-NEXT:   %const = call i8* @malloc(i64 4) #0: icv:1 ici:1
; CHECK-NEXT:   %tup = call i8* @tuple(i8* %const) #1: icv:1 ici:1
; CHECK-NEXT:   %res = call float @use(i8* %tup, float %tmp1): icv:0 ici:0
; CHECK-NEXT:   ret float %res: icv:1 ici:1
