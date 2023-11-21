; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -activity-analysis-inactive-args -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -activity-analysis-inactive-args -S | FileCheck %s

define void @f(i8* %cv, i8** %cptr) {
entry:
  %v2 = call i8* @mycopy(i8* %cv)
  store i8* %v2, i8** %cptr
  ret void
}

declare i8* @mycopy(i8*)

; CHECK: i8* %cv: icv:1
; CHECK: i8** %cptr: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   %v2 = call i8* @mycopy(i8* %cv): icv:1 ici:1
; CHECK-NEXT:   store i8* %v2, i8** %cptr, align 8: icv:1 ici:1
; CHECK-NEXT:   ret void: icv:1 ici:1