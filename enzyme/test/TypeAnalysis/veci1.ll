; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=f -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=f -S -o /dev/null | FileCheck %s

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i1 @f(<2 x i1> %inp) {
entry:
  %e0 = extractelement <2 x i1> %inp, i32 0
  %e1 = extractelement <2 x i1> %inp, i32 1
  %res = and i1 %e0, %e1
  ret i1 %res
}

; CHECK: f - {[-1]:Integer} |{[-1]:Integer}:{} 
; CHECK-NEXT: <2 x i1> %inp: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %e0 = extractelement <2 x i1> %inp, i32 0: {[-1]:Integer}
; CHECK-NEXT:   %e1 = extractelement <2 x i1> %inp, i32 1: {[-1]:Integer}
; CHECK-NEXT:   %res = and i1 %e0, %e1: {[-1]:Integer}
; CHECK-NEXT:   ret i1 %res: {}
