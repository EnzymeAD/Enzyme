; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @callee(float** %slot, float %val) {
entry:
  %cur = load float*, float** %slot, align 8
  store float %val, float* %cur, align 4
  %next = getelementptr inbounds float, float* %cur, i64 1
  store float* %next, float** %slot, align 8
  ret void
}

; CHECK: callee - {} |{[-1]:Pointer, [-1,-1]:Pointer}:{} {[-1]:Float@float}:{}
; CHECK-NEXT: float** %slot: {[-1]:Pointer, [-1,-1]:Pointer, [-1,0,0]:Float@float}
; CHECK-NEXT: float %val: {[-1]:Float@float}
; CHECK-NEXT: entry
; CHECK-NEXT:   %cur = load float*, float** %slot, align 8: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   store float %val, float* %cur, align 4: {}
; CHECK-NEXT:   %next = getelementptr inbounds float, float* %cur, i64 1: {[-1]:Pointer}
; CHECK-NEXT:   store float* %next, float** %slot, align 8: {}
; CHECK-NEXT:   ret void: {}
