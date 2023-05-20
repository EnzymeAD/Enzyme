; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=a0 -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=a0 -S -o /dev/null | FileCheck %s

source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

define fastcc void @a0() {
bb:
  %i = insertvalue { { i1, i1, i1 }, i8 } undef, i1 false, 0, 0
  %i2 = insertvalue { { i1, i1, i1 }, i8 } %i, i8 3, 1
  ret void
}

; CHECK: a0 - {} |
; CHECK-NEXT: bb
; CHECK-NEXT:   %i = insertvalue { { i1, i1, i1 }, i8 } {{(undef|poison)}}, i1 false, 0, 0: {[-1]:Anything}
; CHECK-NEXT:   %i2 = insertvalue { { i1, i1, i1 }, i8 } %i, i8 3, 1: {[0]:Anything, [1]:Anything, [2]:Anything, [3]:Integer}
; CHECK-NEXT:   ret void: {}
