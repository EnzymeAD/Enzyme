; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

; An `and` with a mask wider than 64 bits (here the UUIDv4 version/variant
; mask on an i128) must not be treated as a small negative mask.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f(i128 %x)

define void @caller(i128 %n) {
entry:
  %m = and i128 %n, -1133381790946770133450753
  %v = or i128 %m, 302240678275694148452352
  call void @f(i128 %v)
  ret void
}

; CHECK: caller - {} |{[-1]:Integer}:{}
; CHECK-NEXT: i128 %n: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %m = and i128 %n, -1133381790946770133450753: {[-1]:Anything}
; CHECK-NEXT:   %v = or i128 %m, 302240678275694148452352: {[-1]:Anything}
; CHECK-NEXT:   call void @f(i128 %v): {}
; CHECK-NEXT:   ret void: {}
