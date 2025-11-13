; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller() {
entry:
  %a.dbg.spill = alloca i128, align 16
  store i128 -12, i128* %a.dbg.spill, align 16
  %a.dbg.spill2 = alloca i128, align 16
  store i128 -12078052328127107563834081753142289173, i128* %a.dbg.spill2, align 16
  ret void
}


; CHECK: caller - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:  %a.dbg.spill = alloca i128, align 16: {[-1]:Pointer, [-1,-1]:Integer}
; CHECK-NEXT:  store i128 -12, i128* %a.dbg.spill, align 16: {}
; CHECK-NEXT:  %a.dbg.spill2 = alloca i128, align 16: {[-1]:Pointer}
; CHECK-NEXT:  store i128 -12078052328127107563834081753142289173, i128* %a.dbg.spill2, align 16: {}
; CHECK-NEXT:  ret void: {}
