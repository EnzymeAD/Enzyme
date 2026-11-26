; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @callee(<2 x double> %v, ptr %p, i64 %k) {
entry:
  %i = load i64, ptr %p, align 8
  %b = bitcast <2 x double> %v to <2 x i64>
  %e = extractelement <2 x i64> %b, i64 %i
  %w = insertelement <2 x i64> %b, i64 %k, i64 1
  %m = extractelement <2 x i64> %w, i64 %i
  ret void
}

; CHECK: callee - {} |{[-1]:Float@double}:{} {[-1]:Pointer}:{} {[-1]:Integer}:{}
; CHECK-NEXT: <2 x double> %v: {[-1]:Float@double}
; CHECK-NEXT: ptr %p: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: i64 %k: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %i = load i64, ptr %p, align 8: {[-1]:Integer}
; CHECK-NEXT:   %b = bitcast <2 x double> %v to <2 x i64>: {[-1]:Float@double}
; CHECK-NEXT:   %e = extractelement <2 x i64> %b, i64 %i: {[-1]:Float@double}
; CHECK-NEXT:   %w = insertelement <2 x i64> %b, i64 %k, i64 1: {[0]:Float@double, [1]:Float@double, [2]:Float@double, [3]:Float@double, [4]:Float@double, [5]:Float@double, [6]:Float@double, [7]:Float@double, [8]:Integer, [9]:Integer, [10]:Integer, [11]:Integer, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer}
; CHECK-NEXT:   %m = extractelement <2 x i64> %w, i64 %i: {}
; CHECK-NEXT:   ret void: {}
