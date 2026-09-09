; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -enzyme-julia-addr-load -type-analysis-func=caller -S -o /dev/null | FileCheck %s

; With -enzyme-julia-addr-load, every value whose type contains Julia tracked
; pointers (address spaces 10, 11, 13) is augmented: a type made only of tracked
; pointers is a pointer everywhere, and a mixed aggregate gets a Pointer at each
; tracked pointer's byte offset.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:10:11:12:13"

%all = type { ptr addrspace(10), [2 x ptr addrspace(10)] }
%mixed = type { i64, ptr addrspace(10), { double, [2 x ptr addrspace(11)] }, ptr addrspace(13) }
%none = type { i64, ptr, [2 x double] }

define void @caller(%all %a, %mixed %m, %none %n, ptr addrspace(10) %p, [3 x ptr addrspace(13)] %arr, i64 %i) {
entry:
  %a1 = insertvalue %all %a, ptr addrspace(10) %p, 0
  %m1 = insertvalue %mixed %m, i64 %i, 0
  %n1 = insertvalue %none %n, i64 %i, 0
  %e = extractvalue [3 x ptr addrspace(13)] %arr, 1
  ret void
}

; CHECK: %all %a: {[-1]:Pointer}
; CHECK-NEXT: %mixed %m: {[8]:Pointer, [24]:Pointer, [32]:Pointer, [40]:Pointer}
; CHECK-NEXT: %none %n: {}
; CHECK-NEXT: ptr addrspace(10) %p: {[-1]:Pointer}
; CHECK-NEXT: [3 x ptr addrspace(13)] %arr: {[-1]:Pointer}
; CHECK-NEXT: i64 %i: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %a1 = insertvalue %all %a, ptr addrspace(10) %p, 0: {[-1]:Pointer}
; CHECK-NEXT:   %m1 = insertvalue %mixed %m, i64 %i, 0: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Integer, [5]:Integer, [6]:Integer, [7]:Integer, [8]:Pointer, [24]:Pointer, [32]:Pointer, [40]:Pointer}
; CHECK-NEXT:   %n1 = insertvalue %none %n, i64 %i, 0: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Integer, [5]:Integer, [6]:Integer, [7]:Integer}
; CHECK-NEXT:   %e = extractvalue [3 x ptr addrspace(13)] %arr, 1: {[-1]:Pointer}
; CHECK-NEXT:   ret void: {}
