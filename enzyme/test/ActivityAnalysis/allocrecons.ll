; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=f -activity-analysis-inactive-args -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f -activity-analysis-inactive-args -S | FileCheck %s

; ModuleID = 'start'
source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

declare fastcc void @sub({} addrspace(10)** nocapture readonly %a0) 

define void @f([1 x {} addrspace(10)*] %in) {
entry:
  %al = alloca {} addrspace(10)*, align 8
  %.fca = extractvalue [1 x {} addrspace(10)*] %in, 0
  store {} addrspace(10)* %.fca, {} addrspace(10)** %al, align 8
  call fastcc void @sub({} addrspace(10)** nocapture readonly %al)
  ret void
}

; CHECK: [1 x {} addrspace(10)*] %in: icv:1
; CHECK-NEXT: entry
; CHECK-NEXT:   %al = alloca {} addrspace(10)*, align 8: icv:1 ici:1
; CHECK-NEXT:   %.fca = extractvalue [1 x {} addrspace(10)*] %in, 0: icv:1 ici:1
; CHECK-NEXT:   store {} addrspace(10)* %.fca, {} addrspace(10)** %al, align 8: icv:1 ici:1
; CHECK-NEXT:   call fastcc void @sub({} addrspace(10)** nocapture readonly %al): icv:1 ici:1
; CHECK-NEXT:   ret void: icv:1 ici:1
