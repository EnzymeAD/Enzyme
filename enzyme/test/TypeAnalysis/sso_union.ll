; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=move_ctor -S -o /dev/null | FileCheck %s --check-prefix=MOVE

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @use_cstr(ptr)
declare void @free(ptr)

define i8 @caller(ptr %heap, i1 %is_long) {
entry:
  %src = alloca [24 x i8], align 8
  %dst = alloca [24 x i8], align 8
  store ptr %heap, ptr %src, align 8
  store i8 0, ptr %src, align 8
  store i64 5, ptr %dst, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 24, i1 false)
  %lp = load ptr, ptr %src, align 8
  %data = select i1 %is_long, ptr %lp, ptr %src
  %ch = load i8, ptr %data, align 1
  ret i8 %ch
}

define i8 @move_ctor(ptr %src, i1 %is_long) {
entry:
  %dst = alloca [24 x i8], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 24, i1 false)
  store i8 0, ptr %src, align 8
  %size_slot = getelementptr inbounds i8, ptr %src, i64 23
  store i8 0, ptr %size_slot, align 1
  %lp = load ptr, ptr %dst, align 8
  %data = select i1 %is_long, ptr %lp, ptr %dst
  %ch = load i8, ptr %data, align 1
  call void @free(ptr %lp)
  ret i8 %ch
}

; CHECK: caller - {[-1]:Integer} |
; CHECK: entry
; CHECK:   %lp = load ptr, ptr %src, align 8: {[-1]:Pointer
; CHECK-NEXT:   %data = select i1 %is_long, ptr %lp, ptr %src: {[-1]:Pointer, [-1,0]:Integer}
; CHECK-NEXT:   %ch = load i8, ptr %data, align 1: {[-1]:Integer}

; MOVE: move_ctor - {[-1]:Integer} |
; MOVE: entry
; MOVE:   %lp = load ptr, ptr %dst, align 8: {[-1]:Pointer
; MOVE-NEXT:   %data = select i1 %is_long, ptr %lp, ptr %dst: {[-1]:Pointer, [-1,0]:Integer}
; MOVE-NEXT:   %ch = load i8, ptr %data, align 1: {[-1]:Integer}
