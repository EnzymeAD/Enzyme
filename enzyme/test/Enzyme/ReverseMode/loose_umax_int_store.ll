; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-loose-types -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-loose-types -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.abs.i32(i32, i1 immarg)
declare void @__enzyme_autodiff(...)

define void @compute_alpha(ptr noalias %op, i32 %face_idx, ptr noalias %nbr) {
entry:
  %out_ptr = load ptr, ptr %op, align 8
  %flags_gep = getelementptr i8, ptr %op, i64 8
  %flags_ptr = load ptr, ptr %flags_gep, align 8
  %clip_gep = getelementptr i8, ptr %op, i64 16
  %clip_val = load float, ptr %clip_gep, align 4
  %idx = zext i32 %face_idx to i64
  %face_gep = getelementptr float, ptr %out_ptr, i64 %idx
  %face_val = load float, ptr %face_gep, align 4
  %dot = fmul float %face_val, %clip_val
  %cmp = fcmp ole float %dot, 0.0
  br i1 %cmp, label %left_handed, label %normal

left_handed:
  br label %merge

normal:
  br label %merge

merge:
  %result = phi float [ %clip_val, %left_handed ], [ %dot, %normal ]
  %flag = phi i32 [ 1, %left_handed ], [ 0, %normal ]
  store float %result, ptr %face_gep, align 4
  %n0 = load i32, ptr %nbr, align 4
  %n0_ext = zext i32 %n0 to i64
  %flags_elem = getelementptr i32, ptr %flags_ptr, i64 %n0_ext
  %old_flag = load i32, ptr %flags_elem, align 4
  %abs_flag = call i32 @llvm.abs.i32(i32 %old_flag, i1 false)
  %new_flag = tail call i32 @llvm.umax.i32(i32 %abs_flag, i32 %flag)
  store i32 %new_flag, ptr %flags_elem, align 4
  ret void
}

define void @caller(ptr %op, ptr %d_op, i32 %face_idx, ptr %nbr) {
entry:
  call void (...) @__enzyme_autodiff(ptr @compute_alpha, metadata !"enzyme_dup", ptr %op, ptr %d_op, metadata !"enzyme_const", i32 %face_idx, metadata !"enzyme_const", ptr %nbr)
  ret void
}

; CHECK-LABEL: define internal void @diffecompute_alpha(
; CHECK: %"old_flag'ipl" = load i32, ptr %"flags_elem'ipg"
; CHECK-NEXT: %old_flag = load i32, ptr %flags_elem
; CHECK-NEXT: %"abs_flag'ipi" = call i32 @llvm.abs.i32(i32 %"old_flag'ipl", i1 false)
; CHECK-NEXT: %abs_flag = call i32 @llvm.abs.i32(i32 %old_flag, i1 false)
; CHECK-NEXT: %"new_flag'ipi" = call i32 @llvm.umax.i32(i32 %"abs_flag'ipi", i32 %flag)
; CHECK-NEXT: %new_flag = tail call i32 @llvm.umax.i32(i32 %abs_flag, i32 %flag)
; CHECK-NEXT: store i32 %"new_flag'ipi", ptr %"flags_elem'ipg"
; CHECK-NEXT: store i32 %new_flag, ptr %flags_elem
; CHECK: invertentry:
; CHECK: %[[DDOT:.+]] = load float, ptr %"dot'de"
; CHECK: fmul fast float %[[DDOT]], %clip_val
; CHECK: fmul fast float %[[DDOT]], %face_val
