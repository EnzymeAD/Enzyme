; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -S -enzyme-loose-types | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify)" -S -enzyme-loose-types | FileCheck %s
; UNSUPPORTED: llvm16
;
; Test that reverse-mode AD with -enzyme-loose-types handles @llvm.umax.i32
; intrinsic calls on integer values without crashing in invertPointerM.
;
; The pattern: a struct passed via enzyme_dup contains both float* and i32*
; fields. The function does active float computations, derives an i32 flag
; from a float comparison, then uses load -> @llvm.umax.i32 -> store on the
; i32* field (SerialOperations::AtomicMax pattern).
;
; With loose-types, visitCommonStore calls invertPointerM on the i32 result of
; @llvm.umax.i32. Without the fix, this hits:
;   assert(0 && "cannot find deal with ptr that isnt arg")
;
; The fix returns a zero shadow for integer values under loose-types analysis.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @llvm.umax.i32(i32, i32)

define void @compute_alpha(ptr noalias %op, i32 %face_idx, ptr noalias %nbr) {
entry:
  %out_gep = getelementptr i8, ptr %op, i64 0
  %out_ptr = load ptr, ptr %out_gep, align 8
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
  %new_flag = tail call i32 @llvm.umax.i32(i32 %old_flag, i32 %flag)
  store i32 %new_flag, ptr %flags_elem, align 4
  ret void
}

define void @caller(ptr %op, ptr %d_op, i32 %face_idx, ptr %nbr) {
entry:
  call void (...) @__enzyme_autodiff(
    ptr @compute_alpha,
    metadata !"enzyme_dup", ptr %op, ptr %d_op,
    metadata !"enzyme_const", i32 %face_idx,
    metadata !"enzyme_const", ptr %nbr
  )
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffecompute_alpha(ptr noalias
; CHECK-SAME: %op, ptr
; CHECK-SAME: %"op'"
; CHECK-SAME: i32 %face_idx, ptr noalias
; CHECK-SAME: %nbr)

; Verify the primal block correctly computes and stores i32 umax with zero shadow
; CHECK:        %new_flag = tail call i32 @llvm.umax.i32(i32 %old_flag, i32 %flag)
; CHECK-NEXT:   store i32 0, ptr %"flags_elem'ipg"
; CHECK-NEXT:   store i32 %new_flag, ptr %flags_elem

; Verify the reverse block correctly propagates float derivatives
; CHECK:      invertentry:
; CHECK:        %[[m0:.+]] = fmul fast float %[[dres:.+]], %clip_val
; CHECK:        %[[m1:.+]] = fmul fast float %[[dres]], %face_val
; CHECK:        ret void
