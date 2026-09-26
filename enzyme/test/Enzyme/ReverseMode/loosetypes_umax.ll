; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -S -enzyme-loose-types | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify)" -S -enzyme-loose-types | FileCheck %s; fi
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
; The fix handles llvm.umax as a binop-like intrinsic during shadow
; reconstruction, instead of falling through to the generic no-shadow path.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @llvm.umax.i32(i32, i32)

define void @compute_alpha(i8* noalias %op, i32 %face_idx, i32* noalias %nbr) {
entry:
  %out_gep = getelementptr i8, i8* %op, i64 0
  %out_ptr_gep = bitcast i8* %out_gep to float**
  %out_ptr = load float*, float** %out_ptr_gep, align 8
  %flags_gep = getelementptr i8, i8* %op, i64 8
  %flags_ptr_gep = bitcast i8* %flags_gep to i32**
  %flags_ptr = load i32*, i32** %flags_ptr_gep, align 8
  %clip_gep = getelementptr i8, i8* %op, i64 16
  %clip_ptr = bitcast i8* %clip_gep to float*
  %clip_val = load float, float* %clip_ptr, align 4
  %idx = zext i32 %face_idx to i64
  %face_gep = getelementptr float, float* %out_ptr, i64 %idx
  %face_val = load float, float* %face_gep, align 4
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
  store float %result, float* %face_gep, align 4
  %n0 = load i32, i32* %nbr, align 4
  %n0_ext = zext i32 %n0 to i64
  %flags_elem = getelementptr i32, i32* %flags_ptr, i64 %n0_ext
  %old_flag = load i32, i32* %flags_elem, align 4
  %new_flag = tail call i32 @llvm.umax.i32(i32 %old_flag, i32 %flag)
  store i32 %new_flag, i32* %flags_elem, align 4
  ret void
}

define void @caller(i8* %op, i8* %d_op, i32 %face_idx, i32* %nbr) {
entry:
  call void (...) @__enzyme_autodiff(
    i8* bitcast (void (i8*, i32, i32*)* @compute_alpha to i8*),
    metadata !"enzyme_dup", i8* %op, i8* %d_op,
    metadata !"enzyme_const", i32 %face_idx,
    metadata !"enzyme_const", i32* %nbr
  )
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffecompute_alpha(
; CHECK-SAME: %op,
; CHECK-SAME: %"op'",
; CHECK-SAME: i32 %face_idx,
; CHECK-SAME: %nbr)

; Verify the primal block no longer uses the old zero-shadow fallback.
; CHECK:        %new_flag = {{(tail )?}}call i32 @llvm.umax.i32(i32 %old_flag, i32 %flag)
; CHECK-NOT:    store i32 0, {{(ptr|i32\*)}} %"flags_elem'ipg"
; CHECK:        store i32 {{.+}}, {{(ptr|i32\*)}} %"flags_elem'ipg"
; CHECK:        store i32 %new_flag, {{(ptr|i32\*)}} %flags_elem

; Verify the reverse block correctly propagates float derivatives
; CHECK:      invertentry:
; CHECK:        %[[m0:.+]] = fmul fast float %[[dres:.+]], %clip_val
; CHECK:        %[[m1:.+]] = fmul fast float %[[dres]], %face_val
; CHECK:        ret void
