; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: fwd=11.000000 rev=6.000000,5.000000

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare double @__enzyme_fwddiff(ptr, ...)
declare double @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr nocapture readonly, ...)

@.str = private unnamed_addr constant [18 x i8] c"fwd=%f rev=%f,%f\0A\00", align 1

define internal void @copy_short_mask(ptr nocapture writeonly %dst_rec, ptr nocapture readonly %src_mask) noinline {
entry:
  %mask_slot = getelementptr inbounds i8, ptr %dst_rec, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %mask_slot, ptr align 4 %src_mask, i64 20, i1 false), !tbaa.struct !5
  ret void
}

define internal void @copy_variant_record(ptr nocapture writeonly %dst_var, ptr nocapture readonly %src_var) noinline {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst_var, ptr align 8 %src_var, i64 16, i1 false), !tbaa.struct !9
  ret void
}

; CHECK-LABEL: define internal double @fwddiffecompute_view_and_variant(
; CHECK: call void @fwddiffecopy_short_mask(
; CHECK: call void @fwddiffecopy_variant_record(

; CHECK-LABEL: define internal void @fwddiffecopy_short_mask(
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %"mask_slot'ipg", ptr align 4 %src_mask, i64 20, i1 false)

; CHECK-LABEL: define internal void @diffecompute_view_and_variant(
; CHECK: call void @diffecopy_variant_record(
; CHECK: call void @diffecopy_short_mask(

; CHECK-LABEL: define internal void @diffecopy_short_mask(
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertentry
; CHECK: invertentry:
; CHECK-NEXT:   ret void
define double @compute_view_and_variant(ptr nocapture readonly %x) {
entry:
  %mask = alloca [20 x i8], align 4
  %rec = alloca [32 x i8], align 8
  %var_src = alloca [16 x i8], align 8
  %var_dst = alloca [16 x i8], align 8
  call void @llvm.memset.p0.i64(ptr align 4 %mask, i8 -1, i64 20, i1 false)
  store ptr %x, ptr %rec, align 8
  call void @copy_short_mask(ptr %rec, ptr %mask)

  %x1_ptr = getelementptr inbounds double, ptr %x, i64 1
  %x1 = load double, ptr %x1_ptr, align 8
  store double %x1, ptr %var_src, align 8
  %tag_ptr = getelementptr inbounds i8, ptr %var_src, i64 8
  store i16 1, ptr %tag_ptr, align 8, !tbaa !3
  call void @copy_variant_record(ptr %var_dst, ptr %var_src)

  %data_ptr = load ptr, ptr %rec, align 8
  %x0 = load double, ptr %data_ptr, align 8
  %v1 = load double, ptr %var_dst, align 8
  %sq0 = fmul double %x0, %x0
  %prod1 = fmul double %v1, 5.000000e+00
  %sum = fadd double %sq0, %prod1
  ret double %sum
}

define i32 @main() {
entry:
  %x = alloca [2 x double], align 16
  %dx_fwd = alloca [2 x double], align 16
  %dx_rev = alloca [2 x double], align 16
  store double 3.000000e+00, ptr %x, align 16
  %x1 = getelementptr inbounds double, ptr %x, i64 1
  store double 4.000000e+00, ptr %x1, align 8

  store double 1.000000e+00, ptr %dx_fwd, align 16
  %dx_fwd1 = getelementptr inbounds double, ptr %dx_fwd, i64 1
  store double 1.000000e+00, ptr %dx_fwd1, align 8

  store double 0.000000e+00, ptr %dx_rev, align 16
  %dx_rev1 = getelementptr inbounds double, ptr %dx_rev, i64 1
  store double 0.000000e+00, ptr %dx_rev1, align 8

  %fwd = call double (ptr, ...) @__enzyme_fwddiff(ptr @compute_view_and_variant, metadata !"enzyme_dup", ptr %x, ptr %dx_fwd)
  %rev = call double (ptr, ...) @__enzyme_autodiff(ptr @compute_view_and_variant, metadata !"enzyme_dup", ptr %x, ptr %dx_rev)
  %g0 = load double, ptr %dx_rev, align 16
  %g1 = load double, ptr %dx_rev1, align 8
  %printed = call i32 (ptr, ...) @printf(ptr @.str, double %fwd, double %g0, double %g1)
  ret i32 0
}

!0 = !{!"Simple C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!"short", !1, i64 0}
!3 = !{!2, !2, i64 0}
!5 = !{i64 0, i64 2, !3, i64 2, i64 2, !3, i64 4, i64 2, !3, i64 6, i64 2, !3, i64 8, i64 2, !3, i64 10, i64 2, !3, i64 12, i64 2, !3, i64 14, i64 2, !3, i64 16, i64 2, !3, i64 18, i64 2, !3}
!6 = !{!"double", !1, i64 0}
!7 = !{!6, !6, i64 0}
!8 = !{!"int", !1, i64 0}
!10 = !{!8, !8, i64 0}
!9 = !{i64 0, i64 8, !7, i64 0, i64 4, !10, i64 8, i64 2, !3}
