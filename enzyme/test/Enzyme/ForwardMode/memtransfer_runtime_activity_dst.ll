; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,default<O2>" -S | %lli -

@const_doubles = internal constant [2 x double] [double 7.000000e+00, double 8.000000e+00], align 8

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @__enzyme_fwddiff(...)
declare void @__enzyme_fwdsplit(...)
declare void @__enzyme_autodiff(...)
declare void @abort()

; CHECK-LABEL: define internal void @fwddiffeshift_left_doubles(
; CHECK:   %[[dst_inact1:.+]] = icmp eq ptr %"buf'", %buf
; CHECK:   %[[src_inact1:.+]] = icmp eq ptr %{{.+}}, %{{.+}}
; CHECK:   %[[skip1:.+]] = or i1 %[[dst_inact1]], %[[src_inact1]]
; CHECK:   %[[cplen1:.+]] = select i1 %[[skip1]], i64 0, i64 16
; CHECK:   %[[szlen1:.+]] = select i1 %[[src_inact1]], i64 16, i64 0
; CHECK:   %[[zlen1:.+]] = select i1 %[[dst_inact1]], i64 0, i64 %[[szlen1]]
; CHECK:   call void @llvm.memmove.p0.p0.i64(ptr align 8 %"buf'", ptr align 8 %{{.+}}, i64 %[[cplen1]], i1 false)
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %"buf'", i8 0, i64 %[[zlen1]], i1 false)
define void @shift_left_doubles(ptr nocapture %buf) {
entry:
  %dummy = load double, ptr %buf, align 8, !enzyme_type !0
  %src = getelementptr inbounds double, ptr %buf, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %buf, ptr align 8 %src, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffecopy_const_doubles(
; CHECK:   %[[dst_inact2:.+]] = icmp eq ptr %"dst'", %dst
; CHECK:   %[[zlen2:.+]] = select i1 %[[dst_inact2]], i64 0, i64 16
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %"dst'", i8 0, i64 %[[zlen2]], i1 false)
define void @copy_const_doubles(ptr nocapture %dst) {
entry:
  %dummy = load double, ptr %dst, align 8, !enzyme_type !0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 @const_doubles, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffecopy_ptrs(
; CHECK:   %[[dst_inact3:.+]] = icmp eq ptr %"dst'", %dst
; CHECK:   %[[cplen3:.+]] = select i1 %[[dst_inact3]], i64 0, i64 16
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %"dst'", ptr align 8 %"src'", i64 %[[cplen3]], i1 false)
define void @copy_ptrs(ptr nocapture %dst, ptr nocapture readonly %src) {
entry:
  %dummy_dst = load ptr, ptr %dst, align 8, !enzyme_type !2
  %dummy_src = load ptr, ptr %src, align 8, !enzyme_type !2
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffesplit_copy_doubles(
; CHECK:   %[[dst_inact4:.+]] = icmp eq ptr %"dst'", %dst
; CHECK:   %[[src_inact4:.+]] = icmp eq ptr %"src'", %src
; CHECK:   %[[skip4:.+]] = or i1 %[[dst_inact4]], %[[src_inact4]]
; CHECK:   %[[cplen4:.+]] = select i1 %[[skip4]], i64 0, i64 16
; CHECK:   %[[szlen4:.+]] = select i1 %[[src_inact4]], i64 16, i64 0
; CHECK:   %[[zlen4:.+]] = select i1 %[[dst_inact4]], i64 0, i64 %[[szlen4]]
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %"dst'", ptr align 8 %"src'", i64 %[[cplen4]], i1 false)
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %"dst'", i8 0, i64 %[[zlen4]], i1 false)
define void @split_copy_doubles(ptr nocapture %dst, ptr nocapture readonly %src) {
entry:
  %dummy_dst = load double, ptr %dst, align 8, !enzyme_type !0
  %dummy_src = load double, ptr %src, align 8, !enzyme_type !0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 16, i1 false)
  ret void
}

; CHECK-LABEL: define internal void @differev_copy_const_doubles(
; CHECK:   %[[dst_inact5:.+]] = icmp eq ptr %"dst'", %dst
; CHECK:   %[[zlen5:.+]] = select i1 %[[dst_inact5]], i64 0, i64 16
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %"dst'", i8 0, i64 %[[zlen5]], i1 false)
define void @rev_copy_const_doubles(ptr nocapture %dst) {
entry:
  %dummy = load double, ptr %dst, align 8, !enzyme_type !0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 @const_doubles, i64 16, i1 false)
  ret void
}

define void @check_eq(double %actual, double %expected) {
entry:
  %ok = fcmp oeq double %actual, %expected
  br i1 %ok, label %done, label %fail

fail:
  call void @abort()
  unreachable

done:
  ret void
}

define void @check_ptr_eq(ptr %actual, ptr %expected) {
entry:
  %ok = icmp eq ptr %actual, %expected
  br i1 %ok, label %done, label %fail

fail:
  call void @abort()
  unreachable

done:
  ret void
}

define i32 @main() {
entry:
  ; 1. ForwardMode in-place overlapping memmove with runtime-inactive buffer.
  %buf = alloca [3 x double], align 8
  %buf1 = getelementptr inbounds double, ptr %buf, i64 1
  %buf2 = getelementptr inbounds double, ptr %buf, i64 2
  store double 1.000000e+01, ptr %buf, align 8
  store double 2.000000e+01, ptr %buf1, align 8
  store double 3.000000e+01, ptr %buf2, align 8
  call void (...) @__enzyme_fwddiff(ptr @shift_left_doubles, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %buf, ptr %buf)
  %s0 = load double, ptr %buf, align 8
  %s1 = load double, ptr %buf1, align 8
  call void @check_eq(double %s0, double 2.000000e+01)
  call void @check_eq(double %s1, double 3.000000e+01)

  ; 2. ForwardMode memcpy from constant source with enzyme_dupnoneed and runtime-inactive dst.
  %cdst = alloca [2 x double], align 8
  %cdst1 = getelementptr inbounds double, ptr %cdst, i64 1
  store double 5.000000e+00, ptr %cdst, align 8
  store double 6.000000e+00, ptr %cdst1, align 8
  call void (...) @__enzyme_fwddiff(ptr @copy_const_doubles, metadata !"enzyme_runtime_activity", metadata !"enzyme_dupnoneed", ptr %cdst, ptr %cdst)
  %c0 = load double, ptr %cdst, align 8
  %c1 = load double, ptr %cdst1, align 8
  call void @check_eq(double %c0, double 5.000000e+00)
  call void @check_eq(double %c1, double 6.000000e+00)

  ; Active dst must still be zeroed when copying from constant source.
  %dcdst = alloca [2 x double], align 8
  %dcdst1 = getelementptr inbounds double, ptr %dcdst, i64 1
  store double 9.000000e+00, ptr %dcdst, align 8
  store double 9.000000e+00, ptr %dcdst1, align 8
  call void (...) @__enzyme_fwddiff(ptr @copy_const_doubles, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %cdst, ptr %dcdst)
  %dc0 = load double, ptr %dcdst, align 8
  %dc1 = load double, ptr %dcdst1, align 8
  call void @check_eq(double %dc0, double 0.000000e+00)
  call void @check_eq(double %dc1, double 0.000000e+00)

  ; 3. ForwardMode pointer memcpy with enzyme_dupnoneed and runtime-inactive dst.
  %x = alloca double, align 8
  %dx = alloca double, align 8
  %p_dst = alloca [2 x ptr], align 8
  %p_src = alloca [2 x ptr], align 8
  %dp_src = alloca [2 x ptr], align 8
  %p_dst1 = getelementptr inbounds ptr, ptr %p_dst, i64 1
  %p_src1 = getelementptr inbounds ptr, ptr %p_src, i64 1
  %dp_src1 = getelementptr inbounds ptr, ptr %dp_src, i64 1
  store ptr %x, ptr %p_dst, align 8
  store ptr %x, ptr %p_dst1, align 8
  store ptr %x, ptr %p_src, align 8
  store ptr %x, ptr %p_src1, align 8
  store ptr %dx, ptr %dp_src, align 8
  store ptr %dx, ptr %dp_src1, align 8
  call void (...) @__enzyme_fwddiff(ptr @copy_ptrs, metadata !"enzyme_runtime_activity", metadata !"enzyme_dupnoneed", ptr %p_dst, ptr %p_dst, metadata !"enzyme_dup", ptr %p_src, ptr %dp_src)
  %pd0 = load ptr, ptr %p_dst, align 8
  call void @check_ptr_eq(ptr %pd0, ptr %x)

  ; 4. ForwardModeSplit float memcpy with runtime-inactive dst or runtime-inactive src.
  %f_dst = alloca [2 x double], align 8
  %df_dst = alloca [2 x double], align 8
  %f_src = alloca [2 x double], align 8
  %df_src = alloca [2 x double], align 8
  %f_dst1 = getelementptr inbounds double, ptr %f_dst, i64 1
  %df_dst1 = getelementptr inbounds double, ptr %df_dst, i64 1
  %f_src1 = getelementptr inbounds double, ptr %f_src, i64 1
  %df_src1 = getelementptr inbounds double, ptr %df_src, i64 1
  store double 1.100000e+01, ptr %f_dst, align 8
  store double 2.200000e+01, ptr %f_dst1, align 8
  store double 1.100000e+01, ptr %f_src, align 8
  store double 2.200000e+01, ptr %f_src1, align 8
  store double 3.000000e+00, ptr %df_src, align 8
  store double 4.000000e+00, ptr %df_src1, align 8
  call void (...) @__enzyme_fwdsplit(ptr @split_copy_doubles, metadata !"enzyme_nofree", metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %f_dst, ptr %f_dst, metadata !"enzyme_dup", ptr %f_src, ptr %df_src, ptr null)
  %fd0 = load double, ptr %f_dst, align 8
  %fd1 = load double, ptr %f_dst1, align 8
  call void @check_eq(double %fd0, double 1.100000e+01)
  call void @check_eq(double %fd1, double 2.200000e+01)

  store double 9.900000e+01, ptr %df_dst, align 8
  store double 8.800000e+01, ptr %df_dst1, align 8
  call void (...) @__enzyme_fwdsplit(ptr @split_copy_doubles, metadata !"enzyme_nofree", metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %f_dst, ptr %df_dst, metadata !"enzyme_dup", ptr %f_src, ptr %f_src, ptr null)
  %dfd0 = load double, ptr %df_dst, align 8
  %dfd1 = load double, ptr %df_dst1, align 8
  call void @check_eq(double %dfd0, double 0.000000e+00)
  call void @check_eq(double %dfd1, double 0.000000e+00)

  ; 5. ReverseMode memcpy from constant source with runtime-inactive and active dst.
  %r_dst = alloca [2 x double], align 8
  %dr_dst = alloca [2 x double], align 8
  %r_dst1 = getelementptr inbounds double, ptr %r_dst, i64 1
  %dr_dst1 = getelementptr inbounds double, ptr %dr_dst, i64 1
  store double 0.000000e+00, ptr %r_dst, align 8
  store double 0.000000e+00, ptr %r_dst1, align 8
  call void (...) @__enzyme_autodiff(ptr @rev_copy_const_doubles, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %r_dst, ptr %r_dst)
  %rd0 = load double, ptr %r_dst, align 8
  %rd1 = load double, ptr %r_dst1, align 8
  call void @check_eq(double %rd0, double 7.000000e+00)
  call void @check_eq(double %rd1, double 8.000000e+00)

  store double 5.000000e+00, ptr %dr_dst, align 8
  store double 6.000000e+00, ptr %dr_dst1, align 8
  call void (...) @__enzyme_autodiff(ptr @rev_copy_const_doubles, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %r_dst, ptr %dr_dst)
  %drd0 = load double, ptr %dr_dst, align 8
  %drd1 = load double, ptr %dr_dst1, align 8
  call void @check_eq(double %drd0, double 0.000000e+00)
  call void @check_eq(double %drd1, double 0.000000e+00)

  ret i32 0
}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Float@double"}
!2 = !{!"Unknown", i32 -1, !3}
!3 = !{!"Pointer", i32 -1, !1}
