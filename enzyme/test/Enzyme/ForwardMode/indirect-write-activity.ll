; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

@enzyme_dup = global i32 0
@g_closure = global { ptr, double } zeroinitializer, align 8, !enzyme_shadow !0
@g_closure_shadow = global { ptr, double } zeroinitializer, align 8
@const_val = private unnamed_addr constant double 3.000000e+00, align 8

declare void @__enzyme_fwddiff(...)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

; Writing a captured value through a pointer loaded from a local closure alloca.
define void @f_closure_val_ptr(double %x, ptr %y) {
entry:
  %c = alloca { ptr, double }, align 8
  store ptr %y, ptr %c, align 8
  %c1 = getelementptr inbounds i8, ptr %c, i64 8
  store double %x, ptr %c1, align 8
  %v = load double, ptr %c1, align 8
  %yp = load ptr, ptr %c, align 8
  store double %v, ptr %yp, align 8
  ret void
}

define void @df_closure_val_ptr(double %x, double %dx, ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f_closure_val_ptr, double %x, double %dx, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Writing through a 2-hop loaded pointer chain.
define void @f_2hop_ptr(double %x, ptr %y) {
entry:
  %c = alloca [2 x ptr], align 8
  %d = alloca [2 x ptr], align 8
  store ptr %y, ptr %d, align 8
  store ptr %d, ptr %c, align 8
  %p = load ptr, ptr %c, align 8
  %yp = load ptr, ptr %p, align 8
  store double %x, ptr %yp, align 8
  ret void
}

define void @df_2hop_ptr(double %x, double %dx, ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f_2hop_ptr, double %x, double %dx, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Writing a captured value through a pointer loaded from a global closure.
define void @f_global_closure(double %x, ptr %y) {
entry:
  store ptr %y, ptr @g_closure, align 8
  %g1 = getelementptr inbounds i8, ptr @g_closure, i64 8
  store double %x, ptr %g1, align 8
  %v = load double, ptr %g1, align 8
  %yp = load ptr, ptr @g_closure, align 8
  store double %v, ptr %yp, align 8
  ret void
}

define void @df_global_closure(double %x, double %dx, ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f_global_closure, double %x, double %dx, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Overwriting external memory with a constant through a loaded pointer must
; still zero the external shadow.
define void @f_overwrite_const(ptr %y) {
entry:
  %c = alloca [2 x ptr], align 8
  store ptr %y, ptr %c, align 8
  %yp = load ptr, ptr %c, align 8
  store double 0.000000e+00, ptr %yp, align 8
  ret void
}

define void @df_overwrite_const(ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f_overwrite_const, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Overwriting external memory with memset through a loaded pointer must still
; zero the external shadow.
define void @f_overwrite_memset(ptr %y) {
entry:
  %c = alloca [2 x ptr], align 8
  store ptr %y, ptr %c, align 8
  %yp = load ptr, ptr %c, align 8
  call void @llvm.memset.p0.i64(ptr %yp, i8 0, i64 8, i1 false), !tbaa !5
  ret void
}

define void @df_overwrite_memset(ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f_overwrite_memset, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Overwriting external memory with memcpy from constant data through a loaded
; pointer must still zero the external shadow.
define void @f_overwrite_memcpy_const(ptr %y) {
entry:
  %c = alloca [2 x ptr], align 8
  store ptr %y, ptr %c, align 8
  %yp = load ptr, ptr %c, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %yp, ptr @const_val, i64 8, i1 false)
  ret void
}

define void @df_overwrite_memcpy_const(ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f_overwrite_memcpy_const, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Overwriting external memory with memset through a pointer loaded from a
; noalias holder must still zero the external shadow.
define void @f_arg_ptr_memset_noalias(ptr noalias %p_holder) {
entry:
  %yp = load ptr, ptr %p_holder, align 8, !tbaa !1
  call void @llvm.memset.p0.i64(ptr %yp, i8 0, i64 8, i1 false), !tbaa !5
  ret void
}

define void @df_arg_ptr_memset_noalias(ptr %p_holder, ptr %dp_holder) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f_arg_ptr_memset_noalias, ptr @enzyme_dup, ptr %p_holder, ptr %dp_holder)
  ret void
}

!0 = !{ptr @g_closure_shadow}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !3, i64 0}

; CHECK: define internal void @fwddiffef_closure_val_ptr(double %x, double %"x'", ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca { ptr, double }
; CHECK: store ptr %"y'", ptr %"c'ipa"
; CHECK: store double %"x'", ptr %"c1'ipg"
; CHECK: %"v'ipl" = load double, ptr %"c1'ipg"
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: store double %"v'ipl", ptr %"yp'ipl"

; CHECK: define internal void @fwddiffef_2hop_ptr(double %x, double %"x'", ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca [2 x ptr]
; CHECK: %"d'ipa" = alloca [2 x ptr]
; CHECK: %"yp'ipl" = load ptr, ptr %"p'ipl"
; CHECK: store double %"x'", ptr %"yp'ipl"

; CHECK: define internal void @fwddiffef_global_closure(double %x, double %"x'", ptr %y, ptr %"y'")
; CHECK: store ptr %"y'", ptr @g_closure_shadow
; CHECK: store double %"x'", ptr getelementptr (i8, ptr @g_closure_shadow, i64 8)
; CHECK: %"v'ipl" = load double, ptr getelementptr (i8, ptr @g_closure_shadow, i64 8)
; CHECK: %"yp'ipl" = load ptr, ptr @g_closure_shadow
; CHECK: store double %"v'ipl", ptr %"yp'ipl"

; CHECK: define internal void @fwddiffef_overwrite_const(ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca [2 x ptr]
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: store double 0.000000e+00, ptr %"yp'ipl"

; CHECK: define internal void @fwddiffef_overwrite_memset(ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca [2 x ptr]
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: call void @llvm.memset.p0.i64(ptr {{(align 1 )?}}%"yp'ipl", i8 0, i64 8, i1 false)

; CHECK: define internal void @fwddiffef_overwrite_memcpy_const(ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca [2 x ptr]
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: call void @llvm.memset.p0.i64(ptr align 1 %"yp'ipl", i8 0, i64 8, i1 false)

; CHECK: define internal void @fwddiffef_arg_ptr_memset_noalias(ptr noalias nocapture readonly %p_holder, ptr nocapture %"p_holder'")
; CHECK: %"yp'ipl" = load ptr, ptr %"p_holder'"
; CHECK: call void @llvm.memset.p0.i64(ptr %"yp'ipl", i8 0, i64 8, i1 false)
