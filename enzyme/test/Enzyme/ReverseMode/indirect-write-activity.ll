; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

@enzyme_dup = global i32 0
@enzyme_const = global i32 0
@const_val = private unnamed_addr constant double 3.000000e+00, align 8

declare double @__enzyme_autodiff(...)
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

define double @df_closure_val_ptr(double %x, ptr %y, ptr %dy) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f_closure_val_ptr, double %x, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret double %r
}

; Overwriting external memory with a constant through a loaded pointer must
; still zero the external shadow in the reverse pass.
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
  call void (...) @__enzyme_autodiff(ptr @f_overwrite_const, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Overwriting external memory with memset through a loaded pointer must still
; zero the external shadow.
define void @f_overwrite_memset(ptr %y) {
entry:
  %c = alloca [2 x ptr], align 8
  store ptr %y, ptr %c, align 8
  %yp = load ptr, ptr %c, align 8
  call void @llvm.memset.p0.i64(ptr %yp, i8 0, i64 8, i1 false), !tbaa !4
  ret void
}

define void @df_overwrite_memset(ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f_overwrite_memset, ptr @enzyme_dup, ptr %y, ptr %dy)
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
  call void (...) @__enzyme_autodiff(ptr @f_overwrite_memcpy_const, ptr @enzyme_dup, ptr %y, ptr %dy)
  ret void
}

; Overwriting external memory with memset through a pointer loaded from a
; noalias holder must still zero the external shadow in the reverse pass.
define void @f_arg_ptr_memset_noalias(ptr noalias %p_holder) {
entry:
  %yp = load ptr, ptr %p_holder, align 8, !tbaa !0
  call void @llvm.memset.p0.i64(ptr %yp, i8 0, i64 8, i1 false), !tbaa !4
  ret void
}

define void @df_arg_ptr_memset_noalias(ptr %p_holder, ptr %dp_holder) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f_arg_ptr_memset_noalias, ptr @enzyme_dup, ptr %p_holder, ptr %dp_holder)
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"double", !2, i64 0}

; CHECK: define internal { double } @diffef_closure_val_ptr(double %x, ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca { ptr, double }
; CHECK: store ptr %"y'", ptr %"c'ipa"
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: invertentry:
; CHECK: load double, ptr %"yp'ipl"
; CHECK: store double 0.000000e+00, ptr %"yp'ipl"
; CHECK: load double, ptr %"c1'ipg"
; CHECK: ret { double }

; CHECK: define internal void @diffef_overwrite_const(ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca [2 x ptr]
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: invertentry:
; CHECK: store double 0.000000e+00, ptr %"yp'ipl"

; CHECK: define internal void @diffef_overwrite_memset(ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca [2 x ptr]
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: invertentry:
; CHECK: call void @llvm.memset.p0.i64(ptr {{(align 1 )?}}%"yp'ipl", i8 0, i64 8, i1 false)

; CHECK: define internal void @diffef_overwrite_memcpy_const(ptr %y, ptr %"y'")
; CHECK: %"c'ipa" = alloca [2 x ptr]
; CHECK: %"yp'ipl" = load ptr, ptr %"c'ipa"
; CHECK: invertentry:
; CHECK: call void @llvm.memset.p0.i64(ptr align 1 %"yp'ipl", i8 0, i64 8, i1 false)

; CHECK: define internal void @diffef_arg_ptr_memset_noalias(ptr noalias nocapture readonly %p_holder, ptr nocapture %"p_holder'")
; CHECK: %"yp'ipl" = load ptr, ptr %"p_holder'"
; CHECK: invertentry:
; CHECK: call void @llvm.memset.p0.i64(ptr %"yp'ipl", i8 0, i64 8, i1 false)
