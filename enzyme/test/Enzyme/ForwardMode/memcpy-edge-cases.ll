; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare ptr @memcpy(ptr noalias returned, ptr noalias, i64)
declare double @__enzyme_fwddiff(...)
declare void @__enzyme_fwddiff.v(...)

%struct.Mixed = type { i64, double }

@const_mixed = private unnamed_addr constant %struct.Mixed { i64 42, double 9.5 }, align 16
@const_dbl = private unnamed_addr constant double 42.0, align 8

; 1. C memcpy returning ptr with return/param attributes and mixed struct:
;    - Resolves shadow placeholder to %"dst'".
;    - Strips ReturnIndex attrs and parameter `returned`.
define double @fwd_c_memcpy_ret(ptr %dst, ptr %src) {
entry:
  %r = call nonnull align 16 ptr @memcpy(ptr align 16 dereferenceable(16) returned %dst, ptr align 16 dereferenceable(16) %src, i64 16)
  %p0 = getelementptr inbounds %struct.Mixed, ptr %r, i32 0, i32 0
  %iv = load i64, ptr %p0, align 8
  %ivf = sitofp i64 %iv to double
  %p1 = getelementptr inbounds %struct.Mixed, ptr %r, i32 0, i32 1
  %v = load double, ptr %p1, align 8
  %sum = fadd double %v, %ivf
  ret double %sum
}

define double @dfwd_c_memcpy_ret(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @fwd_c_memcpy_ret, ptr %dst, ptr %ddst, ptr %src, ptr %dsrc)
  ret double %r
}

; 2. Mixed {i64, double} C memcpy from constant global with align 16 dereferenceable(16):
;    At offset 8, emitted memset must have align 8 dereferenceable(8) (not align 16 / dereferenceable(16)).
define double @fwd_mixed_const_align(ptr %dst) {
entry:
  %r = call nonnull align 16 ptr @memcpy(ptr align 16 dereferenceable(16) returned %dst, ptr align 16 dereferenceable(16) @const_mixed, i64 16)
  %p0 = getelementptr inbounds %struct.Mixed, ptr %r, i32 0, i32 0
  %iv = load i64, ptr %p0, align 8
  %ivf = sitofp i64 %iv to double
  %p1 = getelementptr inbounds %struct.Mixed, ptr %r, i32 0, i32 1
  %v = load double, ptr %p1, align 8
  %sum = fadd double %v, %ivf
  ret double %sum
}

define double @dfwd_mixed_const_align(ptr %dst, ptr %ddst) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @fwd_mixed_const_align, metadata !"enzyme_dup", ptr %dst, ptr %ddst)
  ret double %r
}

; 3. ForwardMode + runtimeActivity memcpy from constant global:
;    Must guard shadow memset length with dst_inactive = (ddst == dst).
define void @fwd_ra_const_src(ptr %dst) {
entry:
  %dummy = load double, ptr %dst, align 8
  %dummy2 = fadd double %dummy, 0.0
  store double %dummy2, ptr %dst, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 @const_dbl, i64 8, i1 false)
  ret void
}

define void @dfwd_ra_const_src(ptr %dst, ptr %ddst) {
entry:
  call void (...) @__enzyme_fwddiff.v(ptr @fwd_ra_const_src, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %dst, ptr %ddst)
  ret void
}

; 4. ForwardMode + runtimeActivity overlapping memmove:
;    Must check dst_inactive = (ddst == dst) so primal memory is not zeroed before primal memmove.
define void @fwd_ra_memmove_overlap(ptr %p) {
entry:
  %p1 = getelementptr inbounds double, ptr %p, i64 1
  %dummy = load double, ptr %p, align 8
  %dummy2 = fadd double %dummy, 0.0
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %p1, ptr align 8 %p, i64 16, i1 false)
  ret void
}

define void @dfwd_ra_memmove_overlap(ptr %p, ptr %dp) {
entry:
  call void (...) @__enzyme_fwddiff.v(ptr @fwd_ra_memmove_overlap, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp)
  ret void
}

; CHECK: define internal double @fwddiffefwd_c_memcpy_ret(ptr %dst, ptr %"dst'", ptr %src, ptr %"src'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 dereferenceable(16) %"dst'", ptr align 16 dereferenceable(16) %"src'", i64 16, i1 false)
; CHECK-NEXT:   %r = call nonnull align 16 ptr @memcpy(ptr returned align 16 dereferenceable(16) %dst, ptr align 16 dereferenceable(16) %src, i64 16)
; CHECK-NEXT:   %"p1'ipg" = getelementptr inbounds %struct.Mixed, ptr %"dst'", i32 0, i32 1
; CHECK-NEXT:   %"v'ipl" = load double, ptr %"p1'ipg", align 8
; CHECK-NEXT:   ret double %"v'ipl"

; CHECK: define internal double @fwddiffefwd_mixed_const_align(ptr %dst, ptr %"dst'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 dereferenceable(16) %"dst'", ptr align 16 dereferenceable(16) @const_mixed, i64 8, i1 false)
; CHECK-NEXT:   %[[off:.+]] = getelementptr inbounds i8, ptr %"dst'", i64 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 dereferenceable(8) %[[off]], i8 0, i64 8, i1 false)

; CHECK: define internal void @fwddiffefwd_ra_const_src(ptr nocapture %dst, ptr nocapture %"dst'")
; CHECK:   %[[cinact:.+]] = icmp eq ptr %"dst'", %dst
; CHECK-NEXT:   %[[czlen:.+]] = select i1 %[[cinact]], i64 0, i64 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"dst'", i8 0, i64 %[[czlen]], i1 false)

; CHECK: define internal void @fwddiffefwd_ra_memmove_overlap(ptr nocapture %p, ptr nocapture %"p'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"p1'ipg" = getelementptr inbounds double, ptr %"p'", i64 1
; CHECK-NEXT:   %p1 = getelementptr inbounds double, ptr %p, i64 1
; CHECK-NEXT:   %[[din:.+]] = icmp eq ptr %"p1'ipg", %p1
; CHECK-NEXT:   %[[sin:.+]] = icmp eq ptr %"p'", %p
; CHECK-NEXT:   %[[skip:.+]] = or i1 %[[din]], %[[sin]]
; CHECK-NEXT:   %[[clen:.+]] = select i1 %[[skip]], i64 0, i64 16
; CHECK-NEXT:   %[[dact:.+]] = xor i1 %[[din]], true
; CHECK-NEXT:   %[[doz:.+]] = and i1 %[[dact]], %[[sin]]
; CHECK-NEXT:   %[[zlen:.+]] = select i1 %[[doz]], i64 16, i64 0
; CHECK-NEXT:   call void @llvm.memmove.p0.p0.i64(ptr align 8 %"p1'ipg", ptr align 8 %"p'", i64 %[[clen]], i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"p1'ipg", i8 0, i64 %[[zlen]], i1 false)
; CHECK-NEXT:   call void @llvm.memmove.p0.p0.i64(ptr align 8 %p1, ptr align 8 %p, i64 16, i1 false)
; CHECK-NEXT:   ret void
