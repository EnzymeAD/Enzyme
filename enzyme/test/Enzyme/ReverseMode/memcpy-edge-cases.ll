; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare ptr @memcpy(ptr noalias returned, ptr noalias, i64)
declare double @__enzyme_autodiff(...)
declare void @__enzyme_autodiff.v(...)

%struct.Mixed = type { i64, double }

@const_mixed = private unnamed_addr constant %struct.Mixed { i64 42, double 9.5 }, align 16
@const_dbl = private unnamed_addr constant double 42.0, align 8

; 1. Mixed {i64, double} memcpy from constant global:
;    - Non-float prefix [0, 8) copies into shadow with align 16.
;    - Float suffix [8, 16) zeroes shadow at offset 8 with align 8 (not offset 0 / align 16).
define double @rev_mixed_const(ptr %dst, double %x) {
entry:
  %p0 = getelementptr inbounds %struct.Mixed, ptr %dst, i32 0, i32 0
  %p1 = getelementptr inbounds %struct.Mixed, ptr %dst, i32 0, i32 1
  store double %x, ptr %p1, align 8
  %v0 = load i64, ptr %p0, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dst, ptr align 16 @const_mixed, i64 16, i1 false)
  %v1 = load double, ptr %p1, align 8
  %v0f = sitofp i64 %v0 to double
  %r = fadd double %v1, %v0f
  ret double %r
}

define double @drev_mixed_const(ptr %dst, ptr %ddst, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @rev_mixed_const, metadata !"enzyme_dup", ptr %dst, ptr %ddst, double %x)
  ret double %r
}

; 2. ReverseMode + runtimeActivity memcpy from constant global:
;    Must guard shadow memset length with dst_inactive = (ddst == dst).
define void @rev_ra_const_src(ptr %dst) {
entry:
  %dummy = load double, ptr %dst, align 8
  %dummy2 = fadd double %dummy, 0.0
  store double %dummy2, ptr %dst, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 @const_dbl, i64 8, i1 false)
  ret void
}

define void @drev_ra_const_src(ptr %dst, ptr %ddst) {
entry:
  call void (...) @__enzyme_autodiff.v(ptr @rev_ra_const_src, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %dst, ptr %ddst)
  ret void
}

; 3. ReverseMode + runtimeActivity non-float (pointer) segment memcpy:
;    Must guard shadow copy length with dst_inactive = (ddst == dst).
define void @rev_ra_ptr(ptr %dst, ptr %src) {
entry:
  %p = load ptr, ptr %src, align 8
  %v = load double, ptr %p, align 8
  %v2 = fadd double %v, 1.0
  store double %v2, ptr %p, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 8, i1 false)
  ret void
}

define void @drev_ra_ptr(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc) {
entry:
  call void (...) @__enzyme_autodiff.v(ptr @rev_ra_ptr, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %dst, ptr %ddst, metadata !"enzyme_dup", ptr %src, ptr %dsrc)
  ret void
}

; 4. C memcpy returning ptr with return/param attributes and mixed struct:
;    Must resolve shadow placeholder, strip returned/return-index attrs,
;    and propagate types across return value.
define double @rev_c_memcpy_ret(ptr %dst, ptr %src) {
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

define void @drev_c_memcpy_ret(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc) {
entry:
  call void (...) @__enzyme_autodiff.v(ptr @rev_c_memcpy_ret, ptr %dst, ptr %ddst, ptr %src, ptr %dsrc)
  ret void
}

; 5. >2GB memcpy (4GB): must not truncate ShiftIndices size to 32-bit int or loop O(size).
define double @rev_huge_copy(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 4294967296, i1 false)
  %v = load double, ptr %dst, align 8
  ret double %v
}

define void @drev_huge_copy(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc) {
entry:
  call void (...) @__enzyme_autodiff.v(ptr @rev_huge_copy, ptr %dst, ptr %ddst, ptr %src, ptr %dsrc)
  ret void
}

; CHECK: define internal { double } @differev_mixed_const(ptr nocapture %dst, ptr nocapture %"dst'", double %x, double %differeturn)
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %"dst'", ptr align 16 @const_mixed, i64 8, i1 false)
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dst, ptr align 16 @const_mixed, i64 16, i1 false)
; CHECK: invertentry:
; CHECK:   %[[off:.+]] = getelementptr inbounds i8, ptr %"dst'", i64 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %[[off]], i8 0, i64 8, i1 false)

; CHECK: define internal void @differev_ra_const_src(ptr nocapture %dst, ptr nocapture %"dst'")
; CHECK: invertentry:
; CHECK-NEXT:   %[[inact:.+]] = icmp eq ptr %"dst'", %dst
; CHECK-NEXT:   %[[zlen:.+]] = select i1 %[[inact]], i64 0, i64 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"dst'", i8 0, i64 %[[zlen]], i1 false)

; CHECK: define internal void @differev_ra_ptr(ptr nocapture writeonly %dst, ptr nocapture %"dst'", ptr nocapture readonly %src, ptr nocapture %"src'")
; CHECK:   %[[pinact:.+]] = icmp eq ptr %"dst'", %dst
; CHECK-NEXT:   %[[plen:.+]] = select i1 %[[pinact]], i64 0, i64 8
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %"dst'", ptr align 8 %"src'", i64 %[[plen]], i1 false)

; CHECK: define internal void @differev_c_memcpy_ret(ptr %dst, ptr %"dst'", ptr %src, ptr %"src'", double %differeturn)
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 dereferenceable(16) %"dst'", ptr align 16 dereferenceable(16) %"src'", i64 8, i1 false)
; CHECK-NEXT:   %r = call nonnull align 16 ptr @memcpy(ptr returned align 16 dereferenceable(16) %dst, ptr align 16 dereferenceable(16) %src, i64 16)
; CHECK-NEXT:   %"p1'ipg" = getelementptr inbounds %struct.Mixed, ptr %"dst'", i32 0, i32 1
; CHECK: invertentry:
; CHECK:   %[[doff:.+]] = getelementptr inbounds i8, ptr %"dst'", i64 8
; CHECK-NEXT:   %[[soff:.+]] = getelementptr inbounds i8, ptr %"src'", i64 8
; CHECK:   load double, ptr {{%.+}}, align 8

; CHECK: define internal void @differev_huge_copy(ptr nocapture %dst, ptr nocapture %"dst'", ptr nocapture readonly %src, ptr nocapture %"src'", double %differeturn)
; CHECK:   icmp eq i64 536870912, %idx.next.i
