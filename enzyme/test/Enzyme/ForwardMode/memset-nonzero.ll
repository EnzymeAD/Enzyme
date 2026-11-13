; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

%struct.mixed = type { double, i64 }

@.pat_2d = private unnamed_addr constant [2 x double] [double 1.5, double 2.5], align 16
@.pat_mixed = private unnamed_addr constant %struct.mixed { double 3.5, i64 123456789 }, align 16

declare void @__enzyme_fwddiff(...)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1)
declare void @memset_pattern16(ptr nocapture writeonly, ptr nocapture readonly, i64)
declare ptr @memset(ptr, i32, i64)

define void @f_nonzero(ptr nocapture %y) {
  call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 8, i1 false)
  %v = load double, ptr %y, align 8
  %v2 = fmul double %v, 2.0
  store double %v2, ptr %y, align 8
  ret void
}

define double @f_mixed(ptr nocapture %s, double %x) {
  call void @llvm.memset.p0.i64(ptr nonnull align 16 dereferenceable(16) %s, i8 7, i64 16, i1 false)
  %d = load double, ptr %s, align 8
  %ip = getelementptr inbounds %struct.mixed, ptr %s, i32 0, i32 1
  %iv = load i64, ptr %ip, align 8
  %ivf = sitofp i64 %iv to double
  %cmp = fcmp oeq double %ivf, 0.0
  %res0 = fadd double %d, %x
  %res = select i1 %cmp, double 0.0, double %res0
  ret double %res
}

define void @f_pattern(ptr nocapture writeonly %x) {
  call void @memset_pattern16(ptr align 16 %x, ptr align 16 @.pat_2d, i64 16)
  ret void
}

define double @f_pattern_mixed(ptr nocapture %s, double %x) {
  call void @memset_pattern16(ptr align 16 %s, ptr align 16 @.pat_mixed, i64 16)
  %d = load double, ptr %s, align 8
  %ip = getelementptr inbounds %struct.mixed, ptr %s, i32 0, i32 1
  %iv = load i64, ptr %ip, align 8
  %ivf = sitofp i64 %iv to double
  %cmp = fcmp oeq double %ivf, 0.0
  %res0 = fadd double %d, %x
  %res = select i1 %cmp, double 0.0, double %res0
  ret double %res
}

define double @f_ret(ptr %p, double %x) {
  %r = call ptr @memset(ptr %p, i32 63, i64 8)
  %old = load double, ptr %r, align 8
  %sum = fadd double %old, %x
  store double %sum, ptr %r, align 8
  %v = load double, ptr %r, align 8
  ret double %v
}

define double @f_rt(ptr nocapture %y) {
  call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 8, i1 false)
  %v = load double, ptr %y, align 8
  ret double %v
}

define void @caller(ptr %y, ptr %dy, ptr %s, ptr %ds, double %x, double %dx) {
  call void (...) @__enzyme_fwddiff(ptr @f_nonzero, metadata !"enzyme_dup", ptr %y, ptr %dy)
  %r1 = call double (...) @__enzyme_fwddiff(ptr @f_mixed, metadata !"enzyme_dup", ptr %s, ptr %ds, double %x, double %dx)
  call void (...) @__enzyme_fwddiff(ptr @f_pattern, metadata !"enzyme_dup", ptr %s, ptr %ds)
  %r2 = call double (...) @__enzyme_fwddiff(ptr @f_pattern_mixed, metadata !"enzyme_dup", ptr %s, ptr %ds, double %x, double %dx)
  %r3 = call double (...) @__enzyme_fwddiff(ptr @f_ret, metadata !"enzyme_dup", ptr %y, ptr %dy, double %x, double %dx)
  %r4 = call double (...) @__enzyme_fwddiff(ptr @f_rt, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %y, ptr %dy)
  ret void
}

; CHECK: define internal void @fwddiffef_nonzero(ptr nocapture %y, ptr nocapture %"y'")
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 8, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"y'", i8 0, i64 8, i1 false)

; CHECK: define internal double @fwddiffef_mixed(ptr nocapture %s, ptr nocapture %"s'", double %x, double %"x'")
; CHECK:   call void @llvm.memset.p0.i64(ptr nonnull align 16 dereferenceable(16) %s, i8 7, i64 16, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull align 16 dereferenceable(16) %"s'", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %[[GEP:.+]] = getelementptr inbounds i8, ptr %"s'", i32 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull align 8 dereferenceable(8) %[[GEP]], i8 7, i64 8, i1 false)

; CHECK: define internal void @fwddiffef_pattern(ptr nocapture writeonly %x, ptr nocapture %"x'")
; CHECK:   call void @memset_pattern16(ptr align 16 %x, ptr align 16 @.pat_2d, i64 16)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 16 %"x'", i8 0, i64 16, i1 false)

; CHECK: define internal double @fwddiffef_pattern_mixed(ptr nocapture %s, ptr nocapture %"s'", double %x, double %"x'")
; CHECK:   call void @memset_pattern16(ptr align 16 %s, ptr align 16 @.pat_mixed, i64 16)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 16 %"s'", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %[[PGEP:.+]] = getelementptr inbounds i8, ptr %"s'", i32 8
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[PGEP]], ptr align 8 getelementptr inbounds (i8, ptr @.pat_mixed, i32 8), i64 8, i1 false)

; CHECK: define internal double @fwddiffef_ret(ptr %p, ptr %"p'", double %x, double %"x'")
; CHECK:   %r = call ptr @memset(ptr %p, i32 63, i64 8)
; CHECK-NEXT:   %{{.+}} = call ptr @memset(ptr %"p'", i32 0, i64 8)
; CHECK-NEXT:   %"old'ipl" = load double, ptr %"p'", align 8

; CHECK: define internal double @fwddiffef_rt(ptr nocapture %y, ptr nocapture %"y'")
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 8, i1 false)
; CHECK-NEXT:   %[[EQ:.+]] = icmp eq ptr %"y'", %y
; CHECK-NEXT:   %[[SEL:.+]] = select i1 %[[EQ]], i64 0, i64 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"y'", i8 0, i64 %[[SEL]], i1 false)
