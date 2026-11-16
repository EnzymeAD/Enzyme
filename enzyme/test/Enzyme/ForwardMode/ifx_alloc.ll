; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s; fi

declare i32 @for_alloc_allocatable(i64, ptr nocapture, i32)
declare i32 @for_alloc_allocatable_handle(i64, ptr nocapture, i32, ptr)
declare i32 @for_allocate(i64, ptr nocapture, i32)
declare i32 @for_allocate_handle(i64, ptr nocapture, i32, ptr)
declare i32 @for_dealloc_allocatable(ptr nocapture readonly, i32)
declare i32 @for_dealloc_allocatable_handle(ptr nocapture readonly, i32, ptr)
declare i32 @for_deallocate(ptr nocapture readonly, i32)

define double @alloc_scalar(double %x, i32 %flags) {
entry:
  %pp = alloca ptr, align 8
  %a = call i32 @for_alloc_allocatable(i64 8, ptr %pp, i32 %flags)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  %f = call i32 @for_dealloc_allocatable(ptr %p, i32 %flags)
  ret double %sq
}

define double @alloc_handle(double %x, i32 %flags, ptr %coarray) {
entry:
  %pp = alloca ptr, align 8
  %a = call i32 @for_alloc_allocatable_handle(i64 8, ptr %pp, i32 %flags, ptr %coarray)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  %f = call i32 @for_dealloc_allocatable_handle(ptr %p, i32 %flags, ptr %coarray)
  ret double %sq
}

define double @alloc_ptr_variants(double %x, i32 %flags, ptr %coarray) {
entry:
  %pp1 = alloca ptr, align 8
  %pp2 = alloca ptr, align 8
  %a1 = call i32 @for_allocate(i64 8, ptr %pp1, i32 %flags)
  %a2 = call i32 @for_allocate_handle(i64 8, ptr %pp2, i32 %flags, ptr %coarray)
  %p1 = load ptr, ptr %pp1, align 8
  %p2 = load ptr, ptr %pp2, align 8
  store double %x, ptr %p1, align 8
  store double %x, ptr %p2, align 8
  %v1 = load double, ptr %p1, align 8
  %v2 = load double, ptr %p2, align 8
  %prod = fmul double %v1, %v2
  %f1 = call i32 @for_deallocate(ptr %p1, i32 262144)
  %f2 = call i32 @for_dealloc_allocatable_handle(ptr %p2, i32 %flags, ptr %coarray)
  ret double %prod
}

define double @inactive_int_alloc(double %x, i64 %n, i32 %flags) {
entry:
  %pp = alloca ptr, align 8
  %a = call i32 @for_alloc_allocatable(i64 8, ptr %pp, i32 %flags)
  %p = load ptr, ptr %pp, align 8
  store i64 %n, ptr %p, align 8
  %k = load i64, ptr %p, align 8
  %f = call i32 @for_dealloc_allocatable(ptr %p, i32 %flags)
  %kf = sitofp i64 %k to double
  %res = fmul double %x, %kf
  ret double %res
}

declare double @__enzyme_fwddiff(...)

define double @test_alloc_scalar(double %x, double %dx, i32 %flags) {
  %r = call double (...) @__enzyme_fwddiff(ptr @alloc_scalar, double %x, double %dx, i32 %flags)
  ret double %r
}

define double @test_alloc_handle(double %x, double %dx, i32 %flags, ptr %coarray) {
  %r = call double (...) @__enzyme_fwddiff(ptr @alloc_handle, double %x, double %dx, i32 %flags, metadata !"enzyme_const", ptr %coarray)
  ret double %r
}

define double @test_alloc_ptr_variants(double %x, double %dx, i32 %flags, ptr %coarray) {
  %r = call double (...) @__enzyme_fwddiff(ptr @alloc_ptr_variants, double %x, double %dx, i32 %flags, metadata !"enzyme_const", ptr %coarray)
  ret double %r
}

define double @test_inactive_int_alloc(double %x, double %dx, i64 %n, i32 %flags) {
  %r = call double (...) @__enzyme_fwddiff(ptr @inactive_int_alloc, double %x, double %dx, i64 %n, i32 %flags)
  ret double %r
}

; CHECK-LABEL: define internal double @fwddiffealloc_scalar(
; CHECK: call i32 @for_alloc_allocatable(i64 8, ptr %"pp'ipa", i32 %flags)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_alloc_allocatable(i64 8, ptr %pp, i32 %flags)
; CHECK: call i32 @for_dealloc_allocatable(ptr %p, i32 %flags)
; CHECK: call i32 @for_dealloc_allocatable(ptr %{{.+}}, i32 %flags)

; CHECK-LABEL: define internal double @fwddiffealloc_handle(
; CHECK: call i32 @for_alloc_allocatable_handle(i64 8, ptr %"pp'ipa", i32 %flags, ptr %coarray)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_alloc_allocatable_handle(i64 8, ptr %pp, i32 %flags, ptr %coarray)
; CHECK: call i32 @for_dealloc_allocatable_handle(ptr %p, i32 %flags, ptr %coarray)
; CHECK: call i32 @for_dealloc_allocatable_handle(ptr %{{.+}}, i32 %flags, ptr %coarray)

; CHECK-LABEL: define internal double @fwddiffealloc_ptr_variants(
; CHECK: call i32 @for_allocate(i64 8, ptr %"pp1'ipa", i32 %flags)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_allocate(i64 8, ptr %pp1, i32 %flags)
; CHECK: call i32 @for_allocate_handle(i64 8, ptr %"pp2'ipa", i32 %flags, ptr %coarray)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_allocate_handle(i64 8, ptr %pp2, i32 %flags, ptr %coarray)
; CHECK: call i32 @for_deallocate(ptr %p1, i32 262144)
; CHECK: call i32 @for_deallocate(ptr %{{.+}}, i32 262144)
; CHECK: call i32 @for_dealloc_allocatable_handle(ptr %p2, i32 %flags, ptr %coarray)
; CHECK: call i32 @for_dealloc_allocatable_handle(ptr %{{.+}}, i32 %flags, ptr %coarray)

; CHECK-LABEL: define internal double @fwddiffeinactive_int_alloc(
; CHECK-NOT: %"pp'ipa"
; CHECK: call i32 @for_alloc_allocatable(i64 8, ptr %pp, i32 %flags)
; CHECK-NOT: call i32 @for_alloc_allocatable
; CHECK: call i32 @for_dealloc_allocatable(ptr %p, i32 %flags)
; CHECK: ret double %{{.+}}
