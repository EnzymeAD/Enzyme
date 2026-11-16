; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s; fi

declare i32 @for_alloc_allocatable(i64, ptr nocapture, i32)
declare i32 @for_alloc_allocatable_handle(i64, ptr nocapture, i32, ptr)
declare i32 @for_allocate(i64, ptr nocapture, i32)
declare i32 @for_allocate_handle(i64, ptr nocapture, i32, ptr)
declare i32 @for_dealloc_allocatable(ptr nocapture readonly, i32)
declare i32 @for_dealloc_allocatable_handle(ptr nocapture readonly, i32, ptr)
declare i32 @for_deallocate(ptr nocapture readonly, i32)

define double @alloc_scalar(double %x, i32 %inflags) {
entry:
  %flags = or i32 %inflags, 262144
  %pp = alloca ptr, align 8
  %a = call i32 @for_alloc_allocatable(i64 8, ptr %pp, i32 %flags)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  %f = call i32 @for_dealloc_allocatable(ptr %p, i32 %flags)
  %ok = icmp eq i32 %f, 0
  br i1 %ok, label %done, label %err

err:
  ret double 0.0

done:
  ret double %sq
}

define double @alloc_handle(double %x, i32 %inflags, ptr %coarray) {
entry:
  %flags = or i32 %inflags, 262144
  %pp = alloca ptr, align 8
  %a = call i32 @for_alloc_allocatable_handle(i64 8, ptr %pp, i32 %flags, ptr %coarray)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  %f = call i32 @for_dealloc_allocatable_handle(ptr %p, i32 %flags, ptr %coarray)
  ret double %sq
}

define internal double @callee_handle(double %x, i32 %inflags, ptr %coarray) noinline {
entry:
  %flags = or i32 %inflags, 262144
  %pp = alloca ptr, align 8
  %a = call i32 @for_allocate_handle(i64 8, ptr %pp, i32 %flags, ptr %coarray)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  %f = call i32 @for_dealloc_allocatable_handle(ptr %p, i32 %flags, ptr %coarray)
  ret double %sq
}

define double @caller_split(double %x, i32 %inflags, ptr %coarray) {
entry:
  %r = call double @callee_handle(double %x, i32 %inflags, ptr %coarray)
  %sq = fmul double %r, %r
  ret double %sq
}

define double @alloc_ptr(double %x, i32 %flags) {
entry:
  %pp = alloca ptr, align 8
  %a = call i32 @for_allocate(i64 8, ptr %pp, i32 %flags)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %sq = fmul double %v, %v
  %f = call i32 @for_deallocate(ptr %p, i32 262144)
  ret double %sq
}

declare double @__enzyme_autodiff(...)

define double @test_alloc_scalar(double %x, i32 %inflags) {
  %r = call double (...) @__enzyme_autodiff(ptr @alloc_scalar, double %x, i32 %inflags)
  ret double %r
}

define double @test_alloc_handle(double %x, i32 %inflags, ptr %coarray) {
  %r = call double (...) @__enzyme_autodiff(ptr @alloc_handle, double %x, i32 %inflags, metadata !"enzyme_const", ptr %coarray)
  ret double %r
}

define double @test_caller_split(double %x, i32 %inflags, ptr %coarray) {
  %r = call double (...) @__enzyme_autodiff(ptr @caller_split, double %x, i32 %inflags, metadata !"enzyme_const", ptr %coarray)
  ret double %r
}

define double @test_alloc_ptr(double %x, i32 %flags) {
  %r = call double (...) @__enzyme_autodiff(ptr @alloc_ptr, double %x, i32 %flags)
  ret double %r
}

; CHECK-LABEL: define internal { double } @diffealloc_scalar(
; CHECK: call i32 @for_alloc_allocatable(i64 8, ptr %"pp'ipa", i32 %flags)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_alloc_allocatable(i64 8, ptr %pp, i32 %flags)
; CHECK-NOT: call i32 @for_dealloc_allocatable
; CHECK: invertentry:
; CHECK: call i32 @for_dealloc_allocatable(ptr %{{.+}}, i32 %flags)
; CHECK: call i32 @for_dealloc_allocatable(ptr %{{.+}}, i32 %flags)
; CHECK: ret { double }

; CHECK-LABEL: define internal { double } @diffealloc_handle(
; CHECK: call i32 @for_alloc_allocatable_handle(i64 8, ptr %"pp'ipa", i32 %flags, ptr %coarray)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_alloc_allocatable_handle(i64 8, ptr %pp, i32 %flags, ptr %coarray)
; CHECK-NOT: call i32 @for_dealloc_allocatable_handle
; CHECK: invertentry:
; CHECK: call i32 @for_dealloc_allocatable_handle(ptr %{{.+}}, i32 %flags, ptr %coarray)
; CHECK: call i32 @for_dealloc_allocatable_handle(ptr %{{.+}}, i32 %flags, ptr %coarray)
; CHECK: ret { double }

; CHECK-LABEL: define internal { double } @diffecaller_split(
; CHECK: call { { ptr, ptr, ptr }, double } @augmented_callee_handle(
; CHECK: call { double } @diffecallee_handle(

; CHECK-LABEL: define internal { { ptr, ptr, ptr }, double } @augmented_callee_handle(
; CHECK: call i32 @for_allocate_handle(i64 8, ptr %{{.+}}, i32 %flags, ptr %coarray)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_allocate_handle(i64 8, ptr %{{.+}}, i32 %flags, ptr %coarray)
; CHECK-NOT: call i32 @for_dealloc_allocatable_handle
; CHECK: ret { { ptr, ptr, ptr }, double }

; CHECK-LABEL: define internal { double } @diffecallee_handle(
; CHECK: call i32 @for_dealloc_allocatable_handle(ptr %{{.+}}, i32 %flags, ptr %coarray)
; CHECK: ret { double }

; CHECK-LABEL: define internal { double } @diffealloc_ptr(
; CHECK: call i32 @for_allocate(i64 8, ptr %"pp'ipa", i32 %flags)
; CHECK: call void @llvm.memset.p0.i64(ptr %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: call i32 @for_allocate(i64 8, ptr %pp, i32 %flags)
; CHECK-NOT: call i32 @for_deallocate
; CHECK: invertentry:
; CHECK: call i32 @for_dealloc_allocatable(ptr %{{.+}}, i32 %flags)
; CHECK: call i32 @for_dealloc_allocatable(ptr %{{.+}}, i32 %flags)
; CHECK: ret { double }
