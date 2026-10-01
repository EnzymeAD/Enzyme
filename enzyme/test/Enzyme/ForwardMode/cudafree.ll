; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Enzyme already knows how to give a CUDA allocation a shadow, but the frees
; that release one were not recognized as deallocations, so differentiating one
; reported a missing derivative and the shadow was leaked. Both APIs have to
; work: the runtime API, whose allocation is an ordinary pointer, and the
; driver API, whose CUdeviceptr is an integer -- the latter also covers the
; checked-free wrapper, which must not put nocapture on a non-pointer.

declare i32 @cudaMalloc(ptr, i64)
declare i32 @cudaFree(ptr)
declare i32 @cuMemAlloc_v2(ptr, i64)
declare i32 @cuMemFree_v2(i64)

define double @runtime(double %x) {
entry:
  %pp = alloca ptr
  %a = call i32 @cudaMalloc(ptr %pp, i64 8)
  %p = load ptr, ptr %pp
  store double %x, ptr %p
  %v = load double, ptr %p
  %f = call i32 @cudaFree(ptr %p)
  ret double %v
}

define double @driver(double %x) {
entry:
  %pp = alloca i64
  %a = call i32 @cuMemAlloc_v2(ptr %pp, i64 8)
  %d = load i64, ptr %pp
  %p = inttoptr i64 %d to ptr
  store double %x, ptr %p
  %v = load double, ptr %p
  %f = call i32 @cuMemFree_v2(i64 %d)
  ret double %v
}

declare double @__enzyme_fwddiff(...)

define double @test_runtime(double %x, double %dx) {
  %r = call double (...) @__enzyme_fwddiff(ptr @runtime, double %x, double %dx)
  ret double %r
}

define double @test_driver(double %x, double %dx) {
  %r = call double (...) @__enzyme_fwddiff(ptr @driver, double %x, double %dx)
  ret double %r
}

; The shadow allocation is zeroed, carries the tangent, and is released next to
; the primal through the checked free.

; CHECK: define internal double @fwddifferuntime(double %x, double %"x'")
; CHECK: call i32 @cudaMalloc(ptr %"pp'ipa", i64 8)
; CHECK: call i32 @cudaMemset(
; CHECK: %a = call i32 @cudaMalloc(ptr %pp, i64 8)
; CHECK: store double %"x'", ptr %"p'ipl"
; CHECK: %"v'ipl" = load double, ptr %"p'ipl"
; CHECK: %f = call i32 @cudaFree(ptr %p)
; CHECK: icmp ne ptr %p, %"p'ipl"
; CHECK: call i32 @cudaFree(ptr %"p'ipl")
; CHECK: ret double %"v'ipl"

; CHECK: define internal double @fwddiffedriver(double %x, double %"x'")
; CHECK: call i32 @cuMemAlloc_v2(ptr %"pp'ipa", i64 8)
; CHECK: call i32 @cuMemsetD8_v2(
; CHECK: %a = call i32 @cuMemAlloc_v2(ptr %pp, i64 8)
; CHECK: store double %"x'", ptr %"p'ipc"
; CHECK: %"v'ipl" = load double, ptr %"p'ipc"
; CHECK: %f = call i32 @cuMemFree_v2(i64 %d)
; CHECK: icmp ne i64 %d, %"d'ipl"
; CHECK: call i32 @cuMemFree_v2(i64 %"d'ipl")
; CHECK: ret double %"v'ipl"

; A CUdeviceptr is passed as an integer, so the checked-free wrapper built for
; it must leave nocapture off its parameters.
; CHECK: define internal void @__enzyme_checked_free_1_cuMemFree_v2(i64 %0, i64 %1)
