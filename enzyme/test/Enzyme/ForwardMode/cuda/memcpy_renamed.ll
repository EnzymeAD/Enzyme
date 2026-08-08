; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Julia reaches the CUDA driver through lazily bound declarations named
; "ejlstr$<function>$<library>", tagging them with the real entry point in
; enzyme_math. Any helper Enzyme introduces alongside them (here the memset
; that zeroes a freshly allocated shadow buffer) has to follow the same naming
; convention, or it will not resolve when the module is JIT linked.

declare i32 @"ejlstr$cuMemAlloc_v2$libcuda.so.1"(ptr, i64) "enzyme_math"="cuMemAlloc_v2"
declare i32 @"ejlstr$cuMemFree_v2$libcuda.so.1"(i64) "enzyme_math"="cuMemFree_v2"
declare i32 @"ejlstr$cuMemcpyHtoD_v2$libcuda.so.1"(i64, ptr, i64) "enzyme_math"="cuMemcpyHtoD_v2"
declare i32 @"ejlstr$cuMemcpyDtoH_v2$libcuda.so.1"(ptr, i64, i64) "enzyme_math"="cuMemcpyDtoH_v2"

define double @roundtrip(ptr %host) {
entry:
  %devp = alloca i64
  %a = call i32 @"ejlstr$cuMemAlloc_v2$libcuda.so.1"(ptr %devp, i64 8)
  %dev = load i64, ptr %devp
  %c1 = call i32 @"ejlstr$cuMemcpyHtoD_v2$libcuda.so.1"(i64 %dev, ptr %host, i64 8)
  %out = alloca double
  %c2 = call i32 @"ejlstr$cuMemcpyDtoH_v2$libcuda.so.1"(ptr %out, i64 %dev, i64 8)
  %v = load double, ptr %out
  %fr = call i32 @"ejlstr$cuMemFree_v2$libcuda.so.1"(i64 %dev)
  ret double %v
}

declare double @__enzyme_fwddiff(...)

define double @test(ptr %host, ptr %dhost) {
  %r = call double (...) @__enzyme_fwddiff(ptr @roundtrip, metadata !"enzyme_dup", ptr %host, ptr %dhost)
  ret double %r
}

; CHECK: define internal double @fwddifferoundtrip(ptr %host, ptr %"host'")
; CHECK: call i32 @"ejlstr$cuMemsetD8_v2$libcuda.so.1"(
; CHECK: call i32 @"ejlstr$cuMemcpyHtoD_v2$libcuda.so.1"(i64 %{{.+}}, ptr %"host'", i64 8)
; CHECK: call i32 @"ejlstr$cuMemcpyDtoH_v2$libcuda.so.1"(ptr %"out'ipa", i64 %{{.+}}, i64 8)
; CHECK: %"v'ipl" = load double, ptr %"out'ipa"
; CHECK: call i32 @"ejlstr$cuMemFree_v2$libcuda.so.1"(i64 %"dev'ipl")
