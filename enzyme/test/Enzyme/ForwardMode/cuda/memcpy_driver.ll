; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; A round trip of a double through device memory: host -> device -> host. The
; derivative has to repeat every transfer on the shadow buffers, which means
; calling back into the CUDA driver API rather than emitting an LLVM memcpy.

declare i32 @cuMemAlloc_v2(ptr, i64)
declare i32 @cuMemFree_v2(i64)
declare i32 @cuMemcpyHtoD_v2(i64, ptr, i64)
declare i32 @cuMemcpyDtoH_v2(ptr, i64, i64)

define double @roundtrip(ptr %host) {
entry:
  %devp = alloca i64
  %a = call i32 @cuMemAlloc_v2(ptr %devp, i64 8)
  %dev = load i64, ptr %devp
  %c1 = call i32 @cuMemcpyHtoD_v2(i64 %dev, ptr %host, i64 8)
  %out = alloca double
  %c2 = call i32 @cuMemcpyDtoH_v2(ptr %out, i64 %dev, i64 8)
  %v = load double, ptr %out
  %fr = call i32 @cuMemFree_v2(i64 %dev)
  ret double %v
}

declare double @__enzyme_fwddiff(...)

define double @test(ptr %host, ptr %dhost) {
  %r = call double (...) @__enzyme_fwddiff(ptr @roundtrip, metadata !"enzyme_dup", ptr %host, ptr %dhost)
  ret double %r
}

; CHECK: define internal double @fwddifferoundtrip(ptr %host, ptr %"host'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"devp'ipa" = alloca i64
; CHECK-NEXT:   store i64 0, ptr %"devp'ipa"
; CHECK-NEXT:   %devp = alloca i64
; CHECK-NEXT:   %{{.+}} = call i32 @cuMemAlloc_v2(ptr %"devp'ipa", i64 8)
; CHECK-NEXT:   %[[shadowdev:.+]] = load ptr, ptr %"devp'ipa"
; CHECK-NEXT:   %{{.+}} = call i32 @cuMemsetD8_v2(ptr nonnull %[[shadowdev]], i8 0, i64 8)
; CHECK-NEXT:   %a = call i32 @cuMemAlloc_v2(ptr %devp, i64 8)
; CHECK-NEXT:   %"dev'ipl" = load i64, ptr %"devp'ipa"
; CHECK-NEXT:   %dev = load i64, ptr %devp
; CHECK-NEXT:   %[[h1:.+]] = inttoptr i64 %"dev'ipl" to ptr
; CHECK-NEXT:   %[[h2:.+]] = ptrtoint ptr %[[h1]] to i64
; CHECK-NEXT:   %{{.+}} = call i32 @cuMemcpyHtoD_v2(i64 %[[h2]], ptr %"host'", i64 8)
; CHECK-NEXT:   %c1 = call i32 @cuMemcpyHtoD_v2(i64 %dev, ptr %host, i64 8)
; CHECK-NEXT:   %"out'ipa" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, ptr %"out'ipa"
; CHECK-NEXT:   %out = alloca double
; CHECK-NEXT:   %[[d1:.+]] = inttoptr i64 %"dev'ipl" to ptr
; CHECK-NEXT:   %[[d2:.+]] = ptrtoint ptr %[[d1]] to i64
; CHECK-NEXT:   %{{.+}} = call i32 @cuMemcpyDtoH_v2(ptr %"out'ipa", i64 %[[d2]], i64 8)
; CHECK-NEXT:   %c2 = call i32 @cuMemcpyDtoH_v2(ptr %out, i64 %dev, i64 8)
; CHECK-NEXT:   %"v'ipl" = load double, ptr %"out'ipa"
; CHECK-NEXT:   %fr = call i32 @cuMemFree_v2(i64 %dev)
; CHECK-NEXT:   %{{.+}} = icmp ne i64 %dev, %"dev'ipl"
; CHECK-NEXT:   br i1 %{{.+}}, label %free0.i, label %__enzyme_checked_free_1_cuMemFree_v2.exit

; CHECK: free0.i:
; CHECK-NEXT:   %{{.+}} = call i32 @cuMemFree_v2(i64 %"dev'ipl")
; CHECK-NEXT:   br label %__enzyme_checked_free_1_cuMemFree_v2.exit

; CHECK: __enzyme_checked_free_1_cuMemFree_v2.exit:
; CHECK-NEXT:   ret double %"v'ipl"
