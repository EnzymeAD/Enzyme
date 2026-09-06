; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Same round trip as memcpy_driver.ll, but through the CUDA runtime API, where
; the direction of the transfer is a cudaMemcpyKind argument that is replayed
; unchanged on the shadow.

declare i32 @cudaMalloc(ptr, i64)
declare i32 @cudaFree(ptr)
declare i32 @cudaMemcpy(ptr, ptr, i64, i32)

define double @roundtrip(ptr %host) {
entry:
  %devp = alloca ptr
  %a = call i32 @cudaMalloc(ptr %devp, i64 8)
  %dev = load ptr, ptr %devp
  ; cudaMemcpyHostToDevice
  %c1 = call i32 @cudaMemcpy(ptr %dev, ptr %host, i64 8, i32 1)
  %out = alloca double
  ; cudaMemcpyDeviceToHost
  %c2 = call i32 @cudaMemcpy(ptr %out, ptr %dev, i64 8, i32 2)
  %v = load double, ptr %out
  %fr = call i32 @cudaFree(ptr %dev)
  ret double %v
}

declare double @__enzyme_fwddiff(...)

define double @test(ptr %host, ptr %dhost) {
  %r = call double (...) @__enzyme_fwddiff(ptr @roundtrip, metadata !"enzyme_dup", ptr %host, ptr %dhost)
  ret double %r
}

; CHECK: define internal double @fwddifferoundtrip(ptr %host, ptr %"host'")
; CHECK: %{{.+}} = call i32 @cudaMemcpy(ptr %[[shadowdev:.+]], ptr %"host'", i64 8, i32 1)
; CHECK-NEXT: %c1 = call i32 @cudaMemcpy(ptr %dev, ptr %host, i64 8, i32 1)
; CHECK: %{{.+}} = call i32 @cudaMemcpy(ptr %"out'ipa", ptr %[[shadowdev]], i64 8, i32 2)
; CHECK-NEXT: %c2 = call i32 @cudaMemcpy(ptr %out, ptr %dev, i64 8, i32 2)
; CHECK: %"v'ipl" = load double, ptr %"out'ipa"
