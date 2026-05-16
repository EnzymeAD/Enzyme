; RUN: %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes=enzyme -S -opaque-pointers | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"
target triple = "arm64-apple-darwin24.0.0"

; Function Attrs: allocsize(1)
declare noalias ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64) #0

define fastcc ptr addrspace(10) @julia_mlogloss_core_3044(ptr addrspace(10) %"pred::Array") {
top:
  %"pred::Array.size.sroa.0.0.copyload" = load i64, ptr addrspace(11) null, align 8
  %0 = trunc i64 %"pred::Array.size.sroa.0.0.copyload" to i32
  %1 = zext i32 %0 to i64
  %"Memory{Float32}[]" = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 %1)
  %"new::Array" = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 1)
  %2 = addrspacecast ptr addrspace(10) %"new::Array" to ptr addrspace(11)
  store ptr addrspace(10) %"Memory{Float32}[]", ptr addrspace(11) %2, align 8
  %"new::Tuple19.sroa.3.0.new::Array.size_ptr.sroa_idx" = getelementptr i8, ptr addrspace(11) %2, i64 32
  %value_phi49.size.sroa.3.0.copyload = load i64, ptr addrspace(11) %"new::Tuple19.sroa.3.0.new::Array.size_ptr.sroa_idx", align 8
  %"new::Array103" = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 40)
  %3 = addrspacecast ptr addrspace(10) %"new::Array103" to ptr addrspace(11)
  store ptr addrspace(10) %"pred::Array", ptr addrspace(11) %3, align 8
  store i64 %value_phi49.size.sroa.3.0.copyload, ptr addrspace(11) %3, align 8
  %4 = load ptr addrspace(10), ptr addrspace(10) %"new::Array103", align 8
  ret ptr addrspace(10) %4
}

declare i8* @__enzyme_virtualreverse(...)

define void @test_enzyme() {
  %z = call i8* (...) @__enzyme_virtualreverse(ptr @julia_mlogloss_core_3044)
  ret void
}

attributes #0 = { allocsize(1) }

; CHECK: define internal fastcc void @diffejulia_mlogloss_core_3044(ptr addrspace(10) %"pred::Array", ptr addrspace(10) %"pred::Array'")
; CHECK-NEXT: top:
; CHECK-NEXT:   %"pred::Array.size.sroa.0.0.copyload" = load i64, ptr addrspace(11) null, align 8
; CHECK-NEXT:   %0 = trunc i64 %"pred::Array.size.sroa.0.0.copyload" to i32
; CHECK-NEXT:   %1 = zext i32 %0 to i64
; CHECK-NEXT:   %"Memory{Float32}[]'mi" = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 %1)
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull %"Memory{Float32}[]'mi", i8 0, i64 %1, i1 false)
; CHECK-NEXT:   %"Memory{Float32}[]" = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 %1)
; CHECK-NEXT:   %"new::Array" = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 1)
; CHECK-NEXT:   %2 = addrspacecast ptr addrspace(10) %"new::Array" to ptr addrspace(11)
; CHECK-NEXT:   store ptr addrspace(10) %"Memory{Float32}[]", ptr addrspace(11) %2, align 8
; CHECK-NEXT:   %"new::Tuple19.sroa.3.0.new::Array.size_ptr.sroa_idx" = getelementptr i8, ptr addrspace(11) %2, i64 32
; CHECK-NEXT:   %value_phi49.size.sroa.3.0.copyload = load i64, ptr addrspace(11) %"new::Tuple19.sroa.3.0.new::Array.size_ptr.sroa_idx", align 8
; CHECK-NEXT:   %"new::Array103'mi" = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 40)
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull dereferenceable(40) dereferenceable_or_null(40) %"new::Array103'mi", i8 0, i64 40, i1 false)
; CHECK-NEXT:   %"'ipc" = addrspacecast ptr addrspace(10) %"new::Array103'mi" to ptr addrspace(11)
; CHECK-NEXT:   store ptr addrspace(10) %"pred::Array'", ptr addrspace(11) %"'ipc", align 8
; CHECK-NEXT:   store i64 %value_phi49.size.sroa.3.0.copyload, ptr addrspace(11) %"'ipc", align 8
; CHECK-NEXT:   br label %inverttop
; CHECK: inverttop:
; CHECK-NEXT:   ret void
