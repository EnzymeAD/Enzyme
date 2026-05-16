; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes=enzyme -S -opaque-pointers | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-ni:10:11:12:13"
target triple = "arm64-apple-darwin24.0.0"

; Function Attrs: allocsize(1)
declare noalias ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64) #0

define fastcc ptr addrspace(10) @julia_mlogloss_core_3044(ptr addrspace(10) %"pred::Array", ptr %tl) {
top:
  %"pred::Array.size.sroa.0.0.copyload" = load i64, ptr %tl, align 8
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

; CHECK: define internal fastcc { ptr, ptr addrspace(10), ptr addrspace(10) } @augmented_julia_mlogloss_core_3044(ptr addrspace(10) %"pred::Array", ptr addrspace(10) %"pred::Array'", ptr nocapture readonly %tl, ptr nocapture %"tl'")
; CHECK-NEXT: top:
; CHECK-NEXT:   [[I0:%.+]] = alloca { ptr, ptr addrspace(10), ptr addrspace(10) }, align 8
; CHECK-NEXT:   [[MALLOC:%.+]] = tail call noalias nonnull dereferenceable(24) dereferenceable_or_null(24) ptr @malloc(i64 24)
; CHECK-NEXT:   [[I1:%.+]] = getelementptr inbounds { ptr, ptr addrspace(10), ptr addrspace(10) }, ptr [[I0]], i32 0, i32 0
; CHECK-NEXT:   store ptr [[MALLOC]], ptr [[I1]], align 8
; CHECK-NEXT:   [[SIZE:%.+]] = load i64, ptr %tl, align 8
; CHECK-NEXT:   [[TRUNC:%.+]] = trunc i64 [[SIZE]] to i32
; CHECK-NEXT:   [[GEP2:%.+]] = getelementptr inbounds { ptr addrspace(10), ptr addrspace(10), i32 }, ptr [[MALLOC]], i32 0, i32 2
; CHECK-NEXT:   store i32 [[TRUNC]], ptr [[GEP2]], align 4
; CHECK-NEXT:   [[ZEXT:%.+]] = zext i32 [[TRUNC]] to i64
; CHECK-NEXT:   [[SHADOW_MEM:%.+]] = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 [[ZEXT]])
; CHECK-NEXT:   [[GEP3:%.+]] = getelementptr inbounds { ptr addrspace(10), ptr addrspace(10), i32 }, ptr [[MALLOC]], i32 0, i32 0
; CHECK-NEXT:   store ptr addrspace(10) [[SHADOW_MEM]], ptr [[GEP3]], align 8
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull [[SHADOW_MEM]], i8 0, i64 [[ZEXT]], i1 false)
; CHECK-NEXT:   [[PRIMAL_MEM:%.+]] = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 [[ZEXT]])
; CHECK-NEXT:   [[GEP4:%.+]] = getelementptr inbounds { ptr addrspace(10), ptr addrspace(10), i32 }, ptr [[MALLOC]], i32 0, i32 1
; CHECK-NEXT:   store ptr addrspace(10) [[PRIMAL_MEM]], ptr [[GEP4]], align 8
; CHECK-NEXT:   [[NEW_ARRAY:%.+]] = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 1)
; CHECK-NEXT:   [[I7:%.+]] = addrspacecast ptr addrspace(10) [[NEW_ARRAY]] to ptr addrspace(11)
; CHECK-NEXT:   store ptr addrspace(10) [[PRIMAL_MEM]], ptr addrspace(11) [[I7]], align 8
; CHECK-NEXT:   [[GEP_SIZE:%.+]] = getelementptr i8, ptr addrspace(11) [[I7]], i64 32
; CHECK-NEXT:   [[VAL_PHI:%.+]] = load i64, ptr addrspace(11) [[GEP_SIZE]], align 8
; CHECK-NEXT:   [[ARRAY103_SHADOW:%.+]] = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 40)
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull dereferenceable(40) dereferenceable_or_null(40) [[ARRAY103_SHADOW]], i8 0, i64 40, i1 false)
; CHECK-NEXT:   [[ARRAY103_PRIMAL:%.+]] = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 40)
; CHECK-NEXT:   [[IPC:%.+]] = addrspacecast ptr addrspace(10) [[ARRAY103_SHADOW]] to ptr addrspace(11)
; CHECK-NEXT:   [[I8:%.+]] = addrspacecast ptr addrspace(10) [[ARRAY103_PRIMAL]] to ptr addrspace(11)
; CHECK-NEXT:   store ptr addrspace(10) %"pred::Array'", ptr addrspace(11) [[IPC]], align 8
; CHECK-NEXT:   store ptr addrspace(10) %"pred::Array", ptr addrspace(11) [[I8]], align 8
; CHECK-NEXT:   store i64 [[VAL_PHI]], ptr addrspace(11) [[IPC]], align 8
; CHECK-NEXT:   store i64 [[VAL_PHI]], ptr addrspace(11) [[I8]], align 8
; CHECK-NEXT:   [[IPL:%.+]] = load ptr addrspace(10), ptr addrspace(10) [[ARRAY103_SHADOW]], align 8
; CHECK-NEXT:   [[I9:%.+]] = load ptr addrspace(10), ptr addrspace(10) [[ARRAY103_PRIMAL]], align 8
; CHECK-NEXT:   [[I10:%.+]] = getelementptr inbounds { ptr, ptr addrspace(10), ptr addrspace(10) }, ptr [[I0]], i32 0, i32 1
; CHECK-NEXT:   store ptr addrspace(10) [[I9]], ptr [[I10]], align 8
; CHECK-NEXT:   [[I11:%.+]] = getelementptr inbounds { ptr, ptr addrspace(10), ptr addrspace(10) }, ptr [[I0]], i32 0, i32 2
; CHECK-NEXT:   store ptr addrspace(10) [[IPL]], ptr [[I11]], align 8
; CHECK-NEXT:   [[I12:%.+]] = load { ptr, ptr addrspace(10), ptr addrspace(10) }, ptr [[I0]], align 8
; CHECK-NEXT:   ret { ptr, ptr addrspace(10), ptr addrspace(10) } [[I12]]

; CHECK: define internal fastcc void @diffejulia_mlogloss_core_3044(ptr addrspace(10) %"pred::Array", ptr addrspace(10) %"pred::Array'", ptr nocapture readonly %tl, ptr nocapture %"tl'", ptr %tapeArg)
; CHECK-NEXT: top:
; CHECK-NEXT:   [[TRUETAPE:%.+]] = load { ptr addrspace(10), ptr addrspace(10), i32 }, ptr %tapeArg, align 8
; CHECK-NEXT:   tail call void @free(ptr nonnull %tapeArg)
; CHECK-NEXT:   [[E2:%.+]] = extractvalue { ptr addrspace(10), ptr addrspace(10), i32 } [[TRUETAPE]], 2
; CHECK-NEXT:   [[ZEXT2:%.+]] = zext i32 [[E2]] to i64
; CHECK-NEXT:   [[SHADOW_MEM2:%.+]] = extractvalue { ptr addrspace(10), ptr addrspace(10), i32 } [[TRUETAPE]], 0
; CHECK-NEXT:   [[PRIMAL_MEM2:%.+]] = extractvalue { ptr addrspace(10), ptr addrspace(10), i32 } [[TRUETAPE]], 1
; CHECK-NEXT:   [[NEW_ARRAY2:%.+]] = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 1)
; CHECK-NEXT:   [[I13:%.+]] = addrspacecast ptr addrspace(10) [[NEW_ARRAY2]] to ptr addrspace(11)
; CHECK-NEXT:   store ptr addrspace(10) [[PRIMAL_MEM2]], ptr addrspace(11) [[I13]], align 8
; CHECK-NEXT:   [[GEP_SIZE2:%.+]] = getelementptr i8, ptr addrspace(11) [[I13]], i64 32
; CHECK-NEXT:   [[VAL_PHI2:%.+]] = load i64, ptr addrspace(11) [[GEP_SIZE2]], align 8
; CHECK-NEXT:   [[ARRAY103_SHADOW2:%.+]] = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 40)
; CHECK-NEXT:   call void @llvm.memset.p10.i64(ptr addrspace(10) nonnull dereferenceable(40) dereferenceable_or_null(40) [[ARRAY103_SHADOW2]], i8 0, i64 40, i1 false)
; CHECK-NEXT:   [[IPC2:%.+]] = addrspacecast ptr addrspace(10) [[ARRAY103_SHADOW2]] to ptr addrspace(11)
; CHECK-NEXT:   store ptr addrspace(10) %"pred::Array'", ptr addrspace(11) [[IPC2]], align 8
; CHECK-NEXT:   store i64 [[VAL_PHI2]], ptr addrspace(11) [[IPC2]], align 8
; CHECK-NEXT:   br label %inverttop
; CHECK: inverttop:
; CHECK-NEXT:   ret void
