; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

declare void @__enzyme_autodiff0(...) local_unnamed_addr

declare void @_mlir_memref_to_llvm_free(ptr)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define void @jit_compiled(ptr %a) {
  tail call void (...) @__enzyme_autodiff0(ptr nonnull @f, metadata !"enzyme_const", ptr %a, ptr %a, ptr %a, i64 0, metadata !"enzyme_const", ptr %a, metadata !"enzyme_const", ptr %a, i64 0, metadata !"enzyme_const", i64 1, metadata !"enzyme_const", ptr %a, metadata !"enzyme_dupnoneed", ptr %a, ptr %a, i64 0)
  ret void
}

define void @f(ptr %arg, ptr %arg1, i64 %arg2, ptr %arg3, ptr %arg4, i64 %arg5, i64 %arg6, ptr nocapture readnone %arg7, ptr nocapture writeonly %arg8, i64 %arg9) {
  %.idx = shl i64 %arg6, 3
  %i = tail call ptr @_mlir_memref_to_llvm_alloc(i64 %.idx)
  %i10 = load double, ptr %arg4, align 8
  %i11 = fcmp ogt double %i10, 1.500000e+00
  %i12 = load double, ptr %arg1, align 8
  %i13 = fmul double %i12, 2.000000e+00
  %storemerge = select i1 %i11, double %i13, double %i12
  store double %storemerge, ptr %i, align 8
  %i14 = alloca { ptr, ptr, i64 }, align 8
  store ptr %arg, ptr %i14, align 8
  %.repack2 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i14, i64 0, i32 1
  store ptr %arg1, ptr %.repack2, align 8
  %.repack4 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i14, i64 0, i32 2
  store i64 %arg2, ptr %.repack4, align 8
  %i15 = alloca { ptr, ptr, i64 }, align 8
  store ptr %arg3, ptr %i15, align 8
  %.repack6 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i15, i64 0, i32 1
  store ptr %arg4, ptr %.repack6, align 8
  %.repack8 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i15, i64 0, i32 2
  store i64 %arg5, ptr %.repack8, align 8
  %i16 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  store ptr %i, ptr %i16, align 8
  %.repack10 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 1
  store ptr %i, ptr %.repack10, align 8
  %.repack12 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 2
  store i64 0, ptr %.repack12, align 8
  %.repack14 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 3
  store i64 %arg6, ptr %.repack14, align 8
  %.repack16 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 4
  store i64 1, ptr %.repack16, align 8
  %i17 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
  %i18 = alloca { ptr, ptr, i64 }, align 8
  store ptr %i17, ptr %i18, align 8
  %.repack18 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i18, i64 0, i32 1
  store ptr %i17, ptr %.repack18, align 8
  %.repack20 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i18, i64 0, i32 2
  store i64 0, ptr %.repack20, align 8
  %i19 = load double, ptr %i17, align 8
  store double %i19, ptr %arg8, align 8
  ret void
}

; CHECK: define internal void @diffef(ptr %arg, ptr %arg1, ptr %"arg1'", i64 %arg2, ptr %arg3, ptr %arg4, i64 %arg5, i64 %arg6, ptr nocapture readnone %arg7, ptr nocapture writeonly %arg8, ptr nocapture %"arg8'", i64 %arg9)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %.idx = shl i64 %arg6, 3
; CHECK-NEXT:   %i = tail call ptr @_mlir_memref_to_llvm_alloc(i64 %.idx)
; CHECK-NEXT:   %i10 = load double, ptr %arg4, align 8
; CHECK-NEXT:   %i11 = fcmp ogt double %i10, 1.500000e+00
; CHECK-NEXT:   %i12 = load double, ptr %arg1, align 8
; CHECK-NEXT:   %i13 = fmul double %i12, 2.000000e+00
; CHECK-NEXT:   %storemerge = select i1 %i11, double %i13, double %i12
; CHECK-NEXT:   store double %storemerge, ptr %i, align 8
; CHECK-NEXT:   %i14 = alloca { ptr, ptr, i64 }, align 8
; CHECK-NEXT:   store ptr %arg, ptr %i14, align 8
; CHECK-NEXT:   %.repack2 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i14, i64 0, i32 1
; CHECK-NEXT:   store ptr %arg1, ptr %.repack2, align 8
; CHECK-NEXT:   %.repack4 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i14, i64 0, i32 2
; CHECK-NEXT:   store i64 %arg2, ptr %.repack4, align 8
; CHECK-NEXT:   %i15 = alloca { ptr, ptr, i64 }, align 8
; CHECK-NEXT:   store ptr %arg3, ptr %i15, align 8
; CHECK-NEXT:   %.repack6 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i15, i64 0, i32 1
; CHECK-NEXT:   store ptr %arg4, ptr %.repack6, align 8
; CHECK-NEXT:   %.repack8 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i15, i64 0, i32 2
; CHECK-NEXT:   store i64 %arg5, ptr %.repack8, align 8
; CHECK-NEXT:   %i16 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
; CHECK-NEXT:   store ptr %i, ptr %i16, align 8
; CHECK-NEXT:   %.repack10 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 1
; CHECK-NEXT:   store ptr %i, ptr %.repack10, align 8
; CHECK-NEXT:   %.repack12 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 2
; CHECK-NEXT:   store i64 0, ptr %.repack12, align 8
; CHECK-NEXT:   %.repack14 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 3
; CHECK-NEXT:   store i64 %arg6, ptr %.repack14, align 8
; CHECK-NEXT:   %.repack16 = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %i16, i64 0, i32 4
; CHECK-NEXT:   store i64 1, ptr %.repack16, align 8
; CHECK-NEXT:   %"i17'mi" = tail call noalias nonnull ptr @_mlir_memref_to_llvm_alloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull dereferenceable(8) dereferenceable_or_null(8) %"i17'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %i17 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 8)
; CHECK-NEXT:   %i18 = alloca { ptr, ptr, i64 }, align 8
; CHECK-NEXT:   store ptr %i17, ptr %i18, align 8
; CHECK-NEXT:   %.repack18 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i18, i64 0, i32 1
; CHECK-NEXT:   store ptr %i17, ptr %.repack18, align 8
; CHECK-NEXT:   %.repack20 = getelementptr inbounds { ptr, ptr, i64 }, ptr %i18, i64 0, i32 2
; CHECK-NEXT:   store i64 0, ptr %.repack20, align 8
; CHECK-NEXT:   %0 = load double, ptr %"arg8'", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"arg8'", align 8
; CHECK-NEXT:   %1 = load double, ptr %"i17'mi", align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %0
; CHECK-NEXT:   store double %2, ptr %"i17'mi", align 8
; CHECK-NEXT:   call void @_mlir_memref_to_llvm_free(ptr nonnull %"i17'mi")
; CHECK-NEXT:   call void @_mlir_memref_to_llvm_free(ptr %i17)
; CHECK-NEXT:   call void @_mlir_memref_to_llvm_free(ptr %i)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
