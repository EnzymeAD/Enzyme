; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

declare void @__enzyme_autodiff0(...) local_unnamed_addr

declare void @_mlir_memref_to_llvm_free(ptr)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

define void @jit_compiled(ptr %a) {
  tail call void (...) @__enzyme_autodiff0(ptr nonnull @f, ptr nonnull @enzyme_const, ptr %a, ptr %a, ptr %a, i64 0, ptr nonnull @enzyme_const, ptr %a, ptr nonnull @enzyme_const, ptr %a, i64 0, ptr nonnull @enzyme_const, i64 1, ptr nonnull @enzyme_const, ptr %a, ptr @enzyme_dupnoneed, ptr %a, ptr %a, i64 0)
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
