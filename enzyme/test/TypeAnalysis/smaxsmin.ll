; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=smax -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=smaxsmin -S -o /dev/null | FileCheck %s

declare void @_mlir_memref_to_llvm_free(ptr)

declare ptr @_mlir_memref_to_llvm_alloc(i64)

declare i64 @llvm.smin.i64(i64, i64)

declare i32 @llvm.smax.i32(i32, i32)

define internal void @smaxsmin(ptr %0) {
entry:
  %1 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = and i64 %3, -64
  %5 = inttoptr i64 %4 to ptr
  %6 = load i32, ptr %5, align 64
  %7 = tail call i32 @llvm.smax.i32(i32 %6, i32 0)
  %8 = zext nneg i32 %7 to i64
  %9 = tail call i64 @llvm.smin.i64(i64 %8, i64 3)
  %10 = getelementptr double, ptr %0, i64 %9
  %11 = load double, ptr %10, align 8
  ret void
}

; CHECK-NOT: %7 = tail call i32 @llvm.smax.i32(i32 %6, i32 0): {}
; CHECK-NOT: %9 = tail call i64 @llvm.smin.i64(i64 %8, i64 3): {}