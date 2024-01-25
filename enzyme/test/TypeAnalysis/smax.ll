; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -print-type-analysis -type-analysis-func=smax -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="print-type-analysis" -type-analysis-func=smax -S -o /dev/null | FileCheck %s

define void @smax(ptr %inp) {
entry:
  %0 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72)
  %1 = ptrtoint ptr %0 to i64
  %2 = add i64 %1, 63
  %3 = and i64 %2, -64
  %4 = inttoptr i64 %3 to ptr
  %5 = load i32, ptr %4, align 64
  %6 = tail call i32 @llvm.smax.i32(i32 %5, i32 0)
  %7 = getelementptr double, ptr %inp, i32 %6
  %8 = load double, ptr %7, align 8
  ret void
}

declare ptr @_mlir_memref_to_llvm_alloc(i64)

declare i32 @llvm.smax.i32(i32, i32)


; CHECK: smax - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: ptr %inp: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %0 = tail call ptr @_mlir_memref_to_llvm_alloc(i64 72): {}
; CHECK-NEXT:   %1 = ptrtoint ptr %0 to i64: {}
; CHECK-NEXT:   %2 = add i64 %1, 63: {}
; CHECK-NEXT:   %3 = and i64 %2, -64: {[-1]:Pointer}
; CHECK-NEXT:   %4 = inttoptr i64 %3 to ptr: {[-1]:Pointer}
; CHECK-NEXT:   %5 = load i32, ptr %4, align 64: {}
; CHECK-NEXT:   %6 = tail call i32 @llvm.smax.i32(i32 %5, i32 0): {[-1]:Integer}
; CHECK-NEXT:   %7 = getelementptr double, ptr %inp, i32 %6: {[-1]:Pointer}
; CHECK-NEXT:   %8 = load double, ptr %7, align 8: {}
; CHECK-NEXT:   ret void: {}