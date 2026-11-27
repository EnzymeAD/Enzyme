; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=selectfirst -S | FileCheck %s; fi

define float @selectfirst(ptr noalias nocapture readonly %X, i32 %n) local_unnamed_addr {
entry:
  %n_ext = sext i32 %n to i64
  %"X[]" = tail call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr nonnull elementtype(float) %X, i64 1, i64 %n_ext)
  %val = load float, ptr %"X[]", align 4
  ret float %val
}

declare ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8, i64, i64, ptr, i64, i64)

; CHECK: selectfirst - {[-1]:Float@float} |{[-1]:Pointer, [-1,0]:Float@float}:{} {[-1]:Integer}:{}
; CHECK-NEXT: ptr %X: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT: i32 %n: {[-1]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %n_ext = sext i32 %n to i64: {[-1]:Integer}
; CHECK-NEXT:   %"X[]" = tail call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr nonnull elementtype(float) %X, i64 1, i64 %n_ext): {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %val = load float, ptr %"X[]", align 4: {[-1]:Float@float}
; CHECK-NEXT:   ret float %val: {}
