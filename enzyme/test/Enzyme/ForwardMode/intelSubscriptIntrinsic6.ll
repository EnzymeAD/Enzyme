; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

define float @selectfirst(ptr noalias nocapture readonly %X, i32 %n) local_unnamed_addr {
entry:
  %n_ext = sext i32 %n to i64
  %"X[]" = tail call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr nonnull elementtype(float) %X, i64 1, i64 %n_ext)
  %val = load float, ptr %"X[]", align 4
  ret float %val
}

declare ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8, i64, i64, ptr, i64, i64)

declare float @__enzyme_fwddiff(...)
declare [2 x float] @__enzyme_fwddiff_batch(...)

define float @fwd_selectfirst(ptr %X, ptr %DX, i32 %n) local_unnamed_addr {
entry:
  %r = call float (...) @__enzyme_fwddiff(ptr @selectfirst, ptr %X, ptr %DX, i32 %n)
  ret float %r
}

define [2 x float] @batch_selectfirst(ptr %X, ptr %DX1, ptr %DX2, i32 %n) local_unnamed_addr {
entry:
  %r = call [2 x float] (...) @__enzyme_fwddiff_batch(ptr @selectfirst, metadata !"enzyme_width", i64 2, ptr %X, ptr %DX1, ptr %DX2, i32 %n)
  ret [2 x float] %r
}

; CHECK-LABEL: define internal float @fwddiffeselectfirst(ptr noalias nocapture readonly %X, ptr nocapture %"X'", i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %n_ext = sext i32 %n to i64
; CHECK-NEXT:   %0 = call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr elementtype(float) %"X'", i64 1, i64 %n_ext)
; CHECK-NEXT:   %"X[]" = tail call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr nonnull elementtype(float) %X, i64 1, i64 %n_ext)
; CHECK-NEXT:   %"val'ipl" = load float, ptr %0, align 4
; CHECK-NEXT:   ret float %"val'ipl"

; CHECK-LABEL: define internal [2 x float] @fwddiffe2selectfirst(ptr noalias nocapture readonly %X, [2 x ptr] %"X'", i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %n_ext = sext i32 %n to i64
; CHECK-NEXT:   %0 = extractvalue [2 x ptr] %"X'", 0
; CHECK-NEXT:   %1 = call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr elementtype(float) %0, i64 1, i64 %n_ext)
; CHECK-NEXT:   %2 = extractvalue [2 x ptr] %"X'", 1
; CHECK-NEXT:   %3 = call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr elementtype(float) %2, i64 1, i64 %n_ext)
; CHECK-NEXT:   %"X[]" = tail call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr nonnull elementtype(float) %X, i64 1, i64 %n_ext)
; CHECK-NEXT:   %"val'ipl" = load float, ptr %1, align 4
; CHECK-NEXT:   %4 = insertvalue [2 x float] undef, float %"val'ipl", 0
; CHECK-NEXT:   %"val'ipl1" = load float, ptr %3, align 4
; CHECK-NEXT:   %5 = insertvalue [2 x float] %4, float %"val'ipl1", 1
; CHECK-NEXT:   ret [2 x float] %5
