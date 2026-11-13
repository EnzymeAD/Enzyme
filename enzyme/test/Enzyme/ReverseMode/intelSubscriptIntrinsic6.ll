; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

define float @selectfirst(ptr noalias nocapture readonly %X, i32 %n) local_unnamed_addr {
entry:
  %n_ext = sext i32 %n to i64
  %"X[]" = tail call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr nonnull elementtype(float) %X, i64 1, i64 %n_ext)
  %val = load float, ptr %"X[]", align 4
  ret float %val
}

declare ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8, i64, i64, ptr, i64, i64)

declare void @__enzyme_autodiff(...)

define void @grad_selectfirst(ptr %X, ptr %DX, i32 %n) local_unnamed_addr {
entry:
  call void (...) @__enzyme_autodiff(ptr @selectfirst, ptr %X, ptr %DX, i32 %n)
  ret void
}

; CHECK-LABEL: define internal void @diffeselectfirst(ptr noalias nocapture readonly %X, ptr nocapture %"X'", i32 %n, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %n_ext = sext i32 %n to i64
; CHECK-NEXT:   %0 = call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr elementtype(float) %"X'", i64 1, i64 %n_ext)
; CHECK-NEXT:   %"X[]" = tail call ptr @llvm.intel.subscript.p0.i64.i64.p0.i64.i64(i8 0, i64 1, i64 4, ptr nonnull elementtype(float) %X, i64 1, i64 %n_ext)
; CHECK-NEXT:   %1 = load float, ptr %0, align 4
; CHECK-NEXT:   %2 = fadd fast float %1, %differeturn
; CHECK-NEXT:   store float %2, ptr %0, align 4
; CHECK-NEXT:   ret void
