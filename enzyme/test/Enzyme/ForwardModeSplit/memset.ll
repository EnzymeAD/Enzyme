; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_fwdsplit(...)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1)

define void @f(ptr nocapture %y) {
entry:
  call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 8, i1 false)
  %v = load double, ptr %y, align 8
  %v2 = fmul double %v, 2.0
  store double %v2, ptr %y, align 8
  ret void
}

define void @df(ptr %y, ptr %dy, ptr %tape) {
entry:
  %t = call ptr (...) @__enzyme_augmentfwd(ptr @f, metadata !"enzyme_dup", ptr %y, ptr %dy)
  call void (...) @__enzyme_fwdsplit(ptr @f, metadata !"enzyme_dup", ptr %y, ptr %dy, ptr %t)
  ret void
}

; CHECK: define internal void @fwddiffef(ptr nocapture %y, ptr nocapture %"y'", ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"y'", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %"v'ipl" = load double, ptr %"y'", align 8
; CHECK-NEXT:   %[[MUL:.+]] = fmul fast double %"v'ipl", 2.000000e+00
; CHECK-NEXT:   store double %[[MUL]], ptr %"y'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
