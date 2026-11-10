; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare void @cblas_dsymv(i32, i32, i32, double, ptr, i32, ptr, i32, double, ptr, i32)

declare void @__enzyme_autodiff(...)

define void @f(ptr %A, ptr %x, ptr %y) {
entry:
  call void @cblas_dsymv(i32 102, i32 121, i32 3, double 1.000000e+00, ptr %A, i32 3, ptr %x, i32 1, double 0.000000e+00, ptr %y, i32 1)
  ret void
}

define void @active(ptr %A, ptr %dA, ptr %x, ptr %y, ptr %dy) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %A, ptr %dA, metadata !"enzyme_const", ptr %x, metadata !"enzyme_dup", ptr %y, ptr %dy)
  ret void
}

; CHECK-LABEL: define internal void @diffef(ptr nocapture readonly %A, ptr nocapture %"A'", ptr nocapture readonly %x, ptr nocapture %y, ptr nocapture %"y'")
; CHECK: invertentry:
; CHECK: call void @cblas_dcopy(i32 3, ptr {{(nocapture )?}}{{(readonly )?}}%"A'", i32 4, ptr {{(nocapture )?}}%[[TMP:.+]], i32 1)
; CHECK-NEXT: call void @cblas_dsyr2(i32 102, i32 121, i32 3, double 1.000000e+00, ptr {{(nocapture )?}}{{(readonly )?}}%x, i32 1, ptr {{(nocapture )?}}{{(readonly )?}}%"y'", i32 1, ptr {{(nocapture )?}}%"A'", i32 3)
; CHECK-NEXT: call void @cblas_daxpy(i32 3, double -1.000000e+00, ptr {{(nocapture )?}}{{(readonly )?}}%"A'", i32 4, ptr {{(nocapture )?}}%[[TMP]], i32 1)
; CHECK-NEXT: call void @cblas_daxpy(i32 3, double 5.000000e-01, ptr {{(nocapture )?}}{{(readonly )?}}%[[TMP]], i32 1, ptr {{(nocapture )?}}%"A'", i32 4)
; CHECK: call void @cblas_dscal(i32 3, double 0.000000e+00, ptr {{(nocapture )?}}%"y'", i32 1)
