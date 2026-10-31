; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Reduced from an AMDGPU HIP kernel body (MFEM dFEM forall). The body holds
; {ptr data, i64 size, [3 x i64] strides, ptr y}. Argument promotion moved the
; strides[0] load out of an accessor without its TBAA, so nothing types it,
; and the data access is a GEP that is not inbounds. The GEP scales its index
; by sizeof(double), so the index cannot be a pointer and is an integer.

declare void @__enzyme_fwddiff(...)

@enzyme_dup = global i32 0
@enzyme_const = global i32 0

define void @body(ptr %0, i32 %1) {
  %3 = load ptr, ptr %0, align 8, !tbaa !0
  %4 = sext i32 %1 to i64
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = load i64, ptr %5, align 8
  %7 = mul i64 %6, %4
  %8 = getelementptr double, ptr %3, i64 %7
  %9 = load double, ptr %8, align 8
  %10 = fmul double %9, %9
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %12 = load ptr, ptr %11, align 8, !tbaa !0
  %13 = getelementptr inbounds double, ptr %12, i64 %4
  store double %10, ptr %13, align 8
  ret void
}

define void @dbody(ptr %b, ptr %db, i32 %k) {
  call void (...) @__enzyme_fwddiff(ptr @body, ptr @enzyme_dup, ptr %b, ptr %db, ptr @enzyme_const, i32 %k)
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"p1 double", !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}

; CHECK: define internal void @fwddiffebody(ptr {{.*}}%0, ptr {{.*}}%"'", i32 %1)
; CHECK:   %7 = mul i64 %6, %4
; CHECK:   %"'ipg" = getelementptr double, ptr %"'ipl", i64 %7
; CHECK:   %[[dv:.+]] = load double, ptr %"'ipg", align 8
; CHECK:   store double %{{[0-9]+}}, ptr %"'ipg{{[0-9]+}}", align 8
