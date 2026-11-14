; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

@g = internal global [4 x double] zeroinitializer, align 16, !enzyme_type !0

declare double @__enzyme_fwddiff(...)

define double @f_gep(double %x) {
entry:
  store double %x, ptr getelementptr inbounds ([4 x double], ptr @g, i64 0, i64 2), align 8
  %v = load double, ptr getelementptr inbounds ([4 x double], ptr @g, i64 0, i64 2), align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @f_add(double %x) {
entry:
  %p = inttoptr i64 add (i64 ptrtoint (ptr @g to i64), i64 8) to ptr
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @test_gep(double %x, double %dx) {
  %r = call double (...) @__enzyme_fwddiff(ptr @f_gep, double %x, double %dx)
  ret double %r
}

define double @test_add(double %x, double %dx) {
  %r = call double (...) @__enzyme_fwddiff(ptr @f_add, double %x, double %dx)
  ret double %r
}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Pointer", i32 -1, !2}
!2 = !{!"Float@double"}

; CHECK: define internal double @fwddiffef_gep(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"'ipg" = getelementptr inbounds [4 x double], ptr %"g'ipa", i64 0, i64 2

; CHECK: define internal double @fwddiffef_add(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"'ipc" = ptrtoint ptr %"g'ipa" to i64
; CHECK:   %"'ipa" = add i64 %"'ipc", 8
