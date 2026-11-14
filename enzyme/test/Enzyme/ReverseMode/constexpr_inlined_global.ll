; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

@g = internal global [4 x double] zeroinitializer, align 16, !enzyme_type !0
@sym = external constant i8
@sym2 = external constant i8

declare double @__enzyme_autodiff(...)
declare [2 x double] @__enzyme_autodiff_vec(...)

define double @f_gep(double %x) {
entry:
  store double %x, ptr getelementptr inbounds ([4 x double], ptr @g, i64 0, i64 2), align 8
  %v = load double, ptr getelementptr inbounds ([4 x double], ptr @g, i64 0, i64 2), align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @f_gep_ceidx(double %x) {
entry:
  store double %x, ptr getelementptr (i8, ptr @g, i64 ptrtoint (ptr @sym to i64)), align 8
  %v = load double, ptr getelementptr (i8, ptr @g, i64 ptrtoint (ptr @sym to i64)), align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @f_add_rhs(double %x) {
entry:
  %p = inttoptr i64 add (i64 ptrtoint (ptr @g to i64), i64 8) to ptr
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @f_add_lhs(double %x) {
entry:
  %p = inttoptr i64 add (i64 sub (i64 ptrtoint (ptr @sym2 to i64), i64 ptrtoint (ptr @sym to i64)), i64 ptrtoint (ptr @g to i64)) to ptr
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @f_sub_rhs(double %x) {
entry:
  %p = inttoptr i64 sub (i64 ptrtoint (ptr @g to i64), i64 sub (i64 ptrtoint (ptr @sym2 to i64), i64 ptrtoint (ptr @sym to i64))) to ptr
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @test_gep(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f_gep, double %x)
  ret double %r
}

define [2 x double] @test_gep_vec(double %x) {
  %r = call [2 x double] (...) @__enzyme_autodiff_vec(ptr @f_gep, metadata !"enzyme_width", i64 2, double %x)
  ret [2 x double] %r
}

define double @test_gep_ceidx(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f_gep_ceidx, double %x)
  ret double %r
}

define double @test_add_rhs(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f_add_rhs, double %x)
  ret double %r
}

define double @test_add_lhs(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f_add_lhs, double %x)
  ret double %r
}

define double @test_sub_rhs(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f_sub_rhs, double %x)
  ret double %r
}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Pointer", i32 -1, !2}
!2 = !{!"Float@double"}

; CHECK: define internal { double } @diffef_gep(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"'ipg" = getelementptr inbounds [4 x double], ptr %"g'ipa", i64 0, i64 2

; CHECK: define internal { [2 x double] } @diffe2f_gep(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"g'ipa1" = alloca [4 x double], align 16
; CHECK:   %"'ipg" = getelementptr inbounds [4 x double], ptr %"g'ipa", i64 0, i64 2
; CHECK:   %"'ipg2" = getelementptr inbounds [4 x double], ptr %"g'ipa1", i64 0, i64 2

; CHECK: define internal { double } @diffef_gep_ceidx(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"'ipg" = getelementptr i8, ptr %"g'ipa", i64 ptrtoint (ptr @sym to i64)

; CHECK: define internal { double } @diffef_add_rhs(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"'ipc" = ptrtoint ptr %"g'ipa" to i64
; CHECK:   %"'ipa" = add i64 %"'ipc", 8

; CHECK: define internal { double } @diffef_add_lhs(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"'ipc" = ptrtoint ptr %"g'ipa" to i64
; CHECK:   %"'ipa" = add i64 sub (i64 ptrtoint (ptr @sym2 to i64), i64 ptrtoint (ptr @sym to i64)), %"'ipc"

; CHECK: define internal { double } @diffef_sub_rhs(
; CHECK:   %"g'ipa" = alloca [4 x double], align 16
; CHECK:   %"'ipc" = ptrtoint ptr %"g'ipa" to i64
; CHECK:   %"'ips" = sub i64 %"'ipc", sub (i64 ptrtoint (ptr @sym2 to i64), i64 ptrtoint (ptr @sym to i64))
