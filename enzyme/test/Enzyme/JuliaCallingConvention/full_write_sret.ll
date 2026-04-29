; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

define void @caller(double %arg, { [2 x double], double, i64 }* %sret_box) {
entry:
  %res = call i8* @full_write_callee({ [2 x double], double, i64 }* writeonly "enzyme_sret"="test_type4" %sret_box, double %arg)
  ret void
}

define internal fastcc i8* @full_write_callee({ [2 x double], double, i64 }* noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(32) "enzyme_sret"="test_type4" %0, double %1) {
top:
  store { [2 x double], double, i64 } { [2 x double] [double 1.0, double 2.0], double 3.0, i64 4 }, { [2 x double], double, i64 }* %0, align 8
  ret i8* null
}

; CHECK-LABEL: define void @caller(double %arg, ptr %sret_box) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { ptr, { [2 x double], double, i64 } }, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { ptr, { [2 x double], double, i64 } }, ptr %stack_sret, i32 0, i32 1
; CHECK-NEXT:   call void @full_write_callee(ptr sret({ ptr, { [2 x double], double, i64 } }) %stack_sret, double %arg)
; CHECK-NEXT:   %1 = getelementptr inbounds { ptr, { [2 x double], double, i64 } }, ptr %stack_sret, i32 0, i32 0
; CHECK-NEXT:   %res = load ptr, ptr %1, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 0, i32 0
; CHECK-NEXT:   %3 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %4 = load double, ptr %3, align 8
; CHECK-NEXT:   store double %4, ptr %2, align 8
; CHECK-NEXT:   %5 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 0, i32 1
; CHECK-NEXT:   %6 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 0, i32 1
; CHECK-NEXT:   %7 = load double, ptr %6, align 8
; CHECK-NEXT:   store double %7, ptr %5, align 8
; CHECK-NEXT:   %8 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 1
; CHECK-NEXT:   %9 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 1
; CHECK-NEXT:   %10 = load double, ptr %9, align 8
; CHECK-NEXT:   store double %10, ptr %8, align 8
; CHECK-NEXT:   %11 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 2
; CHECK-NEXT:   %12 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 2
; CHECK-NEXT:   %13 = load i64, ptr %12, align 4
; CHECK-NEXT:   store i64 %13, ptr %11, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
