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

; CHECK-LABEL: define void @caller(double %arg, { [2 x double], double, i64 }* %sret_box) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { i8*, { [2 x double], double, i64 } }, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { i8*, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 } }* %stack_sret, i32 0, i32 1
; CHECK-NEXT:   call void @full_write_callee({ i8*, { [2 x double], double, i64 } }* sret({ i8*, { [2 x double], double, i64 } }) %stack_sret, double %arg)
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 } }* %stack_sret, i32 0, i32 0
; CHECK-NEXT:   %res = load i8*, i8** %1, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 0, i32 0
; CHECK-NEXT:   %3 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %4 = load double, double* %3, align 8
; CHECK-NEXT:   store double %4, double* %2, align 8
; CHECK-NEXT:   %5 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 0, i32 1
; CHECK-NEXT:   %6 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 1
; CHECK-NEXT:   %7 = load double, double* %6, align 8
; CHECK-NEXT:   store double %7, double* %5, align 8
; CHECK-NEXT:   %8 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 1
; CHECK-NEXT:   %9 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 1
; CHECK-NEXT:   %10 = load double, double* %9, align 8
; CHECK-NEXT:   store double %10, double* %8, align 8
; CHECK-NEXT:   %11 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 2
; CHECK-NEXT:   %12 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 2
; CHECK-NEXT:   %13 = load i64, i64* %12, align 4
; CHECK-NEXT:   store i64 %13, i64* %11, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
