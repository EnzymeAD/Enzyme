; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

define void @caller(double %arg, { [2 x double], double, i64 }* %sret_box, { [2 x double], double, i64 }* %sret_box_prime) {
; CHECK-LABEL: define void @caller(double %arg, { [2 x double], double, i64 }* %sret_box, { [2 x double], double, i64 }* %sret_box_prime) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { i8*, { [2 x double], double, i64 } }, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { i8*, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 } }* %stack_sret, i32 0, i32 1
; CHECK-NEXT:   %1 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %2 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 0, i32 0
; CHECK-NEXT:   %3 = load double, double* %2, align 8
; CHECK-NEXT:   store double %3, double* %1, align 8
; CHECK-NEXT:   %4 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 1
; CHECK-NEXT:   %5 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 0, i32 1
; CHECK-NEXT:   %6 = load double, double* %5, align 8
; CHECK-NEXT:   store double %6, double* %4, align 8
; CHECK-NEXT:   %7 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 1
; CHECK-NEXT:   %8 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 1
; CHECK-NEXT:   %9 = load double, double* %8, align 8
; CHECK-NEXT:   store double %9, double* %7, align 8
; CHECK-NEXT:   %10 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 2
; CHECK-NEXT:   %11 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 2
; CHECK-NEXT:   %12 = load i64, i64* %11, align 4
; CHECK-NEXT:   store i64 %12, i64* %10, align 4
; CHECK-NEXT:   call void @readonly_callee({ i8*, { [2 x double], double, i64 } }* sret({ i8*, { [2 x double], double, i64 } }) %stack_sret, double %arg)
; CHECK-NEXT:   %13 = getelementptr inbounds { i8*, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 } }* %stack_sret, i32 0, i32 0
; CHECK-NEXT:   %res = load i8*, i8** %13, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
entry:
  %res = call i8* @readonly_callee({ [2 x double], double, i64 }* readonly "enzyme_sret"="test_type4" %sret_box, double %arg)
  ret void
}

define internal fastcc i8* @readonly_callee({ [2 x double], double, i64 }* noalias nocapture nofree noundef nonnull readonly align 8 dereferenceable(32) "enzyme_sret"="test_type4" %0, double %1) {
top:
  %gep = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i32 0, i32 1
  %val = load double, double* %gep, align 8
  ret i8* null
}
