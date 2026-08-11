; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

define void @caller(double %arg, { [2 x double], double, i64 }* %sret_box, { [2 x double], double, i64 }* %sret_box_prime) {
; CHECK-LABEL: define void @caller(double %arg, { [2 x double], double, i64 }* %sret_box, { [2 x double], double, i64 }* %sret_box_prime) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* %stack_sret, i32 0, i32 2
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* %stack_sret, i32 0, i32 1
; CHECK-NEXT:   %2 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %3 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 0, i32 0
; CHECK-NEXT:   %4 = load double, double* %3, align 8
; CHECK-NEXT:   store double %4, double* %2, align 8
; CHECK-NEXT:   %5 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 1
; CHECK-NEXT:   %6 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 0, i32 1
; CHECK-NEXT:   %7 = load double, double* %6, align 8
; CHECK-NEXT:   store double %7, double* %5, align 8
; CHECK-NEXT:   %8 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 1
; CHECK-NEXT:   %9 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 1
; CHECK-NEXT:   %10 = load double, double* %9, align 8
; CHECK-NEXT:   store double %10, double* %8, align 8
; CHECK-NEXT:   %11 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 2
; CHECK-NEXT:   %12 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 2
; CHECK-NEXT:   %13 = load i64, i64* %12, align 4
; CHECK-NEXT:   store i64 %13, i64* %11, align 4
; CHECK-NEXT:   call void @augmented_julia_rec_3119({ i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* sret({ i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }) %stack_sret, double %arg)
; CHECK-NEXT:   %14 = getelementptr inbounds { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* %stack_sret, i32 0, i32 0
; CHECK-NEXT:   %res = load i8*, i8** %14, align 8
; CHECK-NEXT:   %15 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 0, i32 0
; CHECK-NEXT:   %16 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %1, i64 0, i32 0, i32 0
; CHECK-NEXT:   %17 = load double, double* %16, align 8
; CHECK-NEXT:   store double %17, double* %15, align 8
; CHECK-NEXT:   %18 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 0, i32 1
; CHECK-NEXT:   %19 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %1, i64 0, i32 0, i32 1
; CHECK-NEXT:   %20 = load double, double* %19, align 8
; CHECK-NEXT:   store double %20, double* %18, align 8
; CHECK-NEXT:   %21 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 1
; CHECK-NEXT:   %22 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %1, i64 0, i32 1
; CHECK-NEXT:   %23 = load double, double* %22, align 8
; CHECK-NEXT:   store double %23, double* %21, align 8
; CHECK-NEXT:   %24 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box, i64 0, i32 2
; CHECK-NEXT:   %25 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %1, i64 0, i32 2
; CHECK-NEXT:   %26 = load i64, i64* %25, align 4
; CHECK-NEXT:   store i64 %26, i64* %24, align 4
; CHECK-NEXT:   %27 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 0, i32 0
; CHECK-NEXT:   %28 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %29 = load double, double* %28, align 8
; CHECK-NEXT:   store double %29, double* %27, align 8
; CHECK-NEXT:   %30 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 0, i32 1
; CHECK-NEXT:   %31 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i32 1
; CHECK-NEXT:   %32 = load double, double* %31, align 8
; CHECK-NEXT:   store double %32, double* %30, align 8
; CHECK-NEXT:   %33 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 1
; CHECK-NEXT:   %34 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 1
; CHECK-NEXT:   %35 = load double, double* %34, align 8
; CHECK-NEXT:   store double %35, double* %33, align 8
; CHECK-NEXT:   %36 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %sret_box_prime, i64 0, i32 2
; CHECK-NEXT:   %37 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 2
; CHECK-NEXT:   %38 = load i64, i64* %37, align 4
; CHECK-NEXT:   store i64 %38, i64* %36, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
entry:
  %res = call i8* @augmented_julia_rec_3119({ [2 x double], double, i64 }* "enzyme_sret"="test_type4" %sret_box, { [2 x double], double, i64 }* "enzyme_sret"="test_type4" %sret_box_prime, double %arg)
  ret void
}

define internal fastcc i8* @augmented_julia_rec_3119({ [2 x double], double, i64 }* noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(32) "enzyme_sret"="test_type4" %0, { [2 x double], double, i64 }* nocapture nofree align 8 "enzyme_sret"="test_type4" %"'", double %1) {
; CHECK-LABEL: define internal fastcc void @augmented_julia_rec_3119({ i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* noalias sret({ i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }) %0, double %1) {
; CHECK-NEXT: top:
; CHECK-NEXT:   %2 = getelementptr inbounds { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* %0, i32 0, i32 2
; CHECK-NEXT:   %3 = getelementptr inbounds { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* %0, i32 0, i32 1
; CHECK-NEXT:   %4 = alloca i8*, align 8
; CHECK-NEXT:   %5 = alloca { [1 x double], double, i64 }, i64 1, align 8
; CHECK-NEXT:   %6 = bitcast { [1 x double], double, i64 }* %5 to i8*
; CHECK-NEXT:   %"'mi" = call noalias nonnull dereferenceable(24) i8* @malloc(i64 24)
; CHECK-NEXT:   store i8* %"'mi", i8** %4, align 8
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(24) dereferenceable_or_null(24) %"'mi", i8 0, i64 24, i1 false)
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"'mi" to { [1 x double], double, i64 }*
; CHECK-NEXT:   %7 = bitcast i8* %6 to { [1 x double], double, i64 }*
; CHECK-NEXT:   call fastcc void @augmented_julia_rec_3276({ [1 x double], double, i64 }* nocapture nofree writeonly align 8 "enzyme_sret"="test_type5" %7, { [1 x double], double, i64 }* nocapture nofree align 8 "enzyme_sret"="test_type5" %"'ipc", double %1)
; CHECK-NEXT:   %memcpy_refined_src4 = getelementptr inbounds { [1 x double], double, i64 }, { [1 x double], double, i64 }* %7, i64 0, i32 2
; CHECK-NEXT:   %memcpy_refined_src = getelementptr inbounds { [1 x double], double, i64 }, { [1 x double], double, i64 }* %7, i64 0, i32 0, i64 0
; CHECK-NEXT:   %8 = load double, double* %memcpy_refined_src, align 8
; CHECK-NEXT:   %9 = load i64, i64* %memcpy_refined_src4, align 8
; CHECK-NEXT:   %newstruct3.sroa.0.0..sroa_idx = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %3, i64 0, i32 0, i64 0
; CHECK-NEXT:   store double %1, double* %newstruct3.sroa.0.0..sroa_idx, align 8
; CHECK-NEXT:   %newstruct3.sroa.2.0..sroa_idx6 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %3, i64 0, i32 0, i64 1
; CHECK-NEXT:   store double %8, double* %newstruct3.sroa.2.0..sroa_idx6, align 8
; CHECK-NEXT:   %newstruct3.sroa.3.0..sroa_idx7 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %3, i64 0, i32 1
; CHECK-NEXT:   store double %1, double* %newstruct3.sroa.3.0..sroa_idx7, align 8
; CHECK-NEXT:   %"newstruct3.sroa.4.0..sroa_idx8'ipg" = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %2, i64 0, i32 2
; CHECK-NEXT:   %newstruct3.sroa.4.0..sroa_idx8 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %3, i64 0, i32 2
; CHECK-NEXT:   store i64 %9, i64* %"newstruct3.sroa.4.0..sroa_idx8'ipg", align 8
; CHECK-NEXT:   store i64 %9, i64* %newstruct3.sroa.4.0..sroa_idx8, align 8
; CHECK-NEXT:   %10 = load i8*, i8** %4, align 8
; CHECK-NEXT:   %11 = getelementptr inbounds { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, { i8*, { [2 x double], double, i64 }, { [2 x double], double, i64 } }* %0, i32 0, i32 0
; CHECK-NEXT:   store i8* %10, i8** %11, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
top:
  %2 = alloca i8*, align 8
  %3 = alloca { [1 x double], double, i64 }, i64 1, align 8
  %4 = bitcast { [1 x double], double, i64 }* %3 to i8*
  %"'mi" = call noalias nonnull dereferenceable(24) i8* @malloc(i64 24)
  store i8* %"'mi", i8** %2, align 8
  call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(24) dereferenceable_or_null(24) %"'mi", i8 0, i64 24, i1 false)
  %"'ipc" = bitcast i8* %"'mi" to { [1 x double], double, i64 }*
  %5 = bitcast i8* %4 to { [1 x double], double, i64 }*
  call fastcc void @augmented_julia_rec_3276({ [1 x double], double, i64 }* nocapture nofree writeonly align 8 "enzyme_sret"="test_type5" %5, { [1 x double], double, i64 }* nocapture nofree align 8 "enzyme_sret"="test_type5" %"'ipc", double %1)
  %memcpy_refined_src4 = getelementptr inbounds { [1 x double], double, i64 }, { [1 x double], double, i64 }* %5, i64 0, i32 2
  %memcpy_refined_src = getelementptr inbounds { [1 x double], double, i64 }, { [1 x double], double, i64 }* %5, i64 0, i32 0, i64 0
  %6 = load double, double* %memcpy_refined_src, align 8
  %7 = load i64, i64* %memcpy_refined_src4, align 8
  %newstruct3.sroa.0.0..sroa_idx = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i64 0
  store double %1, double* %newstruct3.sroa.0.0..sroa_idx, align 8
  %newstruct3.sroa.2.0..sroa_idx6 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i64 1
  store double %6, double* %newstruct3.sroa.2.0..sroa_idx6, align 8
  %newstruct3.sroa.3.0..sroa_idx7 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 1
  store double %1, double* %newstruct3.sroa.3.0..sroa_idx7, align 8
  %"newstruct3.sroa.4.0..sroa_idx8'ipg" = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %"'", i64 0, i32 2
  %newstruct3.sroa.4.0..sroa_idx8 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 2
  store i64 %7, i64* %"newstruct3.sroa.4.0..sroa_idx8'ipg", align 8
  store i64 %7, i64* %newstruct3.sroa.4.0..sroa_idx8, align 8
  %8 = load i8*, i8** %2, align 8
  ret i8* %8
}

declare i8* @malloc(i64)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1 immarg)
declare void @augmented_julia_rec_3276({ [1 x double], double, i64 }*, { [1 x double], double, i64 }*, double)
