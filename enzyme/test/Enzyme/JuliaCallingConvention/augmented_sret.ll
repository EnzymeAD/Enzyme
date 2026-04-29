; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

define void @caller(double %arg, { [2 x double], double, i64 }* %sret_box, { [2 x double], double, i64 }* %sret_box_prime) {
; CHECK-LABEL: define void @caller(double %arg, ptr %sret_box, ptr %sret_box_prime) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %stack_sret = alloca { ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, ptr %stack_sret, i32 0, i32 1
; CHECK-NEXT:   %1 = getelementptr inbounds { ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, ptr %stack_sret, i32 0, i32 2
; CHECK-NEXT:   %2 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 0, i32 0
; CHECK-NEXT:   %3 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 0, i32 0
; CHECK-NEXT:   %4 = load double, ptr %3, align 8
; CHECK-NEXT:   store double %4, ptr %2, align 8
; CHECK-NEXT:   %5 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 0, i32 1
; CHECK-NEXT:   %6 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 0, i32 1
; CHECK-NEXT:   %7 = load double, ptr %6, align 8
; CHECK-NEXT:   store double %7, ptr %5, align 8
; CHECK-NEXT:   %8 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 1
; CHECK-NEXT:   %9 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 1
; CHECK-NEXT:   %10 = load double, ptr %9, align 8
; CHECK-NEXT:   store double %10, ptr %8, align 8
; CHECK-NEXT:   %11 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 2
; CHECK-NEXT:   %12 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 2
; CHECK-NEXT:   %13 = load i64, ptr %12, align 4
; CHECK-NEXT:   store i64 %13, ptr %11, align 4
; CHECK-NEXT:   call void @augmented_julia_rec_3119(ptr sret({ ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }) %stack_sret, double %arg)
; CHECK-NEXT:   %14 = getelementptr inbounds { ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, ptr %stack_sret, i32 0, i32 0
; CHECK-NEXT:   %res = load ptr, ptr %14, align 8
; CHECK-NEXT:   %15 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 0, i32 0
; CHECK-NEXT:   %16 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   %17 = load double, ptr %16, align 8
; CHECK-NEXT:   store double %17, ptr %15, align 8
; CHECK-NEXT:   %18 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 0, i32 1
; CHECK-NEXT:   %19 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 0, i32 1
; CHECK-NEXT:   %20 = load double, ptr %19, align 8
; CHECK-NEXT:   store double %20, ptr %18, align 8
; CHECK-NEXT:   %21 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 1
; CHECK-NEXT:   %22 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 1
; CHECK-NEXT:   %23 = load double, ptr %22, align 8
; CHECK-NEXT:   store double %23, ptr %21, align 8
; CHECK-NEXT:   %24 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box, i64 0, i32 2
; CHECK-NEXT:   %25 = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i64 0, i32 2
; CHECK-NEXT:   %26 = load i64, ptr %25, align 4
; CHECK-NEXT:   store i64 %26, ptr %24, align 4
; CHECK-NEXT:   %27 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 0, i32 0
; CHECK-NEXT:   %28 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 0, i32 0
; CHECK-NEXT:   %29 = load double, ptr %28, align 8
; CHECK-NEXT:   store double %29, ptr %27, align 8
; CHECK-NEXT:   %30 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 0, i32 1
; CHECK-NEXT:   %31 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 0, i32 1
; CHECK-NEXT:   %32 = load double, ptr %31, align 8
; CHECK-NEXT:   store double %32, ptr %30, align 8
; CHECK-NEXT:   %33 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 1
; CHECK-NEXT:   %34 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 1
; CHECK-NEXT:   %35 = load double, ptr %34, align 8
; CHECK-NEXT:   store double %35, ptr %33, align 8
; CHECK-NEXT:   %36 = getelementptr inbounds { [2 x double], double, i64 }, ptr %sret_box_prime, i64 0, i32 2
; CHECK-NEXT:   %37 = getelementptr inbounds { [2 x double], double, i64 }, ptr %1, i64 0, i32 2
; CHECK-NEXT:   %38 = load i64, ptr %37, align 4
; CHECK-NEXT:   store i64 %38, ptr %36, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
entry:
  %res = call i8* @augmented_julia_rec_3119({ [2 x double], double, i64 }* "enzyme_sret"="test_type4" %sret_box, { [2 x double], double, i64 }* "enzyme_sret"="test_type4" %sret_box_prime, double %arg)
  ret void
}

define internal fastcc i8* @augmented_julia_rec_3119({ [2 x double], double, i64 }* noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(32) "enzyme_sret"="test_type4" %0, { [2 x double], double, i64 }* nocapture nofree align 8 "enzyme_sret"="test_type4" %"'", double %1) {
; CHECK-LABEL: define internal fastcc void @augmented_julia_rec_3119(ptr noalias sret({ ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }) %0, double %1) {
; CHECK-NEXT: top:
; CHECK-NEXT:   %2 = getelementptr inbounds { ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, ptr %0, i32 0, i32 2
; CHECK-NEXT:   %3 = getelementptr inbounds { ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, ptr %0, i32 0, i32 1
; CHECK-NEXT:   %4 = alloca ptr, align 8
; CHECK-NEXT:   %5 = alloca { [1 x double], double, i64 }, i64 1, align 8
; CHECK-NEXT:   %6 = bitcast ptr %5 to ptr
; CHECK-NEXT:   %"'mi" = call noalias nonnull dereferenceable(24) ptr @malloc(i64 24)
; CHECK-NEXT:   store ptr %"'mi", ptr %4, align 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull dereferenceable(24) dereferenceable_or_null(24) %"'mi", i8 0, i64 24, i1 false)
; CHECK-NEXT:   %"'ipc" = bitcast ptr %"'mi" to ptr
; CHECK-NEXT:   %7 = bitcast ptr %6 to ptr
; CHECK-NEXT:   call fastcc void @augmented_julia_rec_3276(ptr nocapture nofree writeonly align 8 "enzyme_sret"="test_type5" %7, ptr nocapture nofree align 8 "enzyme_sret"="test_type5" %"'ipc", double %1)
; CHECK-NEXT:   %memcpy_refined_src4 = getelementptr inbounds { [1 x double], double, i64 }, ptr %7, i64 0, i32 2
; CHECK-NEXT:   %memcpy_refined_src = getelementptr inbounds { [1 x double], double, i64 }, ptr %7, i64 0, i32 0, i64 0
; CHECK-NEXT:   %8 = load double, ptr %memcpy_refined_src, align 8
; CHECK-NEXT:   %9 = load i64, ptr %memcpy_refined_src4, align 8
; CHECK-NEXT:   %newstruct3.sroa.0.0..sroa_idx = getelementptr inbounds { [2 x double], double, i64 }, ptr %3, i64 0, i32 0, i64 0
; CHECK-NEXT:   store double %1, ptr %newstruct3.sroa.0.0..sroa_idx, align 8
; CHECK-NEXT:   %newstruct3.sroa.2.0..sroa_idx6 = getelementptr inbounds { [2 x double], double, i64 }, ptr %3, i64 0, i32 0, i64 1
; CHECK-NEXT:   store double %8, ptr %newstruct3.sroa.2.0..sroa_idx6, align 8
; CHECK-NEXT:   %newstruct3.sroa.3.0..sroa_idx7 = getelementptr inbounds { [2 x double], double, i64 }, ptr %3, i64 0, i32 1
; CHECK-NEXT:   store double %1, ptr %newstruct3.sroa.3.0..sroa_idx7, align 8
; CHECK-NEXT:   %"newstruct3.sroa.4.0..sroa_idx8'ipg" = getelementptr inbounds { [2 x double], double, i64 }, ptr %2, i64 0, i32 2
; CHECK-NEXT:   %newstruct3.sroa.4.0..sroa_idx8 = getelementptr inbounds { [2 x double], double, i64 }, ptr %3, i64 0, i32 2
; CHECK-NEXT:   store i64 %9, ptr %"newstruct3.sroa.4.0..sroa_idx8'ipg", align 8
; CHECK-NEXT:   store i64 %9, ptr %newstruct3.sroa.4.0..sroa_idx8, align 8
; CHECK-NEXT:   %10 = load ptr, ptr %4, align 8
; CHECK-NEXT:   %11 = getelementptr inbounds { ptr, { [2 x double], double, i64 }, { [2 x double], double, i64 } }, ptr %0, i32 0, i32 0
; CHECK-NEXT:   store ptr %10, ptr %11, align 8
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
  %9 = load double, double* %memcpy_refined_src, align 8
  %10 = load i64, i64* %memcpy_refined_src4, align 8
  %newstruct3.sroa.0.0..sroa_idx = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i64 0
  store double %1, double* %newstruct3.sroa.0.0..sroa_idx, align 8
  %newstruct3.sroa.2.0..sroa_idx6 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 0, i64 1
  store double %9, double* %newstruct3.sroa.2.0..sroa_idx6, align 8
  %newstruct3.sroa.3.0..sroa_idx7 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 1
  store double %1, double* %newstruct3.sroa.3.0..sroa_idx7, align 8
  %"newstruct3.sroa.4.0..sroa_idx8'ipg" = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %"'", i64 0, i32 2
  %newstruct3.sroa.4.0..sroa_idx8 = getelementptr inbounds { [2 x double], double, i64 }, { [2 x double], double, i64 }* %0, i64 0, i32 2
  store i64 %10, i64* %"newstruct3.sroa.4.0..sroa_idx8'ipg", align 8
  store i64 %10, i64* %newstruct3.sroa.4.0..sroa_idx8, align 8
  %11 = load i8*, i8** %2, align 8
  ret i8* %11
}

declare i8* @malloc(i64)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1 immarg)
declare void @augmented_julia_rec_3276({ [1 x double], double, i64 }*, { [1 x double], double, i64 }*, double)
