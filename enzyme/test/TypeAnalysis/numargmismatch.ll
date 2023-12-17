; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=_Z19kernel_main_wrappedPfS_S_ -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=_Z19kernel_main_wrappedPfS_S_ -S -o /dev/null | FileCheck %s

; ModuleID = '/app/example.ll'
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@enzyme_dup = dso_local global i32 0, align 4
@0 = private constant [3 x float] [float 2.000000e+00, float 2.000000e+00, float 2.000000e+00]

; Function Attrs: noinline optnone
define dso_local void @_Z19kernel_main_wrappedPfS_S_(float* noundef %0, float* noundef %1, float* noundef %2) #0 {
  %4 = alloca float*, align 8
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  store float* %0, float** %4, align 8
  store float* %1, float** %5, align 8
  store float* %2, float** %6, align 8
  %7 = load float*, float** %4, align 8
  %8 = load float*, float** %5, align 8
  %9 = call float* bitcast ({ float*, float*, i64 } (float*, float*, i64, i64, i64, float*, float*, i64, i64, i64)* @kernel_main to float* (float*, float*, i64, i64, i64)*)(float* noundef %7, float* noundef %8, i64 noundef 0, i64 noundef 0, i64 noundef 0)
  ret void
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znam(i64 noundef) #2

define { float*, float*, i64 } @kernel_main(float* %0, float* %1, i64 %2, i64 %3, i64 %4, float* %5, float* %6, i64 %7, i64 %8, i64 %9) {
  %11 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %0, 0
  %12 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %11, float* %1, 1
  %13 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %12, i64 %2, 2
  %14 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %13, i64 %3, 3, 0
  %15 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %14, i64 %4, 4, 0
  %16 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %5, 0
  %17 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %16, float* %6, 1
  %18 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %17, i64 %7, 2
  %19 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %18, i64 %8, 3, 0
  %20 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %19, i64 %9, 4, 0
  %21 = call i8* @malloc(i64 add (i64 mul (i64 ptrtoint (float* getelementptr (float, float* null, i32 1) to i64), i64 3), i64 128))
  %22 = bitcast i8* %21 to float*
  %23 = ptrtoint float* %22 to i64
  %24 = add i64 %23, 127
  %25 = urem i64 %24, 128
  %26 = sub i64 %24, %25
  %27 = inttoptr i64 %26 to float*
  %28 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %22, 0
  %29 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %28, float* %27, 1
  %30 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %29, i64 0, 2
  %31 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %30, i64 3, 3, 0
  %32 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %31, i64 1, 4, 0
  br label %33

33:                                               ; preds = %36, %10
  %34 = phi i64 [ %45, %36 ], [ 0, %10 ]
  %35 = icmp slt i64 %34, 3
  br i1 %35, label %36, label %46

36:                                               ; preds = %33
  %37 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %15, 1
  %38 = getelementptr float, float* %37, i64 %34
  %39 = load float, float* %38, align 4
  %40 = extractvalue { float*, float*, i64, [1 x i64], [1 x i64] } %20, 1
  %41 = getelementptr float, float* %40, i64 %34
  %42 = load float, float* %41, align 4
  %43 = fmul float %39, %42
  %44 = getelementptr float, float* %27, i64 %34
  store float %43, float* %44, align 4
  %45 = add i64 %34, 1
  br label %33

46:                                               ; preds = %33
  %47 = call i8* @malloc(i64 add (i64 mul (i64 ptrtoint (float* getelementptr (float, float* null, i32 1) to i64), i64 3), i64 128))
  %48 = bitcast i8* %47 to float*
  %49 = ptrtoint float* %48 to i64
  %50 = add i64 %49, 127
  %51 = urem i64 %50, 128
  %52 = sub i64 %50, %51
  %53 = inttoptr i64 %52 to float*
  %54 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %48, 0
  %55 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %54, float* %53, 1
  %56 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %55, i64 0, 2
  %57 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %56, i64 3, 3, 0
  %58 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %57, i64 1, 4, 0
  br label %59

59:                                               ; preds = %62, %46
  %60 = phi i64 [ %69, %62 ], [ 0, %46 ]
  %61 = icmp slt i64 %60, 3
  br i1 %61, label %62, label %70

62:                                               ; preds = %59
  %63 = getelementptr float, float* %27, i64 %60
  %64 = load float, float* %63, align 4
  %65 = getelementptr float, float* getelementptr inbounds ([3 x float], [3 x float]* @0, i64 0, i64 0), i64 %60
  %66 = load float, float* %65, align 4
  %67 = fadd float %64, %66
  %68 = getelementptr float, float* %53, i64 %60
  store float %67, float* %68, align 4
  %69 = add i64 %60, 1
  br label %59

70:                                               ; preds = %59
  call void @free(i8* %21)
  %71 = call i8* @malloc(i64 add (i64 ptrtoint (float* getelementptr (float, float* null, i64 1) to i64), i64 128))
  %72 = bitcast i8* %71 to float*
  %73 = ptrtoint float* %72 to i64
  %74 = add i64 %73, 127
  %75 = urem i64 %74, 128
  %76 = sub i64 %74, %75
  %77 = inttoptr i64 %76 to float*
  %78 = insertvalue { float*, float*, i64 } undef, float* %72, 0
  %79 = insertvalue { float*, float*, i64 } %78, float* %77, 1
  %80 = insertvalue { float*, float*, i64 } %79, i64 0, 2
  %81 = call i8* @malloc(i64 add (i64 ptrtoint (float* getelementptr (float, float* null, i64 1) to i64), i64 128))
  %82 = bitcast i8* %81 to float*
  %83 = ptrtoint float* %82 to i64
  %84 = add i64 %83, 127
  %85 = urem i64 %84, 128
  %86 = sub i64 %84, %85
  %87 = inttoptr i64 %86 to float*
  %88 = insertvalue { float*, float*, i64 } undef, float* %82, 0
  %89 = insertvalue { float*, float*, i64 } %88, float* %87, 1
  %90 = insertvalue { float*, float*, i64 } %89, i64 0, 2
  store float 0.000000e+00, float* %77, align 4
  call void @llvm.memcpy.p0f32.p0f32.i64(float* %87, float* %77, i64 ptrtoint (float* getelementptr (float, float* null, i64 1) to i64), i1 false)
  call void @free(i8* %71)
  br label %91

91:                                               ; preds = %94, %70
  %92 = phi i64 [ %99, %94 ], [ 0, %70 ]
  %93 = icmp slt i64 %92, 3
  br i1 %93, label %94, label %100

94:                                               ; preds = %91
  %95 = getelementptr float, float* %53, i64 %92
  %96 = load float, float* %95, align 4
  %97 = load float, float* %87, align 4
  %98 = fadd float %96, %97
  store float %98, float* %87, align 4
  %99 = add i64 %92, 1
  br label %91

100:                                              ; preds = %91
  call void @free(i8* %47)
  ret { float*, float*, i64 } %90
}

declare i8* @malloc(i64)

declare void @free(i8*)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0f32.p0f32.i64(float* noalias nocapture writeonly, float* noalias nocapture readonly, i64, i1 immarg) #4

attributes #0 = { noinline optnone }
attributes #1 = { noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { argmemonly nounwind willreturn }
attributes #5 = { builtin allocsize(0) }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5, !6}

!0 = !{!"clang version 15.0.0 (git@github.com:llvm/llvm-project.git e4a21e1644f2015dd4f9c3a7c67378879aa912cc)"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: _Z19kernel_main_wrappedPfS_S_ - {} |{[-1]:Pointer, [-1,-1]:Float@float}:{} {[-1]:Pointer, [-1,-1]:Float@float}:{} {[-1]:Pointer, [-1,-1]:Float@float}:{} 
; CHECK-NEXT: float* %0: {[-1]:Pointer, [-1,-1]:Float@float}
; CHECK-NEXT: float* %1: {[-1]:Pointer, [-1,-1]:Float@float}
; CHECK-NEXT: float* %2: {[-1]:Pointer, [-1,-1]:Float@float}
; CHECK:   %4 = alloca float*, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,0]:Float@float}
; CHECK-NEXT:   %5 = alloca float*, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,0]:Float@float}
; CHECK-NEXT:   %6 = alloca float*, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,0]:Float@float}
; CHECK-NEXT:   store float* %0, float** %4, align 8: {}
; CHECK-NEXT:   store float* %1, float** %5, align 8: {}
; CHECK-NEXT:   store float* %2, float** %6, align 8: {}
; CHECK-NEXT:   %7 = load float*, float** %4, align 8: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %8 = load float*, float** %5, align 8: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %9 = call float* bitcast ({ float*, float*, i64 } (float*, float*, i64, i64, i64, float*, float*, i64, i64, i64)* @kernel_main to float* (float*, float*, i64, i64, i64)*)(float* noundef %7, float* noundef %8, i64 noundef 0, i64 noundef 0, i64 noundef 0): {}
; CHECK-NEXT:   ret void: {}