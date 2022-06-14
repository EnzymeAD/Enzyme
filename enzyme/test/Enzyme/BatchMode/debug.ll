; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx12.0.0"

%struct.Vector = type { double, double, double, double }

@enzyme_width = external global i32, align 4
@enzyme_vector = external global i32, align 4
@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: noinline nounwind optnone ssp uwtable
define double @_Z6squared(double %0) #0 {
  %2 = alloca double, align 8
  store double %0, double* %2, align 8
  %3 = load double, double* %2, align 8
  %4 = load double, double* %2, align 8
  %5 = fmul double %3, %4
  ret double %5
}

; Function Attrs: noinline optnone ssp uwtable
define %struct.Vector @_Z7dsquaredddd(double %0, double %1, double %2, double %3) #1 {
  %5 = alloca %struct.Vector, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  %8 = alloca double, align 8
  %9 = alloca double, align 8
  store double %0, double* %6, align 8
  store double %1, double* %7, align 8
  store double %2, double* %8, align 8
  store double %3, double* %9, align 8
  %10 = load i32, i32* @enzyme_width, align 4
  %11 = load i32, i32* @enzyme_vector, align 4
  %12 = load double, double* %6, align 8
  %13 = load double, double* %7, align 8
  %14 = load double, double* %8, align 8
  %15 = load double, double* %9, align 8
  %16 = call %struct.Vector (i8*, ...) @__enzyme_batch(i8* bitcast (double (double)* @_Z6squared to i8*), i32 %10, i32 4, i32 %11, double %12, double %13, double %14, double %15)
  %17 = getelementptr inbounds %struct.Vector, %struct.Vector* %5, i32 0, i32 0
  %18 = extractvalue %struct.Vector %16, 0
  store double %18, double* %17, align 8
  %19 = getelementptr inbounds %struct.Vector, %struct.Vector* %5, i32 0, i32 1
  %20 = extractvalue %struct.Vector %16, 1
  store double %20, double* %19, align 8
  %21 = getelementptr inbounds %struct.Vector, %struct.Vector* %5, i32 0, i32 2
  %22 = extractvalue %struct.Vector %16, 2
  store double %22, double* %21, align 8
  %23 = getelementptr inbounds %struct.Vector, %struct.Vector* %5, i32 0, i32 3
  %24 = extractvalue %struct.Vector %16, 3
  store double %24, double* %23, align 8
  %25 = load %struct.Vector, %struct.Vector* %5, align 8
  ret %struct.Vector %25
}

declare %struct.Vector @__enzyme_batch(i8*, ...) #2

; Function Attrs: noinline norecurse optnone ssp uwtable
define i32 @main() #3 {
  %1 = alloca %struct.Vector, align 8
  %2 = call %struct.Vector @_Z7dsquaredddd(double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00)
  %3 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 0
  %4 = extractvalue %struct.Vector %2, 0
  store double %4, double* %3, align 8
  %5 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 1
  %6 = extractvalue %struct.Vector %2, 1
  store double %6, double* %5, align 8
  %7 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 2
  %8 = extractvalue %struct.Vector %2, 2
  store double %8, double* %7, align 8
  %9 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 3
  %10 = extractvalue %struct.Vector %2, 3
  store double %10, double* %9, align 8
  %11 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 0
  %12 = load double, double* %11, align 8
  %13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), double %12)
  %14 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 1
  %15 = load double, double* %14, align 8
  %16 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), double %15)
  %17 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 2
  %18 = load double, double* %17, align 8
  %19 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), double %18)
  %20 = getelementptr inbounds %struct.Vector, %struct.Vector* %1, i32 0, i32 3
  %21 = load double, double* %20, align 8
  %22 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), double %21)
  ret i32 0
}

declare i32 @printf(i8*, ...) #2

attributes #0 = { noinline nounwind optnone ssp uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.5a,+zcm,+zcz" }
attributes #1 = { noinline optnone ssp uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.5a,+zcm,+zcz" }
attributes #2 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.5a,+zcm,+zcz" }
attributes #3 = { noinline norecurse optnone ssp uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.5a,+zcm,+zcz" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 3]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, !"branch-target-enforcement", i32 0}
!3 = !{i32 1, !"sign-return-address", i32 0}
!4 = !{i32 1, !"sign-return-address-all", i32 0}
!5 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 1}
!8 = !{i32 7, !"frame-pointer", i32 1}
!9 = !{!"Apple clang version 13.1.6 (clang-1316.0.21.2.5)"}


; CHECK: define internal [4 x double] @batch__Z6squared([4 x double] %0)
; CHECK-NEXT:   %unwrap = extractvalue [4 x double] %0, 0
; CHECK-NEXT:   %unwrap1 = extractvalue [4 x double] %0, 1
; CHECK-NEXT:   %unwrap2 = extractvalue [4 x double] %0, 2
; CHECK-NEXT:   %unwrap3 = extractvalue [4 x double] %0, 3
; CHECK-NEXT:   %2 = alloca double, align 8
; CHECK-NEXT:   store double %unwrap, double* %2, align 8
; CHECK-NEXT:   store double %unwrap3, double* %2, align 8
; CHECK-NEXT:   store double %unwrap2, double* %2, align 8
; CHECK-NEXT:   store double %unwrap1, double* %2, align 8
; CHECK-NEXT:   %3 = load double, double* %2, align 8
; CHECK-NEXT:   %4 = load double, double* %2, align 8
; CHECK-NEXT:   %5 = load double, double* %2, align 8
; CHECK-NEXT:   %6 = load double, double* %2, align 8
; CHECK-NEXT:   %7 = load double, double* %2, align 8
; CHECK-NEXT:   %8 = load double, double* %2, align 8
; CHECK-NEXT:   %9 = load double, double* %2, align 8
; CHECK-NEXT:   %10 = load double, double* %2, align 8
; CHECK-NEXT:   %11 = fmul double %3, %7
; CHECK-NEXT:   %12 = fmul double %4, %8
; CHECK-NEXT:   %13 = fmul double %5, %9
; CHECK-NEXT:   %14 = fmul double %6, %10
; CHECK-NEXT:   %mrv = insertvalue [4 x double] undef, double %11, 0
; CHECK-NEXT:   %mrv4 = insertvalue [4 x double] %mrv, double %12, 1
; CHECK-NEXT:   %mrv5 = insertvalue [4 x double] %mrv4, double %13, 2
; CHECK-NEXT:   %mrv6 = insertvalue [4 x double] %mrv5, double %14, 3
; CHECK-NEXT:   ret [4 x double] %mrv6
; CHECK-NEXT: }