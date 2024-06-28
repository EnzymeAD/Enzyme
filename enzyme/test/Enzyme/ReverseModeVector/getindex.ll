; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

define "enzyme_type"="{[-1]:Float@double}" double @julia_getindex_3292_inner([2 x double] "enzyme_type"="{[-1]:Float@double}" %a0, i64 signext "enzyme_type"="{[-1]:Integer}" %a1) {
entry:
  %r2 = alloca [2 x double], align 8
  store [2 x double] %a0, [2 x double]* %r2, align 8
  %r8 = getelementptr inbounds [2 x double], [2 x double]* %r2, i64 0, i64 %a1
  %unbox.i = load double, double* %r8, align 8
  ret double %unbox.i
}

declare { [3 x [2 x double]] } @__enzyme_reverse(...) 

define { [3 x [2 x double]] } @dsquare([2 x double] %x) {
entry:
  %0 = tail call { [3 x [2 x double]] } (...) @__enzyme_reverse(double ([2 x double], i64)* nonnull @julia_getindex_3292_inner, metadata !"enzyme_width", i64 3, [2 x double] %x, i64 0, [3 x double] zeroinitializer, i8* null)
  ret { [3 x [2 x double]] } %0
}

; CHECK: define internal { [3 x [2 x double]] } @diffe3julia_getindex_3292_inner([2 x double] "enzyme_type"="{[-1]:Float@double}" %a0, i64 signext "enzyme_type"="{[-1]:Integer}" %a1, [3 x double] %differeturn, i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"a0'de" = alloca [3 x [2 x double]], align 8
; CHECK-NEXT:   store [3 x [2 x double]] zeroinitializer, [3 x [2 x double]]* %"a0'de", align 8
; CHECK-NEXT:   %"r2'ai" = alloca [2 x double], i64 1, align 8
; CHECK-NEXT:   %0 = bitcast [2 x double]* %"r2'ai" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(16) dereferenceable_or_null(16) %0, i8 0, i64 16, i1 false)
; CHECK-NEXT:   %"r2'ai7" = alloca [2 x double], i64 1, align 8
; CHECK-NEXT:   %1 = bitcast [2 x double]* %"r2'ai7" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(16) dereferenceable_or_null(16) %1, i8 0, i64 16, i1 false)
; CHECK-NEXT:   %"r2'ai8" = alloca [2 x double], i64 1, align 8
; CHECK-NEXT:   %2 = bitcast [2 x double]* %"r2'ai8" to i8*
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(16) dereferenceable_or_null(16) %2, i8 0, i64 16, i1 false)
; CHECK-NEXT:   %"r8'ipg" = getelementptr inbounds [2 x double], [2 x double]* %"r2'ai", i64 0, i64 %a1
; CHECK-NEXT:   %"r8'ipg3" = getelementptr inbounds [2 x double], [2 x double]* %"r2'ai7", i64 0, i64 %a1
; CHECK-NEXT:   %"r8'ipg4" = getelementptr inbounds [2 x double], [2 x double]* %"r2'ai8", i64 0, i64 %a1
; CHECK-NEXT:   %3 = extractvalue [3 x double] %differeturn, 0
; CHECK-NEXT:   %4 = load double, double* %"r8'ipg", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %3
; CHECK-NEXT:   store double %5, double* %"r8'ipg", align 8
; CHECK-NEXT:   %6 = extractvalue [3 x double] %differeturn, 1
; CHECK-NEXT:   %7 = load double, double* %"r8'ipg3", align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %6
; CHECK-NEXT:   store double %8, double* %"r8'ipg3", align 8
; CHECK-NEXT:   %9 = extractvalue [3 x double] %differeturn, 2
; CHECK-NEXT:   %10 = load double, double* %"r8'ipg4", align 8
; CHECK-NEXT:   %11 = fadd fast double %10, %9
; CHECK-NEXT:   store double %11, double* %"r8'ipg4", align 8
; CHECK-NEXT:   %12 = load [2 x double], [2 x double]* %"r2'ai", align 8
; CHECK-NEXT:   %13 = load [2 x double], [2 x double]* %"r2'ai7", align 8
; CHECK-NEXT:   %14 = load [2 x double], [2 x double]* %"r2'ai8", align 8
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"r2'ai", align 8
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"r2'ai7", align 8
; CHECK-NEXT:   store [2 x double] zeroinitializer, [2 x double]* %"r2'ai8", align 8
; CHECK-NEXT:   %15 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 0
; CHECK-NEXT:   %16 = extractvalue [2 x double] %12, 0
; CHECK-NEXT:   %17 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 0, i32 0
; CHECK-NEXT:   %18 = load double, double* %17, align 8
; CHECK-NEXT:   %19 = fadd fast double %18, %16
; CHECK-NEXT:   store double %19, double* %17, align 8
; CHECK-NEXT:   %20 = extractvalue [2 x double] %12, 1
; CHECK-NEXT:   %21 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 0, i32 1
; CHECK-NEXT:   %22 = load double, double* %21, align 8
; CHECK-NEXT:   %23 = fadd fast double %22, %20
; CHECK-NEXT:   store double %23, double* %21, align 8
; CHECK-NEXT:   %24 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 1
; CHECK-NEXT:   %25 = extractvalue [2 x double] %13, 0
; CHECK-NEXT:   %26 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 1, i32 0
; CHECK-NEXT:   %27 = load double, double* %26, align 8
; CHECK-NEXT:   %28 = fadd fast double %27, %25
; CHECK-NEXT:   store double %28, double* %26, align 8
; CHECK-NEXT:   %29 = extractvalue [2 x double] %13, 1
; CHECK-NEXT:   %30 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 1, i32 1
; CHECK-NEXT:   %31 = load double, double* %30, align 8
; CHECK-NEXT:   %32 = fadd fast double %31, %29
; CHECK-NEXT:   store double %32, double* %30, align 8
; CHECK-NEXT:   %33 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 2
; CHECK-NEXT:   %34 = extractvalue [2 x double] %14, 0
; CHECK-NEXT:   %35 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 2, i32 0
; CHECK-NEXT:   %36 = load double, double* %35, align 8
; CHECK-NEXT:   %37 = fadd fast double %36, %34
; CHECK-NEXT:   store double %37, double* %35, align 8
; CHECK-NEXT:   %38 = extractvalue [2 x double] %14, 1
; CHECK-NEXT:   %39 = getelementptr inbounds [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", i32 0, i32 2, i32 1
; CHECK-NEXT:   %40 = load double, double* %39, align 8
; CHECK-NEXT:   %41 = fadd fast double %40, %38
; CHECK-NEXT:   store double %41, double* %39, align 8
; CHECK-NEXT:   %42 = load [3 x [2 x double]], [3 x [2 x double]]* %"a0'de", align 8
; CHECK-NEXT:   %43 = insertvalue { [3 x [2 x double]] } undef, [3 x [2 x double]] %42, 0
; CHECK-NEXT:   ret { [3 x [2 x double]] } %43
; CHECK-NEXT: }

