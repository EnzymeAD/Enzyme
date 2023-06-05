; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

define double* @z0(double* %a5) {
bb:
  %i = alloca i64, align 8
  %i1 = alloca i64, align 8
  %i2 = call i64 @a32(double* %a5)
  store i64 %i2, i64* %i, align 8
  %i6 = call i1 @a14(double* %a5, i64* nocapture readonly %i)
  store i64 %i2, i64* %i1, align 8
  call void @a25(double* %a5, i64* nocapture readonly %i1, i1 %i6)
  ret double* %a5
}

define i1 @a14(double*, i64*) {
entry:
  ret i1 false
}

define void @a25(double*, i64*, i1) {
entry:
  ret void
}

define i64 @a32(double* %i5) {
bb:
  br label %bb6

bb6:                                              ; preds = %bb6, %bb
  %i7 = phi i64 [ 0, %bb ], [ %i16, %bb6 ]
  %i16 = add i64 %i7, 1
  %i17 = icmp sgt i64 %i7, 10
  br i1 %i17, label %bb18, label %bb6

bb18:                                             ; preds = %bb6
  ret i64 %i7
}

declare void @__enzyme_augmentfwd(...)

define void @dsquare(double *%arg, double* %darg) {
bb:
  call void (...) @__enzyme_augmentfwd(double* (double*)* nonnull @z0, metadata !"enzyme_dup", double *%arg, double* %darg)
  ret void
}

; CHECK: define internal { i8*, double*, double* } @augmented_z0(double* %a5, double* %"a5'")
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = alloca { i8*, double*, double* }
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(12) dereferenceable_or_null(12) i8* @malloc(i64 12)
; CHECK-NEXT:   %tapemem = bitcast i8* %malloccall to { i1, i64 }*
; CHECK-NEXT:   %1 = getelementptr inbounds { i8*, double*, double* }, { i8*, double*, double* }* %0, i32 0, i32 0
; CHECK-NEXT:   store i8* %malloccall, i8** %1
; CHECK-NEXT:   %i1 = alloca i64, i64 1
; CHECK-NEXT:   %2 = bitcast i64* %i1 to i8*
; CHECK-NEXT:   %i = alloca i64, i64 1
; CHECK-NEXT:   %3 = bitcast i64* %i to i8*
; CHECK-NEXT:   %4 = bitcast i8* %3 to i64*, !enzyme_caststack !
; CHECK-NEXT:   %5 = bitcast i8* %2 to i64*, !enzyme_caststack !
; CHECK-NEXT:   %i2 = call i64 @augmented_a32(double* %a5, double* %"a5'")
; CHECK-NEXT:   %6 = getelementptr inbounds { i1, i64 }, { i1, i64 }* %tapemem, i32 0, i32 1
; CHECK-NEXT:   store i64 %i2, i64* %6
; CHECK-NEXT:   store i64 %i2, i64* %4
; CHECK-NEXT:   %i6 = call i1 @augmented_a14(double* %a5, double* %"a5'", i64* nocapture readonly %4)
; CHECK-NEXT:   %7 = getelementptr inbounds { i1, i64 }, { i1, i64 }* %tapemem, i32 0, i32 0
; CHECK-NEXT:   store i1 %i6, i1* %7
; CHECK-NEXT:   store i64 %i2, i64* %5
; CHECK-NEXT:   call void @augmented_a25(double* %a5, double* %"a5'", i64* nocapture readonly %5, i1 %i6)
; CHECK-NEXT:   %8 = insertvalue { i8*, double*, double* } undef, double* %a5, 1
; CHECK-NEXT:   %9 = getelementptr inbounds { i8*, double*, double* }, { i8*, double*, double* }* %0, i32 0, i32 1
; CHECK-NEXT:   store double* %a5, double** %9
; CHECK-NEXT:   %10 = getelementptr inbounds { i8*, double*, double* }, { i8*, double*, double* }* %0, i32 0, i32 2
; CHECK-NEXT:   store double* %"a5'", double** %10
; CHECK-NEXT:   %11 = load { i8*, double*, double* }, { i8*, double*, double* }* %0
; CHECK-NEXT:   ret { i8*, double*, double* } %11
; CHECK-NEXT: }

