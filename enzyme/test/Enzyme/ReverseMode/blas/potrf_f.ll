;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

; dsyrk	(	character 	UPLO,
; character 	TRANS,
; integer 	N,
; integer 	K,
; double precision 	ALPHA,
; double precision, dimension(lda,*) 	A,
; integer 	LDA,
; double precision 	BETA,
; double precision, dimension(ldc,*) 	C,
; integer 	LDC
; )

declare void @dpotrf_64_(i8* nocapture readonly, i64* nocapture readonly, i8* nocapture readonly, i64* nocapture readonly, i8* nocapture, i64)

define void @f(i8* %A) {
entry:
  %info = alloca i64, align 1
  %info_p = bitcast i64* %info to i8*
  %uplo = alloca i8, align 1
  %n = alloca i64, align 16
  %lda = alloca i64, align 16
  store i8 85, i8* %uplo, align 1
  store i64 4, i64* %n, align 16
  store i64 4, i64* %lda, align 16
  call void @dpotrf_64_(i8* %uplo, i64* %n, i8* %A, i64* %lda, i8* %info_p, i64 1) 
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %A, i8* %dA) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*)* @f, metadata !"enzyme_dup", i8* %A, i8* %dA)
  ret void
}

; CHECK: define internal void @diffef(i8* %A, i8* %"A'")
; CHECK: entry:
; CHECK:   call void @dpotrf_64_(i8* %uplo, i64* %n, i8* %A, i64* %lda, i8* %info_p, i64 1) 

; CHECK: invertentry:
; CHECK:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK:   call void @llvm.memset.p0i8.i64(i8* %malloccall, i8 0, i64 %{{.*}}, i1 false)
