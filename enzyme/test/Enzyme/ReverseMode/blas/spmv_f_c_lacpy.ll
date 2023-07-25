;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S | FileCheck %s

declare void @sspmv_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly) 

; 	character  	UPLO,
;		integer  	N,
;		real  	ALPHA,
;		real, dimension(*)  	AP,
;		real, dimension(*)  	X,
;		integer  	INCX,
;		real  	BETA,
;		real, dimension(*)  	Y,
;		integer  	INCY
;
define void @f(i8* noalias %AP, i8* noalias %X, i8* noalias %Y, i8* noalias %alpha, i8* noalias %beta) {
entry:
  %uplo = alloca i8, align 1
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %incx = alloca i64, align 16
  %incx_p = bitcast i64* %incx to i8*
  %incy = alloca i64, align 16
  %incy_p = bitcast i64* %incy to i8*
  ; 85 = U
  store i8 85, i8* %uplo, align 1
  store i64 4, i64* %n, align 16
  store i64 2, i64* %incx, align 16
  store i64 1, i64* %incy, align 16
  call void @sspmv_64_(i8* %uplo, i8* %n_p, i8* %alpha, i8* %AP, i8* %X, i8* %incx_p, i8* %beta, i8* %Y, i8* %incy_p) 
  %ptr = bitcast i8* %AP to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %AP, i8* %dAP, i8* %X, i8* %dX, i8* %Y, i8* %dY, i8* %alpha, i8* %dalpha, i8* %beta, i8* %dbeta) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %AP, i8* %dAP, metadata !"enzyme_dup", i8* %X, i8* %dX, metadata !"enzyme_dup", i8* %Y, i8* %dY, metadata !"enzyme_dup", i8* %alpha, i8* %dalpha, metadata !"enzyme_dup", i8* %beta, i8* %dbeta)
  ret void
}

