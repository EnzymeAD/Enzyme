; RUN: not %opt < %s %loadEnzyme %enzyme -S 2>&1 | FileCheck %s
;
; Pointer-width integer operands are not declared Fortran references. Reject
; them explicitly instead of treating an arbitrary integer as an address.
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"

declare void @dgetrs_(i8*, i64, i32*, double*, i32*, i32*, double*, i32*, i32*)
declare void @__enzyme_fwddiff(...)

define void @solve(double* %rhs) {
  %n = alloca i32
  store i32 2, i32* %n
  call void @dgetrs_(i8* null, i64 2, i32* %n, double* null,
                    i32* %n, i32* null, double* %rhs, i32* %n, i32* %n)
  ret void
}

define void @differentiate(double* %rhs, double* %drhs) {
  call void (...) @__enzyme_fwddiff(void (double*)* @solve,
                                   double* %rhs, double* %drhs)
  ret void
}

; CHECK: getrs requires compatible Fortran by-reference BLAS/LAPACK declarations
