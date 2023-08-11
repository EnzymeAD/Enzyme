; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

declare double @loadSparse(i64, i8*)

declare void @storeSparse(double, i64, i8*)

declare double* @__enzyme_todense(...)

define double @tester(i8* %data) {
entry:
  %ptr = call double* (...) @__enzyme_todense(double (i64, i8*)* @loadSparse, void (double, i64, i8*)* @storeSparse, i8* %data)
  %gep = getelementptr double, double* %ptr, i32 7
  %ld = load double, double* %gep
  store double %ld, double* %ptr
  ret double %ld
}

; CHECK: define double @tester(i8* %data) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep = getelementptr double, double* null, i32 7
; CHECK-NEXT:   %0 = call double @loadSparse(i64 56, i8* %data)
; CHECK-NEXT:   call void @storeSparse(double %0, i64 0, i8* %data)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }
