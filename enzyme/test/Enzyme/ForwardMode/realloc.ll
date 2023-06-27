; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

define dso_local double* @f(double* %x, i64 %arg) {
entry:
  %call = call noalias double* @realloc(double* %x, i64 %arg)
  ret double* %call
}

declare dso_local noalias double* @realloc(double*, i64)

define dso_local double* @df(double* %x, double* %dx) {
entry:
  %call = call double* (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double* (double*,i64)* @f to i8*), double* %x, double* %dx, i64 3)
  ret double* %call
}

declare dso_local double* @__enzyme_fwddiff(i8*, ...)

; CHECK: define internal double* @fwddiffef(double* %x, double* %"x'", i64 %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call noalias double* @realloc(double* %x, i64 %arg)
; CHECK-NEXT:   %0 = call noalias double* @realloc(double* %"x'", i64 %arg)
; CHECK-NEXT:   ret double* %0
; CHECK-NEXT: }
