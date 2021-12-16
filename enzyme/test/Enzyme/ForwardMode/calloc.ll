; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -simplifycfg -S | FileCheck %s


@enzyme_dupnoneed = dso_local global i32 0, align 4

define dso_local double @f(double %x) {
entry:
  %x.addr = alloca double, align 8
  %data = alloca double*, align 8
  store double %x, double* %x.addr, align 8
  %call = call noalias i8* @calloc(i64 8, i64 1)
  %0 = bitcast i8* %call to double*
  store double* %0, double** %data, align 8
  %1 = load double, double* %x.addr, align 8
  %2 = load double*, double** %data, align 8
  %arrayidx = getelementptr inbounds double, double* %2, i64 0
  store double %1, double* %arrayidx, align 8
  %3 = load double*, double** %data, align 8
  %arrayidx1 = getelementptr inbounds double, double* %3, i64 0
  %4 = load double, double* %arrayidx1, align 8
  ret double %4
}

declare dso_local noalias i8* @calloc(i64, i64)

define dso_local double @df(double %x) {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load i32, i32* @enzyme_dupnoneed, align 4
  %1 = load double, double* %x.addr, align 8
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @f to i8*), i32 %0, double %1, double 1.000000e+00)
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(i8*, ...)


; CHECK: define internal double @fwddiffef(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call noalias i8* @calloc(i64 8, i64 1)
; CHECK-NEXT:   %0 = call noalias i8* @calloc(i64 8, i64 1)
; CHECK-NEXT:   %"'ipc" = bitcast i8* %0 to double*
; CHECK-NEXT:   %1 = bitcast i8* %call to double*
; CHECK-NEXT:   store double %x, double* %1, align 8
; CHECK-NEXT:   store double %"x'", double* %"'ipc", align 8
; CHECK-NEXT:   %2 = load double, double* %"'ipc", align 8
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }