; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

define dso_local noalias nonnull double* @_Z6toHeapd(double %x) {
entry:
  %call = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to double*
  store double %x, double* %0, align 8
  ret double* %0
}

declare dso_local nonnull i8* @_Znwm(i64)

define dso_local double @_Z6squared(double %x) {
entry:
  %call = call double* @_Z6toHeapd(double %x)
  %0 = load double, double* %call, align 8
  %mul = fmul double %0, %x
  ret double %mul
}

define dso_local double @_Z7dsquared(double %x) {
entry:
  %call = call double (...) @_Z16__enzyme_fwddiffz(i8* bitcast (double (double)* @_Z6squared to i8*), double %x, double 1.000000e+00)
  ret double %call
}

declare dso_local double @_Z16__enzyme_fwddiffz(...)



; CHECK: define dso_local double @_Z7dsquared(double %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @fwddiffe_Z6squared(double %x, double 1.000000e+00)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffe_Z6squared(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { double*, double* } @fwddiffe_Z6toHeapd(double %x, double %"x'")
; CHECK-NEXT:   %[[i1:.+]] = extractvalue { double*, double* } %[[i0]], 0
; CHECK-NEXT:   %[[i2:.+]] = extractvalue { double*, double* } %[[i0]], 1
; CHECK-NEXT:   %[[i4:.+]] = load double, double* %[[i2]], align 8
; CHECK-NEXT:   %[[i3:.+]] = load double, double* %[[i1]], align 8
; CHECK-NEXT:   %[[i5:.+]] = fmul fast double %[[i4]], %x
; CHECK-NEXT:   %[[i6:.+]] = fmul fast double %"x'", %[[i3]]
; CHECK-NEXT:   %[[i7:.+]] = fadd fast double %[[i5]], %[[i6]]
; CHECK-NEXT:   ret double %[[i7]]
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @fwddiffe_Z6toHeapd(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   %0 = call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
; CHECK-NEXT:   %"'ipc" = bitcast i8* %0 to double*
; CHECK-NEXT:   %1 = bitcast i8* %call to double*
; CHECK-NEXT:   store double %x, double* %1, align 8
; CHECK-NEXT:   store double %"x'", double* %"'ipc", align 8
; CHECK-NEXT:   %2 = insertvalue { double*, double* } undef, double* %1, 0
; CHECK-NEXT:   %3 = insertvalue { double*, double* } %2, double* %"'ipc", 1
; CHECK-NEXT:   ret { double*, double* } %3
; CHECK-NEXT: }
