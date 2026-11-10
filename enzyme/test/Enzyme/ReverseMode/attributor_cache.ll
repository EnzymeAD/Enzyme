; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-attributor=1 -enzyme-postopt=1 -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-attributor=1 -enzyme-postopt=1 -passes="enzyme" -S | FileCheck %s

define void @stress_calculation(double %du_dx, double* nocapture writeonly align 8 %sigma) {
entry:
  store double %du_dx, double* %sigma, align 8
  ret void
}

define void @func1(double* nocapture writeonly align 8 %sigma, double* nocapture align 8 %dir) {
entry:
  %call = tail call double (i8*, double, double*, double*) @__enzyme_autodiff(i8* bitcast (void (double, double*)* @stress_calculation to i8*), double 2.000000e+00, double* %sigma, double* %dir)
  ret void
}

define void @func2(double* nocapture writeonly align 8 %sigma, double* nocapture align 8 %dir) {
entry:
  %call = tail call double (i8*, double, double*, double*) @__enzyme_autodiff(i8* bitcast (void (double, double*)* @stress_calculation to i8*), double 2.000000e+00, double* %sigma, double* %dir)
  ret void
}

declare double @__enzyme_autodiff(i8*, double, double*, double*)

; CHECK-LABEL: define void @func1(
; CHECK-SAME: align 8 dereferenceable(8) %sigma
; CHECK-SAME: align 8 dereferenceable(8) %dir
; CHECK: store double 2.000000e+00, {{double\*|ptr}} %sigma, align 8
; CHECK: store double 0.000000e+00, {{double\*|ptr}} %dir, align 8

; CHECK-LABEL: define void @func2(
; CHECK-SAME: align 8 dereferenceable(8) %sigma
; CHECK-SAME: align 8 dereferenceable(8) %dir
; CHECK: store double 2.000000e+00, {{double\*|ptr}} %sigma, align 8
; CHECK: store double 0.000000e+00, {{double\*|ptr}} %dir, align 8
