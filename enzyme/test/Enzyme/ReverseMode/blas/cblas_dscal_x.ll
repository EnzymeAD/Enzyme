;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(double* %x) {
entry:
  tail call void @cblas_dscal(i32 3, double 2.000000e+00, double* %x, i32 1)
  ret void
}

declare void @cblas_dscal(i32, double, double*, i32)

define void @caller(double* %x, double* %_x) {
entry:
  %call = tail call double @__enzyme_autodiff(i8* bitcast (void (double*)* @wrapper to i8*), double* %x, double* %_x)
  ret void
}

declare double @__enzyme_autodiff(i8*, double*, double*)

;CHECK:define internal void @diffewrapper(double* %x, double* %"x'") {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dscal(i32 3, double 2.000000e+00, double* %x, i32 1)
;CHECK-NEXT:  call void @cblas_dscal(i32 3, double 2.000000e+00, double* %"x'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
