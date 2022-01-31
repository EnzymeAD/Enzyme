;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %n, double* %x, double* %y) {
entry:
  tail call void @cblas_dswap(i32 %n, double* %x, i32 1, double* %y, i32 1)
  ret void
}

declare void @cblas_dswap(i32, double*, i32, double*, i32)

define void @caller(double* %x, double* %_x, double* %y, double* %_y) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i32, double*, double*)* @wrapper to i8*), i32 3, double* %x, double* %_x, double* %y, double* %_y)
  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

;CHECK:define internal void @diffewrapper(i32 %n, double* %x, double* %"x'", double* %y, double* %"y'") {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dswap(i32 %n, double* %x, i32 1, double* %y, i32 1)
;CHECK-NEXT:  call void @cblas_dswap(i32 %n, double* %"x'", i32 1, double* %"y'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}
