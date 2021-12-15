; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -simplifycfg -S | FileCheck %s

@enzyme_dupnoneed = dso_local global i32 0, align 4

define dso_local double @_Z1fd(double returned %x) {
entry:
  ret double %x
}

define dso_local i32 @main() {
entry:
  %0 = load i32, i32* @enzyme_dupnoneed, align 4
  %call = call double (i8*, ...) @_Z16__enzyme_fwddiffPvz(i8* bitcast (double (double)* @_Z1fd to i8*), i32 %0, double 2.000000e+00, double 1.000000e+00)
  ret i32 0
}

declare dso_local double @_Z16__enzyme_fwddiffPvz(i8*, ...)


; CHECK: define internal double @fwddiffe_Z1fd(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }