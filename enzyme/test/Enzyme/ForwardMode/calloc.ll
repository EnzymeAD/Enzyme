; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

@enzyme_dupnoneed = local_unnamed_addr global i32 0, align 4

define double @_Z1fd(double returned %x) {
entry:
  ret double %x
}

define double @_Z2dfd(double %x){
entry:
  %0 = load i32, i32* @enzyme_dupnoneed, align 4
  %call = call double (i8*, ...) @_Z16__enzyme_fwddiffPvz(i8* bitcast (double (double)* @_Z1fd to i8*), i32 %0, double %x, double 1.000000e+00)
  ret double %call
}

declare double @_Z16__enzyme_fwddiffPvz(i8*, ...)


; CHECK: define internal double @fwddiffe_Z1fd(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double %"x'"
; CHECK-NEXT: }