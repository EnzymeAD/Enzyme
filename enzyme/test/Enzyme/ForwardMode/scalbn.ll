; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @scalbn(double, i32)
declare double @__enzyme_fwddiff(i8*, ...)

define double @test(double %x, i32 %exp) {
entry:
  %call = call double @scalbn(double %x, i32 %exp)
  ret double %call
}

define double @dtest(double %x, double %dx, i32 %exp) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double, i32)* @test to i8*), double %x, double %dx, i32 %exp)
  ret double %call
}


; CHECK: define internal double @fwddiffetest(double %x, double %"x'", i32 %exp)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[v:.+]] = call fast double @scalbn(double %"x'", i32 %exp)
; CHECK-NEXT:   ret double %[[v]]
; CHECK-NEXT: }
