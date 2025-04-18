; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

declare double @scalbn(double, i32)
declare double @__enzyme_autodiff(i8*, ...)

define double @test(double %x, i32 %exp) {
entry:
  %call = call double @scalbn(double %x, i32 %exp)
  ret double %call
}

define double @dtest(double %x, i32 %exp) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double, i32)* @test to i8*), double %x, i32 %exp)
  ret double %call
}


; CHECK: define internal { double } @diffetest(double %x, i32 %exp, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[v:.+]] = call fast double @scalbn(double %differeturn, i32 %exp)
; CHECK-NEXT:   %[[r4:.+]] = insertvalue { double } undef, double %[[v]], 0
; CHECK-NEXT:   ret { double } %[[r4]]
; CHECK-NEXT: }