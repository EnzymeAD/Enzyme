; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

declare double @__enzyme_autodiff(i8*, ...)
declare double @logb(double)

define double @test(double %x) {
entry:
  %call = call double @logb(double %x)
  ret double %call
}

define double @test_derivative(double %x) {
entry:
  %call = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @test to i8*), double %x)
  ret double %call
}


; CHECK: define internal { double } @diffetest(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }