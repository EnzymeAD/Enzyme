; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -early-cse -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare double @fmod(double, double)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = call double @fmod(double %x, double %y)
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal { double, double } @diffetester(double %x, double %y, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = fadd double 0.000000e+00, %differeturn
; CHECK-NEXT:   %[[i1:.+]] = fdiv double %x, %y
; CHECK-NEXT:   %[[i2:.+]] = call double @llvm.fabs.f64(double %[[i1]])
; CHECK-NEXT:   %[[i3:.+]] = call double @llvm.floor.f64(double %[[i2]])
; CHECK-NEXT:   %[[i4:.+]] = call double @llvm.copysign.f64(double %[[i3]], double %[[i1]])
; CHECK-NEXT:   %[[i5:.+]] = {{(fsub double \-?0.000000e\+00,|fneg double)}} %[[i4]]
; CHECK-NEXT:   %[[i6:.+]] = fmul double %differeturn, %[[i5]]
; CHECK-NEXT:   %[[i7:.+]] = fadd double 0.000000e+00, %[[i6]]
; CHECK-NEXT:   %[[i8:.+]] = insertvalue { double, double } undef, double %[[i0]], 0
; CHECK-NEXT:   %[[i9:.+]] = insertvalue { double, double } %[[i8]], double %[[i7]], 1
; CHECK-NEXT:   ret { double, double } %[[i9]]
; CHECK-NEXT: }
