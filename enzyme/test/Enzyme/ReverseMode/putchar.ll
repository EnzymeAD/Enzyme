; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

declare i32 @putchar(i32) 

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %tmp = call i32 @putchar(i32 32)
  %0 = tail call fast double @llvm.exp.f64(double %x)
  ret double %0
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.exp.f64(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmp = call i32 @putchar(i32 32)
; CHECK-NEXT:   %0 = call fast double @llvm.exp.f64(double %x)
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %0
; CHECK-NEXT:   %2 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %2