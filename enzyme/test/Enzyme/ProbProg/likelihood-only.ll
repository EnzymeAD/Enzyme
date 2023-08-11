; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare double @normal(double, double)
declare double @exp(double)
declare double @log(double)

define double @normal_logpdf(double %mean, double %var, double %x) {
  %i = fdiv double 1.000000e+00, %var
  %i3 = fmul double %i, 0x40040D931FF62705
  %i4 = fdiv double %mean, %var
  %i5 = fsub double %x, %i4
  %i6 = fmul double %i5, %i5
  %i7 = fmul double %i6, -5.000000e-01
  %i8 = tail call double @exp(double %i7)
  %i9 = fmul double %i3, %i8
  %i10 = tail call double @log(double %i9)
  ret double %i10
}

@.str = private constant [3 x i8] c"mu\00"
@.str.1 = private constant [2 x i8] c"x\00"

@enzyme_duplikelihood = global i32 0

declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare void @__enzyme_likelihood(void ()*, i32, double*, double*)

define void @test() {
entry:
  %mu = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.0, double 1.0)
  %x = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %mu, double 1.0)
  ret void
}

define double @generate() {
entry:
  %0 = load i32, i32* @enzyme_duplikelihood
  %likelihood = alloca double
  %dlikelihood = alloca double
  store double 1.0, double* %dlikelihood
  tail call void @__enzyme_likelihood(void ()* @test, i32 %0, double* %likelihood, double* %dlikelihood)
  %res = load double, double* %likelihood
  ret double %res
}

; CHECK: define internal void @diffelikelihood_test(double* "enzyme_likelihood" %likelihood, double* %"likelihood'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.mu = call fast double @augmented_normal_logpdf.2(double 0.000000e+00, double 1.000000e+00, double %0)
; CHECK-NEXT:   %log_prob_sum = load double, double* %likelihood
; CHECK-NEXT:   %1 = fadd double %log_prob_sum, %likelihood.mu
; CHECK-NEXT:   store double %1, double* %likelihood
; CHECK-NEXT:   %2 = call double @normal(double %0, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.x = call fast double @augmented_normal_logpdf(double %0, double 1.000000e+00, double %2)
; CHECK-NEXT:   %log_prob_sum1 = load double, double* %likelihood
; CHECK-NEXT:   %3 = fadd double %log_prob_sum1, %likelihood.x
; CHECK-NEXT:   store double %3, double* %likelihood
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %4 = load double, double* %"likelihood'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"likelihood'"
; CHECK-NEXT:   %5 = load double, double* %"likelihood'"
; CHECK-NEXT:   %6 = fadd fast double %5, %4
; CHECK-NEXT:   store double %6, double* %"likelihood'"
; CHECK-NEXT:   %7 = call { double, double } @diffenormal_logpdf(double %0, double 1.000000e+00, double %2, double %4)
; CHECK-NEXT:   %8 = load double, double* %"likelihood'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"likelihood'"
; CHECK-NEXT:   %9 = load double, double* %"likelihood'"
; CHECK-NEXT:   %10 = fadd fast double %9, %8
; CHECK-NEXT:   store double %10, double* %"likelihood'"
; CHECK-NEXT:   %11 = call { double } @diffenormal_logpdf.3(double 0.000000e+00, double 1.000000e+00, double %0, double %8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }