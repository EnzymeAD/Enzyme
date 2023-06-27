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

@enzyme_duptrace = global i32 0

declare i8* @__enzyme_newtrace()
declare void @__enzyme_freetrace(i8*)
declare i8* @__enzyme_get_trace(i8*, i8*)
declare i64 @__enzyme_get_choice(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_call(i8*, i8*, i8*)
declare void @__enzyme_insert_choice(i8*, i8*, double, i8*, i64)
declare void @__enzyme_insert_argument(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_return(i8*, i8*, i64)
declare void @__enzyme_insert_function(i8*, i8*)
declare void @__enzyme_insert_gradient_choice(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_gradient_argument(i8*, i8*, i8*, i64)
declare i1 @__enzyme_has_call(i8*, i8*)
declare i1 @__enzyme_has_choice(i8*, i8*)

declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare void @__enzyme_trace(void ()*, i32, i8*)

define void @test() {
entry:
  %mu = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.0, double 1.0)
  %x = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %mu, double 1.0)
  ret void
}

define i8* @generate() {
entry:
  %0 = load i32, i32* @enzyme_duptrace
  %trace = call i8* @__enzyme_newtrace()
  tail call void @__enzyme_trace(void ()* @test, i32 %0, i8* %trace)
  ret i8* %trace
}


; CHECK: define internal void @diffetrace_test(double* "enzyme_likelihood" %likelihood, double* %"likelihood'", i8* "enzyme_trace" %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @__enzyme_insert_function(i8* %trace, i8* bitcast (void (double*, i8*)* @trace_test to i8*))
; CHECK-NEXT:   %0 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.mu = call fast double @augmented_normal_logpdf.3(double 0.000000e+00, double 1.000000e+00, double %0)
; CHECK-NEXT:   %log_prob_sum = load double, double* %likelihood
; CHECK-NEXT:   %1 = fadd double %log_prob_sum, %likelihood.mu
; CHECK-NEXT:   store double %1, double* %likelihood
; CHECK-NEXT:   %2 = bitcast double %0 to i64
; CHECK-NEXT:   %3 = inttoptr i64 %2 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double %likelihood.mu, i8* %3, i64 8)
; CHECK-NEXT:   %4 = call double @normal(double %0, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.x = call fast double @augmented_normal_logpdf(double %0, double 1.000000e+00, double %4)
; CHECK-NEXT:   %log_prob_sum1 = load double, double* %likelihood
; CHECK-NEXT:   %5 = fadd double %log_prob_sum1, %likelihood.x
; CHECK-NEXT:   store double %5, double* %likelihood
; CHECK-NEXT:   %6 = bitcast double %4 to i64
; CHECK-NEXT:   %7 = inttoptr i64 %6 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.x, i8* %7, i64 8)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %8 = load double, double* %"likelihood'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"likelihood'"
; CHECK-NEXT:   %9 = load double, double* %"likelihood'"
; CHECK-NEXT:   %10 = fadd fast double %9, %8
; CHECK-NEXT:   store double %10, double* %"likelihood'"
; CHECK-NEXT:   %11 = call { double, double } @diffenormal_logpdf(double %0, double 1.000000e+00, double %4, double %8)
; CHECK-NEXT:   %12 = extractvalue { double, double } %11, 0
; CHECK-NEXT:   %13 = extractvalue { double, double } %11, 1
; CHECK-NEXT:   %14 = bitcast double %13 to i64
; CHECK-NEXT:   %15 = inttoptr i64 %14 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_gradient_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %15, i64 8)
; CHECK-NEXT:   %16 = load double, double* %"likelihood'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"likelihood'"
; CHECK-NEXT:   %17 = load double, double* %"likelihood'"
; CHECK-NEXT:   %18 = fadd fast double %17, %16
; CHECK-NEXT:   store double %18, double* %"likelihood'"
; CHECK-NEXT:   %19 = call { double } @diffenormal_logpdf.4(double 0.000000e+00, double 1.000000e+00, double %0, double %16)
; CHECK-NEXT:   %20 = extractvalue { double } %19, 0
; CHECK-NEXT:   %21 = fadd fast double %12, %20
; CHECK-NEXT:   %22 = bitcast double %21 to i64
; CHECK-NEXT:   %23 = inttoptr i64 %22 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_gradient_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %23, i64 8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }