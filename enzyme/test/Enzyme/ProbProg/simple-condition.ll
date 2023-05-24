; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

@enzyme_observations = global i32 0
@enzyme_trace = global i32 0

@.str = private constant [3 x i8] c"mu\00"
@.str.1 = private constant [2 x i8] c"x\00"

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

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
declare void @__enzyme_condition(void ()*, i32, i8*, i32, i8*)

define void @test() {
entry:
  %mu = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.0, double 1.0)
  %x = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %mu, double 1.0)
  ret void
}

define i8* @generate() {
entry:
  %0 = load i32, i32* @enzyme_trace
  %trace = call i8* @__enzyme_newtrace()
  call void @__enzyme_trace(void ()* @test, i32 %0, i8* %trace)
  ret i8* %trace
}

define i8* @condition(i8* %observations) {
entry:
  %0 = load i32, i32* @enzyme_observations
  %1 = load i32, i32* @enzyme_trace
  %trace = call i8* @__enzyme_newtrace()
  call void @__enzyme_condition(void ()* @test, i32 %0, i8* %observations, i32 %1, i8* %trace)
  ret i8* %trace
}


; CHECK: define internal void @condition_test(double* "enzyme_likelihood" %likelihood, i8* "enzyme_observations" %observations, i8* "enzyme_trace" %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x.ptr.i = alloca double
; CHECK-NEXT:   %mu.ptr.i = alloca double
; CHECK-NEXT:   call void @__enzyme_insert_function(i8* %trace, i8* bitcast (void (double*, i8*, i8*)* @condition_test to i8*))
; CHECK-NEXT:   %0 = bitcast double* %mu.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %0)
; CHECK-NEXT:   %has.choice.mu.i = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.mu.i, label %condition.mu.with.trace.i, label %condition.mu.without.trace.i

; CHECK: condition.mu.with.trace.i:                        ; preds = %entry
; CHECK-NEXT:   %1 = bitcast double* %mu.ptr.i to i8*
; CHECK-NEXT:   %mu.size.i = call i64 @__enzyme_get_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %1, i64 8)
; CHECK-NEXT:   %from.trace.mu.i = load double, double* %mu.ptr.i
; CHECK-NEXT:   br label %condition_mu.exit

; CHECK: condition.mu.without.trace.i:                     ; preds = %entry
; CHECK-NEXT:   %sample.mu.i = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %condition_mu.exit

; CHECK: condition_mu.exit:                                ; preds = %condition.mu.with.trace.i, %condition.mu.without.trace.i
; CHECK-NEXT:   %2 = phi double [ %from.trace.mu.i, %condition.mu.with.trace.i ], [ %sample.mu.i, %condition.mu.without.trace.i ]
; CHECK-NEXT:   %3 = bitcast double* %mu.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %3)
; CHECK-NEXT:   %likelihood.mu = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %2)
; CHECK-NEXT:   %log_prob_sum = load double, double* %likelihood
; CHECK-NEXT:   %4 = fadd double %log_prob_sum, %likelihood.mu
; CHECK-NEXT:   store double %4, double* %likelihood
; CHECK-NEXT:   %5 = bitcast double %2 to i64
; CHECK-NEXT:   %6 = inttoptr i64 %5 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double %likelihood.mu, i8* %6, i64 8)
; CHECK-NEXT:   %7 = bitcast double* %x.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %7)
; CHECK-NEXT:   %has.choice.x.i = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.x.i, label %condition.x.with.trace.i, label %condition.x.without.trace.i

; CHECK: condition.x.with.trace.i:                         ; preds = %condition_mu.exit
; CHECK-NEXT:   %8 = bitcast double* %x.ptr.i to i8*
; CHECK-NEXT:   %x.size.i = call i64 @__enzyme_get_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %8, i64 8)
; CHECK-NEXT:   %from.trace.x.i = load double, double* %x.ptr.i
; CHECK-NEXT:   br label %condition_x.exit

; CHECK: condition.x.without.trace.i:                      ; preds = %condition_mu.exit
; CHECK-NEXT:   %sample.x.i = call double @normal(double %2, double 1.000000e+00)
; CHECK-NEXT:   br label %condition_x.exit

; CHECK: condition_x.exit:                                 ; preds = %condition.x.with.trace.i, %condition.x.without.trace.i
; CHECK-NEXT:   %9 = phi double [ %from.trace.x.i, %condition.x.with.trace.i ], [ %sample.x.i, %condition.x.without.trace.i ]
; CHECK-NEXT:   %10 = bitcast double* %x.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %10)
; CHECK-NEXT:   %likelihood.x = call double @normal_logpdf(double %2, double 1.000000e+00, double %9)
; CHECK-NEXT:   %log_prob_sum1 = load double, double* %likelihood
; CHECK-NEXT:   %11 = fadd double %log_prob_sum1, %likelihood.x
; CHECK-NEXT:   store double %11, double* %likelihood
; CHECK-NEXT:   %12 = bitcast double %9 to i64
; CHECK-NEXT:   %13 = inttoptr i64 %12 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.x, i8* %13, i64 8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }