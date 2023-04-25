; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

@.str = private constant [3 x i8] c"mu\00"
@.str.1 = private constant [2 x i8] c"x\00"

@enzyme_trace = global i32 0

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

declare i8* @__enzyme_newtrace()
declare void @__enzyme_freetrace(i8*)
declare i8* @__enzyme_get_trace(i8*, i8*)
declare i64 @__enzyme_get_choice(i8*, i8*, i8*, i64)
declare double @__enzyme_get_likelihood(i8*, i8*)
declare void @__enzyme_insert_call(i8*, i8*, i8*)
declare void @__enzyme_insert_choice(i8*, i8*, double, i8*, i64)
declare void @__enzyme_insert_argument(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_return(i8*, i8*, i64)
declare void @__enzyme_insert_function(i8*, i8*)
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
  %0 = load i32, i32* @enzyme_trace
  %trace = call i8* @__enzyme_newtrace()
  tail call void @__enzyme_trace(void ()* @test, i32 %0, i8* %trace)
  ret i8* %trace
}


; CHECK: define internal void @trace_test(i8* "enzyme_trace" %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @__enzyme_insert_function(i8* %trace, i8* bitcast (void (i8*)* @trace_test to i8*))
; CHECK-NEXT:   %0 = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.mu.i = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %0)
; CHECK-NEXT:   %1 = bitcast double %0 to i64
; CHECK-NEXT:   %2 = inttoptr i64 %1 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double %likelihood.mu.i, i8* %2, i64 8)
; CHECK-NEXT:   %3 = call double @normal(double %0, double 1.000000e+00)
; CHECK-NEXT:   %likelihood.x.i = call double @normal_logpdf(double %0, double 1.000000e+00, double %3)
; CHECK-NEXT:   %4 = bitcast double %3 to i64
; CHECK-NEXT:   %5 = inttoptr i64 %4 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.x.i, i8* %5, i64 8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }