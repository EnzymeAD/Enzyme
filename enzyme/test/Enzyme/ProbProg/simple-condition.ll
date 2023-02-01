; RUN: %opt < %s %loadEnzyme -enzyme -S | FileCheck %s

@enzyme_condition = global i32 0

@.str = private constant [3 x i8] c"mu\00"
@.str.1 = private constant [2 x i8] c"x\00"

declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)

declare i8* @__enzyme_newtrace()
declare void @__enzyme_freetrace(i8*)
declare i8* @__enzyme_get_trace(i8*, i8*)
declare i64 @__enzyme_get_choice(i8*, i8*, i8*, i64)
declare void @__enzyme_insert_call(i8*, i8*, i8*)
declare void @__enzyme_insert_choice(i8* %trace, i8*, double, i8*, i64)
declare i1 @__enzyme_has_call(i8*, i8*)
declare i1 @__enzyme_has_choice(i8*, i8*)
declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare i8* @__enzyme_trace(void ()*)
declare i8* @__enzyme_condition(void ()*, i32, i8*)

define void @test() {
entry:
  %mu = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.0, double 1.0)
  %x = call double @__enzyme_sample(double (double, double)* @normal, double (double, double, double)* @normal_logpdf, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %mu, double 1.0)
  ret void
}

define i8* @generate() {
entry:
  %call = call i8* @__enzyme_trace(void ()* @test)
  ret i8* %call
}

define i8* @condition(i8* %trace) {
entry:
  %0 = load i32, i32* @enzyme_condition
  %call = call i8* @__enzyme_condition(void ()* @test, i32 %0, i8* %trace)
  ret i8* %call
}


; CHECK: define i8* @condition(i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load i32, i32* @enzyme_condition
; CHECK-NEXT:   %1 = call i8* @condition_test(i8* %trace)
; CHECK-NEXT:   ret i8* %1
; CHECK-NEXT: }


; CHECK: define internal i8* @condition_test(i8* %trace)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %trace1 = call i8* @__enzyme_newtrace()
; CHECK-NEXT:   %0 = call i1 @__enzyme_has_choice(i8* %trace, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   br i1 %0, label %condition.mu.with.trace, label %condition.mu.without.trace

; CHECK: condition.mu.with.trace:                          ; preds = %entry
; CHECK-NEXT:   %1 = alloca double
; CHECK-NEXT:   %2 = bitcast double* %1 to i8*
; CHECK-NEXT:   %3 = call i64 @__enzyme_get_choice(i8* %trace, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %2, i64 8)
; CHECK-NEXT:   %4 = load double, double* %1
; CHECK-NEXT:   br label %entry.cntd

; CHECK: condition.mu.without.trace:                       ; preds = %entry
; CHECK-NEXT:   %sample.mu = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd

; CHECK: entry.cntd:                                       ; preds = %condition.mu.without.trace, %condition.mu.with.trace
; CHECK-NEXT:   %5 = phi double [ %4, %condition.mu.with.trace ], [ %sample.mu, %condition.mu.without.trace ]
; CHECK-NEXT:   %likelihood.mu = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %5)
; CHECK-NEXT:   %6 = alloca double
; CHECK-NEXT:   store double %5, double* %6
; CHECK-NEXT:   %7 = bitcast double* %6 to i8**
; CHECK-NEXT:   %8 = load i8*, i8** %7
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double %likelihood.mu, i8* %8, i64 8)
; CHECK-NEXT:   %9 = call i1 @__enzyme_has_choice(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   br i1 %9, label %condition.x.with.trace, label %condition.x.without.trace

; CHECK: condition.x.with.trace:                           ; preds = %entry.cntd
; CHECK-NEXT:   %10 = alloca double
; CHECK-NEXT:   %11 = bitcast double* %10 to i8*
; CHECK-NEXT:   %12 = call i64 @__enzyme_get_choice(i8* %trace, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %11, i64 8)
; CHECK-NEXT:   %13 = load double, double* %10
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: condition.x.without.trace:                        ; preds = %entry.cntd
; CHECK-NEXT:   %sample.x = call double @normal(double %5, double 1.000000e+00)
; CHECK-NEXT:   br label %entry.cntd.cntd

; CHECK: entry.cntd.cntd:                                  ; preds = %condition.x.without.trace, %condition.x.with.trace
; CHECK-NEXT:   %14 = phi double [ %13, %condition.x.with.trace ], [ %sample.x, %condition.x.without.trace ]
; CHECK-NEXT:   %likelihood.x = call double @normal_logpdf(double %5, double 1.000000e+00, double %14)
; CHECK-NEXT:   %15 = alloca double
; CHECK-NEXT:   store double %14, double* %15
; CHECK-NEXT:   %16 = bitcast double* %15 to i8**
; CHECK-NEXT:   %17 = load i8*, i8** %16
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace1, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.x, i8* %17, i64 8)
; CHECK-NEXT:   ret i8* %trace1
; CHECK-NEXT: }