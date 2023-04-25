; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s

@enzyme_observations = global i32 0
@enzyme_duptrace = global i32 0
@enzyme_trace = global i32 0

@.str = private constant [3 x i8] c"mu\00"
@.str.1 = private constant [2 x i8] c"x\00"

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
declare void @__enzyme_condition(void ()*, i32, i8*, i32, i8*, i8*)

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
  %2 = load i32, i32* @enzyme_duptrace
  %trace = call i8* @__enzyme_newtrace()
  %dtrace = call i8* @__enzyme_newtrace()
  call void @__enzyme_condition(void ()* @test, i32 %0, i8* %observations, i32 %2, i8* %trace, i8* %dtrace)
  ret i8* %dtrace
}


; CHECK: define internal void @diffecondition_test(i8* "enzyme_observations" %observations, i8* "enzyme_trace" %trace, i8* %"trace'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %x.ptr.i = alloca double
; CHECK-NEXT:   %mu.ptr.i = alloca double
; CHECK-NEXT:   %"mu'de" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, double* %"mu'de"
; CHECK-NEXT:   %.ptr1 = alloca double
; CHECK-NEXT:   %.ptr = alloca double
; CHECK-NEXT:   call void @__enzyme_insert_function(i8* %trace, i8* bitcast (void (i8*, i8*)* @condition_test to i8*))
; CHECK-NEXT:   %0 = bitcast double* %mu.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %0)
; CHECK-NEXT:   %has.choice.mu.i = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.mu.i, label %condition.mu.with.trace.i, label %condition.mu.without.trace.i

; CHECK: condition.mu.with.trace.i:                        ; preds = %entry
; CHECK-NEXT:   %1 = bitcast double* %mu.ptr.i to i8*
; CHECK-NEXT:   %mu.size.i = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %1, i64 8)
; CHECK-NEXT:   %from.trace.mu.i = load double, double* %mu.ptr.i
; CHECK-NEXT:   br label %condition_mu.exit

; CHECK: condition.mu.without.trace.i:                     ; preds = %entry
; CHECK-NEXT:   %sample.mu.i = call double @normal(double 0.000000e+00, double 1.000000e+00)
; CHECK-NEXT:   br label %condition_mu.exit

; CHECK: condition_mu.exit:                                ; preds = %condition.mu.with.trace.i, %condition.mu.without.trace.i
; CHECK-NEXT:   %2 = phi double [ %from.trace.mu.i, %condition.mu.with.trace.i ], [ %sample.mu.i, %condition.mu.without.trace.i ]
; CHECK-NEXT:   %likelihood.mu.i = call double @normal_logpdf(double 0.000000e+00, double 1.000000e+00, double %2)
; CHECK-NEXT:   %3 = bitcast double %2 to i64
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double %likelihood.mu.i, i8* %4, i64 8)
; CHECK-NEXT:   %5 = bitcast double* %mu.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %5)
; CHECK-NEXT:   %6 = bitcast double* %x.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 8, i8* %6)
; CHECK-NEXT:   %has.choice.x.i = call i1 @__enzyme_has_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   br i1 %has.choice.x.i, label %condition.x.with.trace.i, label %condition.x.without.trace.i

; CHECK: condition.x.with.trace.i:                         ; preds = %condition_mu.exit
; CHECK-NEXT:   %7 = bitcast double* %x.ptr.i to i8*
; CHECK-NEXT:   %x.size.i = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %7, i64 8)
; CHECK-NEXT:   %from.trace.x.i = load double, double* %x.ptr.i
; CHECK-NEXT:   br label %condition_x.exit

; CHECK: condition.x.without.trace.i:                      ; preds = %condition_mu.exit
; CHECK-NEXT:   %sample.x.i = call double @normal(double %2, double 1.000000e+00)
; CHECK-NEXT:   br label %condition_x.exit

; CHECK: condition_x.exit:                                 ; preds = %condition.x.with.trace.i, %condition.x.without.trace.i
; CHECK-NEXT:   %8 = phi double [ %from.trace.x.i, %condition.x.with.trace.i ], [ %sample.x.i, %condition.x.without.trace.i ]
; CHECK-NEXT:   %likelihood.x.i = call double @normal_logpdf(double %2, double 1.000000e+00, double %8)
; CHECK-NEXT:   %9 = bitcast double %8 to i64
; CHECK-NEXT:   %10 = inttoptr i64 %9 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %trace, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double %likelihood.x.i, i8* %10, i64 8)
; CHECK-NEXT:   %11 = bitcast double* %x.ptr.i to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 8, i8* %11)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %condition_x.exit
; CHECK-NEXT:   %12 = call fast double @__enzyme_get_likelihood(i8* %observations, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   %13 = call { double, double } @diffenormal_logpdf(double %2, double 1.000000e+00, double %8, double %12)
; CHECK-NEXT:   %14 = extractvalue { double, double } %13, 0
; CHECK-NEXT:   %15 = load double, double* %"mu'de"
; CHECK-NEXT:   %16 = fadd fast double %15, %14
; CHECK-NEXT:   store double %16, double* %"mu'de"
; CHECK-NEXT:   %17 = extractvalue { double, double } %13, 1
; CHECK-NEXT:   %18 = bitcast double* %.ptr to i8*
; CHECK-NEXT:   %.size = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %18, i64 8)
; CHECK-NEXT:   %from.trace. = load double, double* %.ptr
; CHECK-NEXT:   %19 = fadd fast double %from.trace., %8
; CHECK-NEXT:   %20 = bitcast double %19 to i64
; CHECK-NEXT:   %21 = inttoptr i64 %20 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double 0.000000e+00, i8* %21, i64 8)
; CHECK-NEXT:   %22 = call fast double @__enzyme_get_likelihood(i8* %observations, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   %23 = call { double } @diffenormal_logpdf.1(double 0.000000e+00, double 1.000000e+00, double %2, double %22)
; CHECK-NEXT:   %24 = extractvalue { double } %23, 0
; CHECK-NEXT:   %25 = bitcast double* %.ptr1 to i8*
; CHECK-NEXT:   %.size2 = call i64 @__enzyme_get_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %25, i64 8)
; CHECK-NEXT:   %from.trace.3 = load double, double* %.ptr1
; CHECK-NEXT:   %26 = fadd fast double %from.trace.3, %2
; CHECK-NEXT:   %27 = bitcast double %26 to i64
; CHECK-NEXT:   %28 = inttoptr i64 %27 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %observations, i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.000000e+00, i8* %28, i64 8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }