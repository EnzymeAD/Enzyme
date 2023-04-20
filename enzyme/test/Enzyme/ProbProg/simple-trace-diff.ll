; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

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
declare double @__enzyme_get_likelihood(i8*, i8*)
declare void @__enzyme_insert_call(i8*, i8*, i8*)
declare void @__enzyme_insert_choice(i8* %trace, i8*, double, i8*, i64)
declare i1 @__enzyme_has_call(i8*, i8*)
declare i1 @__enzyme_has_choice(i8*, i8*)
declare double @__enzyme_sample(double (double, double)*, double (double, double, double)*, i8*, double, double)
declare void @__enzyme_trace(void ()*, i32, i8*, i8*)

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
  %dtrace = call i8* @__enzyme_newtrace()
  tail call void @__enzyme_trace(void ()* @test, i32 %0, i8* %trace, i8* %dtrace)
  ret i8* %dtrace
}


; CHECK: define internal void @diffetrace_test(i8* "enzyme_trace" %trace, i8* %"trace'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"mu'de" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, double* %"mu'de"
; CHECK-NEXT:   %.ptr1 = alloca double
; CHECK-NEXT:   %.ptr = alloca double
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
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %6 = call fast double @__enzyme_get_likelihood(i8* %"trace'", i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
; CHECK-NEXT:   %7 = call { double, double } @diffenormal_logpdf(double %0, double 1.000000e+00, double %3, double %6)
; CHECK-NEXT:   %8 = extractvalue { double, double } %7, 0
; CHECK-NEXT:   %9 = load double, double* %"mu'de"
; CHECK-NEXT:   %10 = fadd fast double %9, %8
; CHECK-NEXT:   store double %10, double* %"mu'de"
; CHECK-NEXT:   %11 = extractvalue { double, double } %7, 1
; CHECK-NEXT:   %12 = bitcast double* %.ptr to i8*
; CHECK-NEXT:   %.size = call i64 @__enzyme_get_choice(i8* %"trace'", i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8* %12, i64 8)
; CHECK-NEXT:   %from.trace. = load double, double* %.ptr
; CHECK-NEXT:   %13 = fadd fast double %from.trace., %3
; CHECK-NEXT:   %14 = bitcast double %13 to i64
; CHECK-NEXT:   %15 = inttoptr i64 %14 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %"trace'", i8* nocapture readonly getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), double 0.000000e+00, i8* %15, i64 8)
; CHECK-NEXT:   %16 = call fast double @__enzyme_get_likelihood(i8* %"trace'", i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
; CHECK-NEXT:   %17 = call { double } @diffenormal_logpdf.1(double 0.000000e+00, double 1.000000e+00, double %0, double %16)
; CHECK-NEXT:   %18 = extractvalue { double } %17, 0
; CHECK-NEXT:   %19 = bitcast double* %.ptr1 to i8*
; CHECK-NEXT:   %.size2 = call i64 @__enzyme_get_choice(i8* %"trace'", i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8* %19, i64 8)
; CHECK-NEXT:   %from.trace.3 = load double, double* %.ptr1
; CHECK-NEXT:   %20 = fadd fast double %from.trace.3, %0
; CHECK-NEXT:   %21 = bitcast double %20 to i64
; CHECK-NEXT:   %22 = inttoptr i64 %21 to i8*
; CHECK-NEXT:   call void @__enzyme_insert_choice(i8* %"trace'", i8* nocapture readonly getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), double 0.000000e+00, i8* %22, i64 8)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }