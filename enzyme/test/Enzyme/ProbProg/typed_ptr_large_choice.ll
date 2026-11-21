; RUN: if [ %llvmver -le 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi

@.str_mu = private constant [3 x i8] c"mu\00"
@enzyme_trace = global i32 0

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

declare <2 x double> @sample_v2d()
declare double @v2d_logpdf(<2 x double>)
declare <2 x double> @__enzyme_sample(<2 x double> ()*, double (<2 x double>)*, i8*)
declare <2 x double> @__enzyme_trace(<2 x double> ()*, i32, i8*)

define <2 x double> @model() {
  %v = call <2 x double> @__enzyme_sample(<2 x double> ()* @sample_v2d, double (<2 x double>)* @v2d_logpdf, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str_mu, i64 0, i64 0))
  ret <2 x double> %v
}

define <2 x double> @caller(i8* %tr) {
  %tt = load i32, i32* @enzyme_trace
  %r = call <2 x double> @__enzyme_trace(<2 x double> ()* @model, i32 %tt, i8* %tr)
  ret <2 x double> %r
}

; CHECK-LABEL: define internal <2 x double> @trace_model(double* "enzyme_likelihood" %likelihood, i8* "enzyme_trace" %trace)
; CHECK: alloca <2 x double>
; CHECK: bitcast <2 x double>* {{.*}} to i8*
; CHECK: call void @__enzyme_insert_choice({{.*}}, i64 16)
; CHECK: call void @__enzyme_insert_return({{.*}}, i64 16)
