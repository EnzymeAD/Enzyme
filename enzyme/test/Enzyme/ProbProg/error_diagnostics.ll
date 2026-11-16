; RUN: if [ %llvmver -ge 15 ]; then rm -rf %t && split-file %s %t; fi
; RUN: if [ %llvmver -ge 15 ]; then not %opt < %t/missing_trace.ll %OPnewLoadEnzyme -passes="enzyme" -disable-output 2>&1 | FileCheck %s --check-prefix=MISSING-TRACE; fi
; RUN: if [ %llvmver -ge 15 ]; then not %opt < %t/missing_obs.ll %OPnewLoadEnzyme -passes="enzyme" -disable-output 2>&1 | FileCheck %s --check-prefix=MISSING-OBS; fi
; RUN: if [ %llvmver -ge 15 ]; then not %opt < %t/indirect_sample.ll %OPnewLoadEnzyme -passes="enzyme" -disable-output 2>&1 | FileCheck %s --check-prefix=INDIRECT-SAMPLE; fi
; RUN: if [ %llvmver -ge 15 ]; then not %opt < %t/indirect_observe.ll %OPnewLoadEnzyme -passes="enzyme" -disable-output 2>&1 | FileCheck %s --check-prefix=INDIRECT-OBSERVE; fi
; RUN: if [ %llvmver -ge 15 ]; then not %opt < %t/void_sample.ll %OPnewLoadEnzyme -passes="enzyme" -disable-output 2>&1 | FileCheck %s --check-prefix=VOID-SAMPLE; fi

; MISSING-TRACE: Enzyme: __enzyme_trace/__enzyme_condition requires an enzyme_trace argument
; MISSING-OBS: Enzyme: __enzyme_condition requires an enzyme_observations argument
; INDIRECT-SAMPLE: Enzyme: unable to find static sample or likelihood function for __enzyme_sample
; INDIRECT-OBSERVE: Enzyme: unable to find static likelihood function for __enzyme_observe
; VOID-SAMPLE: Enzyme: __enzyme_sample distribution function cannot return void

;--- missing_trace.ll
@.str_mu = private constant [3 x i8] c"mu\00"
declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)
declare double @__enzyme_sample(ptr, ptr, ptr, double, double)
declare double @__enzyme_trace(ptr)

define double @model() {
  %x = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 0.0, double 1.0)
  ret double %x
}

define double @caller() {
  %r = call double @__enzyme_trace(ptr @model)
  ret double %r
}

;--- missing_obs.ll
@.str_mu = private constant [3 x i8] c"mu\00"
@enzyme_trace = global i32 0
declare double @normal(double, double)
declare double @normal_logpdf(double, double, double)
declare double @__enzyme_sample(ptr, ptr, ptr, double, double)
declare double @__enzyme_condition(ptr, i32, ptr)

define double @model() {
  %x = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 0.0, double 1.0)
  ret double %x
}

define double @caller(ptr %tr) {
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_condition(ptr @model, i32 %tt, ptr %tr)
  ret double %r
}

;--- indirect_sample.ll
@.str_mu = private constant [3 x i8] c"mu\00"
@enzyme_trace = global i32 0
@enzyme_const = global i32 0
declare double @normal_logpdf(double, double, double)
declare ptr @__enzyme_newtrace()
declare void @__enzyme_freetrace(ptr)
declare ptr @__enzyme_get_trace(ptr, ptr)
declare i64 @__enzyme_get_choice(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_call(ptr, ptr, ptr)
declare void @__enzyme_insert_choice(ptr, ptr, double, ptr, i64)
declare void @__enzyme_insert_argument(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_return(ptr, ptr, i64)
declare void @__enzyme_insert_function(ptr, ptr)
declare void @__enzyme_insert_gradient_choice(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_gradient_argument(ptr, ptr, ptr, i64)
declare i1 @__enzyme_has_call(ptr, ptr)
declare i1 @__enzyme_has_choice(ptr, ptr)
declare double @__enzyme_sample(ptr, ptr, ptr, double, double)
declare double @__enzyme_trace(ptr, i32, ptr, i32, ptr)

define double @model(ptr %fn) {
  %x = call double @__enzyme_sample(ptr %fn, ptr @normal_logpdf, ptr @.str_mu, double 0.0, double 1.0)
  ret double %x
}

define double @caller(ptr %fn, ptr %tr) {
  %tk = load i32, ptr @enzyme_const
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace(ptr @model, i32 %tk, ptr %fn, i32 %tt, ptr %tr)
  ret double %r
}

;--- indirect_observe.ll
@.str_mu = private constant [3 x i8] c"mu\00"
@enzyme_trace = global i32 0
@enzyme_const = global i32 0
declare ptr @__enzyme_newtrace()
declare void @__enzyme_freetrace(ptr)
declare ptr @__enzyme_get_trace(ptr, ptr)
declare i64 @__enzyme_get_choice(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_call(ptr, ptr, ptr)
declare void @__enzyme_insert_choice(ptr, ptr, double, ptr, i64)
declare void @__enzyme_insert_argument(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_return(ptr, ptr, i64)
declare void @__enzyme_insert_function(ptr, ptr)
declare void @__enzyme_insert_gradient_choice(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_gradient_argument(ptr, ptr, ptr, i64)
declare i1 @__enzyme_has_call(ptr, ptr)
declare i1 @__enzyme_has_choice(ptr, ptr)
declare double @__enzyme_observe(double, ptr, ptr, double)
declare double @__enzyme_trace(ptr, i32, ptr, i32, ptr)

define double @model(ptr %fn) {
  %x = call double @__enzyme_observe(double 1.0, ptr %fn, ptr @.str_mu, double 0.0)
  ret double %x
}

define double @caller(ptr %fn, ptr %tr) {
  %tk = load i32, ptr @enzyme_const
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace(ptr @model, i32 %tk, ptr %fn, i32 %tt, ptr %tr)
  ret double %r
}

;--- void_sample.ll
@.str_mu = private constant [3 x i8] c"mu\00"
@enzyme_trace = global i32 0
declare void @void_sample(double)
declare double @void_logpdf(double)
declare ptr @__enzyme_newtrace()
declare void @__enzyme_freetrace(ptr)
declare ptr @__enzyme_get_trace(ptr, ptr)
declare i64 @__enzyme_get_choice(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_call(ptr, ptr, ptr)
declare void @__enzyme_insert_choice(ptr, ptr, double, ptr, i64)
declare void @__enzyme_insert_argument(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_return(ptr, ptr, i64)
declare void @__enzyme_insert_function(ptr, ptr)
declare void @__enzyme_insert_gradient_choice(ptr, ptr, ptr, i64)
declare void @__enzyme_insert_gradient_argument(ptr, ptr, ptr, i64)
declare i1 @__enzyme_has_call(ptr, ptr)
declare i1 @__enzyme_has_choice(ptr, ptr)
declare void @__enzyme_sample(ptr, ptr, ptr, double)
declare void @__enzyme_trace(ptr, i32, ptr)

define void @model() {
  call void @__enzyme_sample(ptr @void_sample, ptr @void_logpdf, ptr @.str_mu, double 1.0)
  ret void
}

define void @caller(ptr %tr) {
  %tt = load i32, ptr @enzyme_trace
  call void @__enzyme_trace(ptr @model, i32 %tt, ptr %tr)
  ret void
}
