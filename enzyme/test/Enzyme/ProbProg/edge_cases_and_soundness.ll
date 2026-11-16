; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O1>,verify" -S | %lli; fi

@.str_mu = private constant [3 x i8] c"mu\00"
@.str_x = private constant [2 x i8] c"x\00"
@.str_v2d = private constant [4 x i8] c"v2d\00"
@.str_v2p = private constant [4 x i8] c"v2p\00"
@.str_s1 = private constant [3 x i8] c"s1\00"

@enzyme_trace = global i32 0
@enzyme_observations = global i32 0
@enzyme_likelihood = global i32 0
@enzyme_duplikelihood = global i32 0
@enzyme_const = global i32 0
@enzyme_interface = global i32 0
@enzyme_active_rand_var = global i32 0

; Runtime state inspected by @main
@last_choice_size = global i64 0
@last_ret_size = global i64 0
@last_get_choice_size = global i64 0
@last_pair_first = global double 0.0
@last_pair_second = global double 0.0
@dyn1_hits = global i32 0
@dyn2_hits = global i32 0
@sink_seen = global ptr null
@size_sum = global i64 0

; Static trace interface implementations for %lli E2E execution
define ptr @__enzyme_newtrace() {
  ret ptr null
}

define void @__enzyme_freetrace(ptr %t) {
  ret void
}

define ptr @__enzyme_get_trace(ptr %obs, ptr %addr) {
  ret ptr %obs
}

define i64 @__enzyme_get_choice(ptr %obs, ptr %addr, ptr %dest, i64 %sz) {
  store i64 %sz, ptr @last_get_choice_size
  %is16 = icmp eq i64 %sz, 16
  br i1 %is16, label %w16, label %check1
w16:
  store { double, double } { double 7.0, double 9.0 }, ptr %dest
  ret i64 16
check1:
  %is1 = icmp eq i64 %sz, 1
  br i1 %is1, label %w1, label %w8
w1:
  store i1 true, ptr %dest
  ret i64 1
w8:
  store double 11.0, ptr %dest
  ret i64 %sz
}

define void @__enzyme_insert_call(ptr %tr, ptr %addr, ptr %sub) {
  ret void
}

define void @__enzyme_insert_choice(ptr %tr, ptr %addr, double %score, ptr %val, i64 %sz) {
  store i64 %sz, ptr @last_choice_size
  %prev = load i64, ptr @size_sum
  %ns = add i64 %prev, %sz
  store i64 %ns, ptr @size_sum
  %is16 = icmp eq i64 %sz, 16
  br i1 %is16, label %check_mu, label %done
check_mu:
  %is_mu = icmp eq ptr %addr, @.str_mu
  br i1 %is_mu, label %read_pair, label %done
read_pair:
  %p = load { double, double }, ptr %val
  %f = extractvalue { double, double } %p, 0
  %s = extractvalue { double, double } %p, 1
  store double %f, ptr @last_pair_first
  store double %s, ptr @last_pair_second
  br label %done
done:
  ret void
}

define void @__enzyme_insert_argument(ptr %tr, ptr %name, ptr %val, i64 %sz) {
  ret void
}

define void @__enzyme_insert_return(ptr %tr, ptr %val, i64 %sz) {
  store i64 %sz, ptr @last_ret_size
  ret void
}

define void @__enzyme_insert_function(ptr %tr, ptr %fn) {
  ret void
}

define void @__enzyme_insert_gradient_choice(ptr %tr, ptr %addr, ptr %val, i64 %sz) {
  ret void
}

define void @__enzyme_insert_gradient_argument(ptr %tr, ptr %name, ptr %val, i64 %sz) {
  ret void
}

define i1 @__enzyme_has_call(ptr %obs, ptr %addr) {
  %nonnull = icmp ne ptr %obs, null
  ret i1 %nonnull
}

define i1 @__enzyme_has_choice(ptr %obs, ptr %addr) {
  %nonnull = icmp ne ptr %obs, null
  ret i1 %nonnull
}

; Dynamic interface hooks for caller_dyn_1 vs caller_dyn_2
define void @dyn1_insert_choice(ptr %tr, ptr %addr, double %score, ptr %val, i64 %sz) {
  %c = load i32, ptr @dyn1_hits
  %n = add i32 %c, 1
  store i32 %n, ptr @dyn1_hits
  ret void
}

define void @dyn2_insert_choice(ptr %tr, ptr %addr, double %score, ptr %val, i64 %sz) {
  %c = load i32, ptr @dyn2_hits
  %n = add i32 %c, 1
  store i32 %n, ptr @dyn2_hits
  ret void
}

; Distributions and logpdfs
define double @normal(double %m, double %s) {
  %r = fadd double %m, %s
  ret double %r
}

define double @normal_logpdf(double %m, double %s, double %x) {
  %d = fsub double %x, %m
  %p = fmul double %d, %s
  ret double %p
}

define { double, double } @sample_pair(double %a) {
  %b = fadd double %a, 3.0
  %p0 = insertvalue { double, double } poison, double %a, 0
  %p1 = insertvalue { double, double } %p0, double %b, 1
  ret { double, double } %p1
}

define double @pair_logpdf(double %a, { double, double } %p) {
  %f = extractvalue { double, double } %p, 0
  %s = extractvalue { double, double } %p, 1
  %sum = fadd double %f, %s
  ret double %sum
}

define i1 @bernoulli(double %p) {
  ret i1 false
}

define double @bernoulli_logpdf(double %p, i1 %b) {
  %v = select i1 %b, double 1.0, double 0.0
  ret double %v
}

define <2 x double> @sample_v2d() {
  ret <2 x double> <double 1.0, double 2.0>
}

define double @v2d_logpdf(<2 x double> %v) {
  ret double 1.0
}

define <2 x ptr> @sample_v2p() {
  ret <2 x ptr> zeroinitializer
}

define double @v2p_logpdf(<2 x ptr> %v) {
  ret double 1.0
}

define { i32 } @sample_s1() {
  ret { i32 } { i32 42 }
}

define double @s1_logpdf({ i32 } %s) {
  ret double 1.0
}

define fastcc double @fast_sample(double %a) {
  %r = fadd double %a, 2.0
  ret double %r
}

define fastcc double @fast_logpdf(double %a, double %x) {
  %p = fmul double %a, %x
  ret double %p
}

define void @sink_fn(ptr %p) {
  store ptr %p, ptr @sink_seen
  ret void
}

; Enzyme sample/observe/probprog declarations
declare double @__enzyme_sample(ptr, ptr, ptr, double, double)
declare double @__enzyme_sample.vararg(...)
declare { double, double } @__enzyme_sample.pair(ptr, ptr, ptr, double)
declare i1 @__enzyme_sample.i1(ptr, ptr, ptr, double)
declare <2 x double> @__enzyme_sample.v2d(ptr, ptr, ptr)
declare <2 x ptr> @__enzyme_sample.v2p(ptr, ptr, ptr)
declare { i32 } @__enzyme_sample.s1(ptr, ptr, ptr)
declare double @__enzyme_sample.fast(ptr, ptr, ptr, double)
declare double @__enzyme_observe.fast(double, ptr, ptr, double)

declare double @__enzyme_likelihood.d(ptr, i32, ptr)
declare void @__enzyme_likelihood.void(ptr, i32, ptr, i32, ptr)
declare void @__enzyme_likelihood.dup(ptr, i32, ptr, ptr, i32, ptr)
declare void @__enzyme_likelihood.dup_fast(ptr, i32, ptr, ptr)
declare { double, double } @__enzyme_trace.pair(ptr, i32, ptr)
declare { double, double } @__enzyme_condition.pair(ptr, i32, ptr, i32, ptr)
declare i1 @__enzyme_condition.i1(ptr, i32, ptr, i32, ptr)
declare { i32 } @__enzyme_trace.s1(ptr, i32, ptr)
declare void @__enzyme_trace.void(ptr, i32, ptr, i32, ptr)
declare void @__enzyme_condition.void(ptr, i32, ptr, i32, ptr, i32, ptr)
declare double @__enzyme_trace.rec(ptr, i32, i32, i32, ptr)
declare double @__enzyme_trace.dyn(ptr, i32, ptr, i32, ptr)
declare double @__enzyme_trace.dyn_as1(ptr, i32, ptr addrspace(1), i32, ptr)
declare double @__enzyme_trace.d(ptr, i32, ptr, i32, ptr)
declare double @__enzyme_trace.d0(ptr, i32, ptr)
declare double @__enzyme_condition.d(ptr, i32, ptr, i32, ptr)

; =============================================================================
; 1. Likelihood mode with non-void return (previously SIGSEGV on null interface)
; =============================================================================
define double @simple_model() {
  %x = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 1.0, double 2.0)
  ret double %x
}

define double @caller_likelihood_ret(ptr %lik) {
  %tl = load i32, ptr @enzyme_likelihood
  %r = call double @__enzyme_likelihood.d(ptr @simple_model, i32 %tl, ptr %lik)
  ret double %r
}

; =============================================================================
; 2. Aggregate { double, double } choice & return in Trace and Condition modes
;    (previously Invalid bitcast to i0 + empty entry block SIGSEGV)
; =============================================================================
define { double, double } @pair_model() {
  %p = call { double, double } @__enzyme_sample.pair(ptr @sample_pair, ptr @pair_logpdf, ptr @.str_mu, double 2.0)
  ret { double, double } %p
}

define { double, double } @caller_pair_trace(ptr %tr) {
  %tt = load i32, ptr @enzyme_trace
  %r = call { double, double } @__enzyme_trace.pair(ptr @pair_model, i32 %tt, ptr %tr)
  ret { double, double } %r
}

define { double, double } @caller_pair_condition(ptr %obs, ptr %tr) {
  %tc = load i32, ptr @enzyme_observations
  %tt = load i32, ptr @enzyme_trace
  %r = call { double, double } @__enzyme_condition.pair(ptr @pair_model, i32 %tc, ptr %obs, i32 %tt, ptr %tr)
  ret { double, double } %r
}

; =============================================================================
; 3. Sub-byte i1 choice size (previously passed size 0 instead of 1)
; =============================================================================
define i1 @i1_model() {
  %b = call i1 @__enzyme_sample.i1(ptr @bernoulli, ptr @bernoulli_logpdf, ptr @.str_mu, double 0.5)
  ret i1 %b
}

define i1 @caller_i1_condition(ptr %obs, ptr %tr) {
  %tc = load i32, ptr @enzyme_observations
  %tt = load i32, ptr @enzyme_trace
  %r = call i1 @__enzyme_condition.i1(ptr @i1_model, i32 %tc, ptr %obs, i32 %tt, ptr %tr)
  ret i1 %r
}

; =============================================================================
; 4. Large vector (<2 x double>), vector of pointers (<2 x ptr>), small struct ({ i32 })
; =============================================================================
define { i32 } @vec_struct_model() {
  %v = call <2 x double> @__enzyme_sample.v2d(ptr @sample_v2d, ptr @v2d_logpdf, ptr @.str_v2d)
  %p = call <2 x ptr> @__enzyme_sample.v2p(ptr @sample_v2p, ptr @v2p_logpdf, ptr @.str_v2p)
  %s = call { i32 } @__enzyme_sample.s1(ptr @sample_s1, ptr @s1_logpdf, ptr @.str_s1)
  ret { i32 } %s
}

define { i32 } @caller_vec_struct(ptr %tr) {
  %tt = load i32, ptr @enzyme_trace
  %r = call { i32 } @__enzyme_trace.s1(ptr @vec_struct_model, i32 %tt, ptr %tr)
  ret { i32 } %r
}

; =============================================================================
; 5. Void generative callee in Trace, Condition, and Likelihood modes
;    (previously crashed creating named void call / phi void)
; =============================================================================
define void @void_sub(ptr %out) {
  %x = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 1.0, double 2.0)
  store double %x, ptr %out
  ret void
}

define void @void_model(ptr %out) {
  call void @void_sub(ptr %out)
  ret void
}

define void @caller_void_trace(ptr %out, ptr %tr) {
  %tk = load i32, ptr @enzyme_const
  %tt = load i32, ptr @enzyme_trace
  call void @__enzyme_trace.void(ptr @void_model, i32 %tk, ptr %out, i32 %tt, ptr %tr)
  ret void
}

define void @caller_void_cond(ptr %out, ptr %obs, ptr %tr) {
  %tk = load i32, ptr @enzyme_const
  %tc = load i32, ptr @enzyme_observations
  %tt = load i32, ptr @enzyme_trace
  call void @__enzyme_condition.void(ptr @void_model, i32 %tk, ptr %out, i32 %tc, ptr %obs, i32 %tt, ptr %tr)
  ret void
}

define void @caller_void_lik(ptr %out, ptr %lik) {
  %tk = load i32, ptr @enzyme_const
  %tl = load i32, ptr @enzyme_likelihood
  call void @__enzyme_likelihood.void(ptr @void_model, i32 %tk, ptr %out, i32 %tl, ptr %lik)
  ret void
}

; =============================================================================
; 6. Direct and mutual recursion (previously infinite recursion / stack overflow)
; =============================================================================
define double @even_m(i32 %n) {
  %c = icmp sle i32 %n, 0
  br i1 %c, label %done, label %rec
rec:
  %x = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 1.0, double 2.0)
  %n1 = sub i32 %n, 1
  %y = call double @odd_m(i32 %n1)
  %s = fadd double %x, %y
  ret double %s
done:
  ret double 0.0
}

define double @odd_m(i32 %n) {
  %c = icmp sle i32 %n, 0
  br i1 %c, label %done, label %rec
rec:
  %n1 = sub i32 %n, 1
  %y = call double @even_m(i32 %n1)
  ret double %y
done:
  ret double 1.0
}

define double @caller_mutual_rec(i32 %n, ptr %tr) {
  %tk = load i32, ptr @enzyme_const
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace.rec(ptr @even_m, i32 %tk, i32 %n, i32 %tt, ptr %tr)
  ret double %r
}

; =============================================================================
; 7. DynamicTraceInterface: local alloca dominance + two callers cache isolation
;    (previously dominance error + dangling pointer cache collision)
; =============================================================================
define double @caller_dyn_1(ptr %tr, ptr %iface1) {
  %slot = alloca ptr
  store ptr %iface1, ptr %slot
  %ld = load ptr, ptr %slot
  %ti = load i32, ptr @enzyme_interface
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace.dyn(ptr @simple_model, i32 %ti, ptr %ld, i32 %tt, ptr %tr)
  ret double %r
}

define double @caller_dyn_2(ptr %tr, ptr %iface2) {
  %ti = load i32, ptr @enzyme_interface
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace.dyn(ptr @simple_model, i32 %ti, ptr %iface2, i32 %tt, ptr %tr)
  ret double %r
}

; =============================================================================
; 8. Readonly model attribute stripped on traced clone (previously unsound readonly)
; =============================================================================
define double @readonly_model() readonly {
  %x = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 1.0, double 2.0)
  ret double %x
}

define double @caller_readonly_model(ptr %lik, ptr %tr) {
  %tl = load i32, ptr @enzyme_likelihood
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace.d(ptr @readonly_model, i32 %tl, ptr %lik, i32 %tt, ptr %tr)
  ret double %r
}

; =============================================================================
; 9. fastcc model, samplefn, likelihoodfn, observe, and autodiff calling convention
; =============================================================================
define fastcc double @fastcc_model() {
  %x = call double @__enzyme_sample.fast(ptr @fast_sample, ptr @fast_logpdf, ptr @.str_mu, double 3.0)
  %y = call double @__enzyme_observe.fast(double 4.0, ptr @fast_logpdf, ptr @.str_x, double %x)
  ret double %y
}

define double @caller_fastcc_cond(ptr %obs, ptr %tr) {
  %tc = load i32, ptr @enzyme_observations
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_condition.d(ptr @fastcc_model, i32 %tc, ptr %obs, i32 %tt, ptr %tr)
  ret double %r
}

define void @caller_fastcc_autodiff(ptr %lik, ptr %dlik) {
  %td = load i32, ptr @enzyme_duplikelihood
  call void @__enzyme_likelihood.dup_fast(ptr @fastcc_model, i32 %td, ptr %lik, ptr %dlik)
  ret void
}

; =============================================================================
; 10. Non-call use of generative function does not mark caller as generative
; =============================================================================
define void @helper_takes_ptr() {
  call void @sink_fn(ptr @simple_model)
  ret void
}

define double @model_calls_helper() {
  call void @helper_takes_ptr()
  %x = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 1.0, double 2.0)
  ret double %x
}

define double @caller_noncall_use(ptr %tr) {
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace.d0(ptr @model_calls_helper, i32 %tt, ptr %tr)
  ret double %r
}

; =============================================================================
; 11. Vararg __enzyme_sample declaration (previously hit getNumParams() >= 3 assert)
; =============================================================================
define double @vararg_model() {
  %x = call double (...) @__enzyme_sample.vararg(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 1.0, double 2.0)
  ret double %x
}

define double @caller_vararg_sample(ptr %lik, ptr %tr) {
  %tl = load i32, ptr @enzyme_likelihood
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace.d(ptr @vararg_model, i32 %tl, ptr %lik, i32 %tt, ptr %tr)
  ret double %r
}

; =============================================================================
; 12. __enzyme_trace inside a loop hoists implicit %likelihood alloca to entry
; =============================================================================
define double @caller_loop_trace(i32 %n, ptr %tr) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi double [ 0.0, %entry ], [ %nsum, %loop ]
  %tt = load i32, ptr @enzyme_trace
  %v = call double @__enzyme_trace.d0(ptr @simple_model, i32 %tt, ptr %tr)
  %nsum = fadd double %sum, %v
  %inc = add i32 %i, 1
  %cond = icmp slt i32 %inc, %n
  br i1 %cond, label %loop, label %exit
exit:
  ret double %nsum
}

; =============================================================================
; 13. Distinct ActiveRandomVariables do not collide in TraceCachedFunctions
; =============================================================================
define double @two_var_model() {
  %a = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_mu, double 1.0, double 2.0)
  %b = call double @__enzyme_sample(ptr @normal, ptr @normal_logpdf, ptr @.str_x, double %a, double 2.0)
  ret double %b
}

define void @caller_active_mu(ptr %lik, ptr %dlik) {
  %td = load i32, ptr @enzyme_duplikelihood
  %ta = load i32, ptr @enzyme_active_rand_var
  call void @__enzyme_likelihood.dup(ptr @two_var_model, i32 %td, ptr %lik, ptr %dlik, i32 %ta, ptr @.str_mu)
  ret void
}

define void @caller_active_x(ptr %lik, ptr %dlik) {
  %td = load i32, ptr @enzyme_duplikelihood
  %ta = load i32, ptr @enzyme_active_rand_var
  call void @__enzyme_likelihood.dup(ptr @two_var_model, i32 %td, ptr %lik, ptr %dlik, i32 %ta, ptr @.str_x)
  ret void
}

; =============================================================================
; 14. DynamicTraceInterface pointer in addrspace(1) uses program addrspace 0
; =============================================================================
define double @caller_dyn_as1(ptr %tr, ptr addrspace(1) %iface) {
  %ti = load i32, ptr @enzyme_interface
  %tt = load i32, ptr @enzyme_trace
  %r = call double @__enzyme_trace.dyn_as1(ptr @simple_model, i32 %ti, ptr addrspace(1) %iface, i32 %tt, ptr %tr)
  ret double %r
}

; =============================================================================
; Runtime E2E verification under %lli
; =============================================================================
define i32 @main() {
entry:
  %lik = alloca double, align 8
  %dlik = alloca double, align 8
  %out = alloca double, align 8
  %dummy_obs = alloca i8, align 1
  %iface1 = alloca [13 x ptr], align 8
  %iface2 = alloca [13 x ptr], align 8

  ; 1. Likelihood mode non-void return: x = normal(1, 2) = 3, score = (3-1)*2 = 4
  store double 0.0, ptr %lik
  %r1 = call double @caller_likelihood_ret(ptr %lik)
  %ok1a = fcmp oeq double %r1, 3.0
  %l1 = load double, ptr %lik
  %ok1b = fcmp oeq double %l1, 4.0
  %ok1 = and i1 %ok1a, %ok1b
  br i1 %ok1, label %t2, label %fail1

t2:
  ; 2a. Aggregate trace: pair = {2, 5}, size = 16
  %p2 = call { double, double } @caller_pair_trace(ptr null)
  %p2_0 = extractvalue { double, double } %p2, 0
  %p2_1 = extractvalue { double, double } %p2, 1
  %cs2 = load i64, ptr @last_choice_size
  %rs2 = load i64, ptr @last_ret_size
  %lf2 = load double, ptr @last_pair_first
  %ls2 = load double, ptr @last_pair_second
  %ok2a = fcmp oeq double %p2_0, 2.0
  %ok2b = fcmp oeq double %p2_1, 5.0
  %ok2c = icmp eq i64 %cs2, 16
  %ok2d = icmp eq i64 %rs2, 16
  %ok2e = fcmp oeq double %lf2, 2.0
  %ok2f = fcmp oeq double %ls2, 5.0
  %ok2ab = and i1 %ok2a, %ok2b
  %ok2cd = and i1 %ok2c, %ok2d
  %ok2ef = and i1 %ok2e, %ok2f
  %ok2x = and i1 %ok2ab, %ok2cd
  %ok2 = and i1 %ok2x, %ok2ef
  br i1 %ok2, label %t2cond, label %fail2

t2cond:
  ; 2b. Aggregate condition: conditioned from obs -> {7, 9}, get_choice size = 16
  %p2c = call { double, double } @caller_pair_condition(ptr %dummy_obs, ptr null)
  %p2c_0 = extractvalue { double, double } %p2c, 0
  %p2c_1 = extractvalue { double, double } %p2c, 1
  %gs2 = load i64, ptr @last_get_choice_size
  %ok2ca = fcmp oeq double %p2c_0, 7.0
  %ok2cb = fcmp oeq double %p2c_1, 9.0
  %ok2cc = icmp eq i64 %gs2, 16
  %ok2cab = and i1 %ok2ca, %ok2cb
  %ok2c_all = and i1 %ok2cab, %ok2cc
  br i1 %ok2c_all, label %t3, label %fail3

t3:
  ; 3. i1 choice & return size == 1, conditioned to true
  %b3 = call i1 @caller_i1_condition(ptr %dummy_obs, ptr null)
  %gs3 = load i64, ptr @last_get_choice_size
  %cs3 = load i64, ptr @last_choice_size
  %rs3 = load i64, ptr @last_ret_size
  %ok3a = icmp eq i64 %gs3, 1
  %ok3b = icmp eq i64 %cs3, 1
  %ok3c = icmp eq i64 %rs3, 1
  %ok3ab = and i1 %b3, %ok3a
  %ok3bc = and i1 %ok3b, %ok3c
  %ok3 = and i1 %ok3ab, %ok3bc
  br i1 %ok3, label %t4, label %fail4

t4:
  ; 4. <2 x double> (16) + <2 x ptr> (16) + { i32 } (4) => sum 36, ret size 4
  store i64 0, ptr @size_sum
  %s4 = call { i32 } @caller_vec_struct(ptr null)
  %s4v = extractvalue { i32 } %s4, 0
  %ss4 = load i64, ptr @size_sum
  %rs4 = load i64, ptr @last_ret_size
  %ok4a = icmp eq i32 %s4v, 42
  %ok4b = icmp eq i64 %ss4, 36
  %ok4c = icmp eq i64 %rs4, 4
  %ok4ab = and i1 %ok4a, %ok4b
  %ok4 = and i1 %ok4ab, %ok4c
  br i1 %ok4, label %t5, label %fail5

t5:
  ; 5. Void generative callee in all 3 modes
  store double 0.0, ptr %out
  call void @caller_void_trace(ptr %out, ptr null)
  %v5a = load double, ptr %out
  store double 0.0, ptr %out
  call void @caller_void_cond(ptr %out, ptr %dummy_obs, ptr null)
  %v5b = load double, ptr %out
  store double 0.0, ptr %out
  store double 0.0, ptr %lik
  call void @caller_void_lik(ptr %out, ptr %lik)
  %v5c = load double, ptr %out
  %l5 = load double, ptr %lik
  %ok5a = fcmp oeq double %v5a, 3.0
  %ok5b = fcmp oeq double %v5b, 11.0
  %ok5c = fcmp oeq double %v5c, 3.0
  %ok5d = fcmp oeq double %l5, 4.0
  %ok5ab = and i1 %ok5a, %ok5b
  %ok5cd = and i1 %ok5c, %ok5d
  %ok5 = and i1 %ok5ab, %ok5cd
  br i1 %ok5, label %t6, label %fail6

t6:
  ; 6. Mutual recursion even_m(2) -> 3.0 + odd_m(1) -> even_m(0) = 3.0
  %r6 = call double @caller_mutual_rec(i32 2, ptr null)
  %ok6 = fcmp oeq double %r6, 3.0
  br i1 %ok6, label %t7, label %fail7

t7:
  ; 7. Populate iface1 and iface2 and call caller_dyn_2 FIRST!
  %i1_0 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 0
  store ptr @__enzyme_get_trace, ptr %i1_0
  %i1_1 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 1
  store ptr @__enzyme_get_choice, ptr %i1_1
  %i1_2 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 2
  store ptr @__enzyme_insert_call, ptr %i1_2
  %i1_3 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 3
  store ptr @dyn1_insert_choice, ptr %i1_3
  %i1_4 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 4
  store ptr @__enzyme_insert_argument, ptr %i1_4
  %i1_5 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 5
  store ptr @__enzyme_insert_return, ptr %i1_5
  %i1_6 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 6
  store ptr @__enzyme_insert_function, ptr %i1_6
  %i1_7 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 7
  store ptr @__enzyme_insert_gradient_choice, ptr %i1_7
  %i1_8 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 8
  store ptr @__enzyme_insert_gradient_argument, ptr %i1_8
  %i1_9 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 9
  store ptr @__enzyme_newtrace, ptr %i1_9
  %i1_10 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 10
  store ptr @__enzyme_freetrace, ptr %i1_10
  %i1_11 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 11
  store ptr @__enzyme_has_call, ptr %i1_11
  %i1_12 = getelementptr [13 x ptr], ptr %iface1, i64 0, i64 12
  store ptr @__enzyme_has_choice, ptr %i1_12

  %i2_0 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 0
  store ptr @__enzyme_get_trace, ptr %i2_0
  %i2_1 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 1
  store ptr @__enzyme_get_choice, ptr %i2_1
  %i2_2 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 2
  store ptr @__enzyme_insert_call, ptr %i2_2
  %i2_3 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 3
  store ptr @dyn2_insert_choice, ptr %i2_3
  %i2_4 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 4
  store ptr @__enzyme_insert_argument, ptr %i2_4
  %i2_5 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 5
  store ptr @__enzyme_insert_return, ptr %i2_5
  %i2_6 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 6
  store ptr @__enzyme_insert_function, ptr %i2_6
  %i2_7 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 7
  store ptr @__enzyme_insert_gradient_choice, ptr %i2_7
  %i2_8 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 8
  store ptr @__enzyme_insert_gradient_argument, ptr %i2_8
  %i2_9 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 9
  store ptr @__enzyme_newtrace, ptr %i2_9
  %i2_10 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 10
  store ptr @__enzyme_freetrace, ptr %i2_10
  %i2_11 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 11
  store ptr @__enzyme_has_call, ptr %i2_11
  %i2_12 = getelementptr [13 x ptr], ptr %iface2, i64 0, i64 12
  store ptr @__enzyme_has_choice, ptr %i2_12

  %r7b = call double @caller_dyn_2(ptr null, ptr %iface2)
  %h1_mid = load i32, ptr @dyn1_hits
  %h2_mid = load i32, ptr @dyn2_hits
  %r7a = call double @caller_dyn_1(ptr null, ptr %iface1)
  %h1_end = load i32, ptr @dyn1_hits
  %h2_end = load i32, ptr @dyn2_hits
  %ok7a = icmp eq i32 %h1_mid, 0
  %ok7b = icmp eq i32 %h2_mid, 1
  %ok7c = icmp eq i32 %h1_end, 1
  %ok7d = icmp eq i32 %h2_end, 1
  %ok7ab = and i1 %ok7a, %ok7b
  %ok7cd = and i1 %ok7c, %ok7d
  %ok7 = and i1 %ok7ab, %ok7cd
  br i1 %ok7, label %t8, label %fail7

t8:
  ; 8. readonly model traced updates likelihood even under -O1
  store double 0.0, ptr %lik
  %r8 = call double @caller_readonly_model(ptr %lik, ptr null)
  %l8 = load double, ptr %lik
  %ok8a = fcmp oeq double %r8, 3.0
  %ok8b = fcmp oeq double %l8, 4.0
  %ok8 = and i1 %ok8a, %ok8b
  br i1 %ok8, label %t9, label %fail8

t9:
  ; 9. fastcc condition + fastcc autodiff
  %r9 = call double @caller_fastcc_cond(ptr %dummy_obs, ptr null)
  store double 0.0, ptr %lik
  store double 1.0, ptr %dlik
  call void @caller_fastcc_autodiff(ptr %lik, ptr %dlik)
  %l9 = load double, ptr %lik
  ; x = fast_sample(3) = 5; score1 = fast_logpdf(3, 5) = 15; score2 = fast_logpdf(5, 4) = 20 => 35
  %ok9a = fcmp oeq double %r9, 4.0
  %ok9b = fcmp oeq double %l9, 35.0
  %ok9 = and i1 %ok9a, %ok9b
  br i1 %ok9, label %t10, label %fail9

t10:
  ; 10. Non-call use of generative function
  %r10 = call double @caller_noncall_use(ptr null)
  %seen = load ptr, ptr @sink_seen
  %ok10a = fcmp oeq double %r10, 3.0
  %ok10b = icmp eq ptr %seen, @simple_model
  %ok10 = and i1 %ok10a, %ok10b
  br i1 %ok10, label %t11, label %fail10

t11:
  ; 11. Vararg sample decl
  store double 0.0, ptr %lik
  %r11 = call double @caller_vararg_sample(ptr %lik, ptr null)
  %l11 = load double, ptr %lik
  %ok11a = fcmp oeq double %r11, 3.0
  %ok11b = fcmp oeq double %l11, 4.0
  %ok11 = and i1 %ok11a, %ok11b
  br i1 %ok11, label %t12, label %fail11

t12:
  ; 12. Loop trace (4 iterations * 3.0 = 12.0)
  %r12 = call double @caller_loop_trace(i32 4, ptr null)
  %ok12 = fcmp oeq double %r12, 12.0
  br i1 %ok12, label %pass, label %fail12

pass:
  ret i32 0
fail1:
  ret i32 1
fail2:
  ret i32 2
fail3:
  ret i32 3
fail4:
  ret i32 4
fail5:
  ret i32 5
fail6:
  ret i32 6
fail7:
  ret i32 7
fail8:
  ret i32 8
fail9:
  ret i32 9
fail10:
  ret i32 10
fail11:
  ret i32 11
fail12:
  ret i32 12
}

; CHECK-LABEL: define double @caller_likelihood_ret(ptr %lik)
; CHECK: call double @likelihood_simple_model(ptr %lik)

; CHECK-LABEL: define double @caller_dyn_1(ptr %tr, ptr %iface1)
; CHECK: %slot = alloca ptr
; CHECK: store ptr %iface1, ptr %slot
; CHECK: %ld = load ptr, ptr %slot
; CHECK: getelementptr inbounds ptr, ptr %ld, i32 0
; CHECK: call double @[[TRACE_DYN1:trace_simple_model[0-9.]*]](ptr %likelihood, ptr %tr)

; CHECK-LABEL: define double @caller_dyn_2(ptr %tr, ptr %iface2)
; CHECK: call double @[[TRACE_DYN2:trace_simple_model[0-9.]*]](ptr %likelihood, ptr %tr)
; CHECK-NOT: call double @[[TRACE_DYN1]](

; CHECK-LABEL: define double @caller_fastcc_cond(ptr %obs, ptr %tr)
; CHECK: call fastcc double @condition_fastcc_model(

; CHECK-LABEL: define void @caller_fastcc_autodiff(ptr %lik, ptr %dlik)
; CHECK: call fastcc void @diffelikelihood_fastcc_model(

; CHECK-LABEL: define double @caller_loop_trace(i32 %n, ptr %tr)
; CHECK: entry:
; CHECK-NEXT: %likelihood = alloca double
; CHECK: loop:
; CHECK-NOT: alloca double

; CHECK-LABEL: define void @caller_active_mu(ptr %lik, ptr %dlik)
; CHECK: call void @[[DIFF_MU:diffelikelihood_two_var_model[0-9.]*]](

; CHECK-LABEL: define void @caller_active_x(ptr %lik, ptr %dlik)
; CHECK-NOT: call void @[[DIFF_MU]](
; CHECK: call void @diffelikelihood_two_var_model

; CHECK-LABEL: define double @caller_dyn_as1(ptr %tr, ptr addrspace(1) %iface)
; CHECK-NOT: addrspacecast
; CHECK-NOT: call addrspace(1)
; CHECK: ret double

; CHECK-LABEL: define internal { double, double } @trace_pair_model(
; CHECK: call void @__enzyme_insert_choice({{.*}}, i64 16)
; CHECK: call void @__enzyme_insert_return({{.*}}, i64 16)

; CHECK-LABEL: define internal { double, double } @condition_pair_model(
; CHECK: call i64 @__enzyme_get_choice({{.*}}, i64 16)
; CHECK: call void @__enzyme_insert_choice({{.*}}, i64 16)
; CHECK: call void @__enzyme_insert_return({{.*}}, i64 16)

; CHECK-LABEL: define internal i1 @condition_i1_model(
; CHECK: call i64 @__enzyme_get_choice({{.*}}, i64 1)
; CHECK: call void @__enzyme_insert_choice({{.*}}, i64 1)
; CHECK: call void @__enzyme_insert_return({{.*}}, i64 1)

; CHECK-LABEL: define internal { i32 } @trace_vec_struct_model(
; CHECK: call void @__enzyme_insert_choice({{.*}}, i64 16)
; CHECK: call void @__enzyme_insert_choice({{.*}}, i64 16)
; CHECK: call void @__enzyme_insert_choice({{.*}}, i64 4)
; CHECK: call void @__enzyme_insert_return({{.*}}, i64 4)

; CHECK-LABEL: define internal double @trace_readonly_model(ptr "enzyme_likelihood" %likelihood, ptr "enzyme_trace" %trace) {

; CHECK-LABEL: define internal fastcc double @condition_fastcc_model(
; CHECK: call fastcc double @fast_sample(
; CHECK: call fastcc double @fast_logpdf(

; CHECK-LABEL: define internal double @trace_model_calls_helper(
; CHECK: call void @helper_takes_ptr()
; CHECK-NOT: trace_helper_takes_ptr
