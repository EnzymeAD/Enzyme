; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-loose-types=1 -enzyme-detect-readthrow=0 -enzyme-coalese=1 -enzyme-noalias=1 -enzyme-phi-restructure=1 -enzyme-max-cache=1 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-loose-types=1 -enzyme-detect-readthrow=0 -enzyme-coalese=1 -enzyme-noalias=1 -enzyme-phi-restructure=1 -enzyme-max-cache=1 -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-loose-types=1 -enzyme-detect-readthrow=0 -S | %lli -

declare void @__enzyme_autodiff(...)
declare float @__enzyme_autodiff_f32(...)
declare double @__enzyme_autodiff_f64(...)
declare void @abort()

define { <2 x float>, float } @branch_model() noinline {
entry:
  ret { <2 x float>, float } { <2 x float> zeroinitializer, float poison }
}

define { <2 x float>, float } @runtime_dispatch(i1 %cond) noinline {
entry:
  br i1 %cond, label %merge, label %call_branch

call_branch:
  %branch_value = call { <2 x float>, float } @branch_model()
  br label %merge

merge:
  %state = phi { <2 x float>, float } [ %branch_value, %call_branch ], [ zeroinitializer, %entry ]
  ret { <2 x float>, float } %state
}

; CHECK-LABEL: define internal void @differuntime_dispatch(
; CHECK: invertmerge:
; CHECK:   store { <2 x float>, float } %differeturn, ptr %"state'de"
; CHECK:   fadd fast <2 x float>
; CHECK:   fadd fast float

define void @kernel_body(ptr %out) {
entry:
  %state = call { <2 x float>, float } @runtime_dispatch(i1 false)
  %vec = extractvalue { <2 x float>, float } %state, 0
  %value = extractelement <2 x float> %vec, i64 0
  store float %value, ptr %out, align 4
  ret void
}

define void @kernel_wrapper(i32 %begin, i32 %end, ptr %out, i32 %unused, ptr %shadow) {
entry:
  call void @kernel_body(ptr %out)
  ret void
}

define void @autodiff_entry() {
entry:
  call void (...) @__enzyme_autodiff(ptr @kernel_wrapper, i32 0, i32 0, ptr null, ptr null, i32 0, ptr null, ptr null)
  ret void
}

define { <2 x float>, float } @active_branch_vec(float %x) noinline {
entry:
  %x2 = fmul float %x, 2.000000e+00
  %x3 = fmul float %x, 3.000000e+00
  %v0 = insertelement <2 x float> zeroinitializer, float %x2, i64 0
  %v1 = insertelement <2 x float> %v0, float %x, i64 1
  %s0 = insertvalue { <2 x float>, float } zeroinitializer, <2 x float> %v1, 0
  %s1 = insertvalue { <2 x float>, float } %s0, float %x3, 1
  ret { <2 x float>, float } %s1
}

define { <2 x float>, float } @dispatch_active_vec(i1 %cond, float %x) noinline {
entry:
  br i1 %cond, label %merge, label %call_branch

call_branch:
  %branch_value = call { <2 x float>, float } @active_branch_vec(float %x)
  br label %merge

merge:
  %state = phi { <2 x float>, float } [ %branch_value, %call_branch ], [ zeroinitializer, %entry ]
  ret { <2 x float>, float } %state
}

define float @eval_vec_struct(float %x, i1 %cond) {
entry:
  %state = call { <2 x float>, float } @dispatch_active_vec(i1 %cond, float %x)
  %vec = extractvalue { <2 x float>, float } %state, 0
  %lane0 = extractelement <2 x float> %vec, i64 0
  %tail = extractvalue { <2 x float>, float } %state, 1
  %sum = fadd float %lane0, %tail
  ret float %sum
}

define { i32, float } @active_branch_int_float(float %x, i32 %scale) noinline {
entry:
  %s0 = insertvalue { i32, float } zeroinitializer, i32 %scale, 0
  %s1 = insertvalue { i32, float } %s0, float %x, 1
  ret { i32, float } %s1
}

define { i32, float } @dispatch_int_float(i1 %cond, float %x, i32 %scale) noinline {
entry:
  br i1 %cond, label %merge, label %call_branch

call_branch:
  %branch_value = call { i32, float } @active_branch_int_float(float %x, i32 %scale)
  br label %merge

merge:
  %state = phi { i32, float } [ %branch_value, %call_branch ], [ zeroinitializer, %entry ]
  ret { i32, float } %state
}

; CHECK-LABEL: define internal { float } @diffedispatch_int_float(
; CHECK: invertmerge:
; CHECK:   fadd fast float

define float @eval_int_float(float %x, i1 %cond, i32 %scale) {
entry:
  %state = call { i32, float } @dispatch_int_float(i1 %cond, float %x, i32 %scale)
  %tag = extractvalue { i32, float } %state, 0
  %tag_float = sitofp i32 %tag to float
  %val = extractvalue { i32, float } %state, 1
  %prod = fmul float %val, %tag_float
  ret float %prod
}

define { float, double } @active_branch_mixed_fp(double %x) noinline {
entry:
  %xf = fptrunc double %x to float
  %x4 = fmul double %x, 4.000000e+00
  %s0 = insertvalue { float, double } zeroinitializer, float %xf, 0
  %s1 = insertvalue { float, double } %s0, double %x4, 1
  ret { float, double } %s1
}

define { float, double } @dispatch_mixed_fp(i1 %cond, double %x) noinline {
entry:
  br i1 %cond, label %merge, label %call_branch

call_branch:
  %branch_value = call { float, double } @active_branch_mixed_fp(double %x)
  br label %merge

merge:
  %state = phi { float, double } [ %branch_value, %call_branch ], [ zeroinitializer, %entry ]
  ret { float, double } %state
}

; CHECK-LABEL: define internal { double } @diffedispatch_mixed_fp(
; CHECK: invertmerge:
; CHECK:   fadd fast float
; CHECK:   fadd fast double

define double @eval_mixed_fp(double %x, i1 %cond) {
entry:
  %state = call { float, double } @dispatch_mixed_fp(i1 %cond, double %x)
  %fval = extractvalue { float, double } %state, 0
  %fext = fpext float %fval to double
  %dval = extractvalue { float, double } %state, 1
  %sum = fadd double %fext, %dval
  ret double %sum
}

define i32 @main() {
entry:
  ; When cond is false, eval_vec_struct(x) = 2*x + 3*x = 5*x -> d/dx = 5.0
  %g1 = call float (...) @__enzyme_autodiff_f32(ptr @eval_vec_struct, float 1.500000e+00, i1 false)
  %err1 = fsub float %g1, 5.000000e+00
  %bad1a = fcmp ogt float %err1, 0.0009765625
  %bad1b = fcmp olt float %err1, -0.0009765625
  %bad1 = or i1 %bad1a, %bad1b
  br i1 %bad1, label %fail, label %check2

check2:
  ; When cond is true, eval_vec_struct(x) = 0.0 -> d/dx = 0.0
  %g2 = call float (...) @__enzyme_autodiff_f32(ptr @eval_vec_struct, float 1.500000e+00, i1 true)
  %bad2 = fcmp one float %g2, 0.000000e+00
  br i1 %bad2, label %fail, label %check3

check3:
  ; When cond is false and scale is 7, eval_int_float(x) = 7*x -> d/dx = 7.0
  %g3 = call float (...) @__enzyme_autodiff_f32(ptr @eval_int_float, float 2.000000e+00, i1 false, i32 7)
  %err3 = fsub float %g3, 7.000000e+00
  %bad3a = fcmp ogt float %err3, 0.0009765625
  %bad3b = fcmp olt float %err3, -0.0009765625
  %bad3 = or i1 %bad3a, %bad3b
  br i1 %bad3, label %fail, label %check4

check4:
  ; When cond is false, eval_mixed_fp(x) = float(x) + 4*x -> d/dx = 5.0
  %g4 = call double (...) @__enzyme_autodiff_f64(ptr @eval_mixed_fp, double 2.000000e+00, i1 false)
  %err4 = fsub double %g4, 5.000000e+00
  %bad4a = fcmp ogt double %err4, 0.0009765625
  %bad4b = fcmp olt double %err4, -0.0009765625
  %bad4 = or i1 %bad4a, %bad4b
  br i1 %bad4, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret i32 0
}
