; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

@ext = global double 3.0, align 8, !enzyme_shadow !0
@dext = global double 0.0, align 8
!0 = !{ptr @dext}

@cyc_act_a = internal constant { ptr, ptr } { ptr @cyc_act_b, ptr @ext }, align 8
@cyc_act_b = internal constant ptr @cyc_act_a, align 8

@self_mut = internal global ptr @self_mut, align 8

declare double @__enzyme_autodiff(...)

define double @f_cyc_active(double %x) {
entry:
  %p = load ptr, ptr @cyc_act_b, align 8
  %gep = getelementptr { ptr, ptr }, ptr %p, i64 0, i32 1
  %ep = load ptr, ptr %gep, align 8
  %v = load double, ptr %ep, align 8
  %r = fmul double %x, %v
  ret double %r
}

define double @f_self_mut(double %x) {
entry:
  %a = alloca double, align 8
  store double %x, ptr %a, align 8
  store ptr %a, ptr @self_mut, align 8
  %p = load ptr, ptr @self_mut, align 8
  %v = load double, ptr %p, align 8
  ret double %v
}

define double @df_cyc_active(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f_cyc_active, double %x)
  ret double %r
}

define double @df_self_mut(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f_self_mut, double %x)
  ret double %r
}

; CHECK: @cyc_act_a_shadow = internal constant { ptr, ptr } { ptr @cyc_act_b_shadow, ptr @dext }, align 8
; CHECK: @cyc_act_b_shadow = internal constant ptr @cyc_act_a_shadow, align 8
; CHECK: @self_mut_shadow = internal global ptr @self_mut_shadow, align 8

; CHECK-LABEL: define internal { double } @diffef_cyc_active(double %x, double %differeturn)
; CHECK: %"p'ipl" = load ptr, ptr @cyc_act_b_shadow, align 8
; CHECK: %"gep'ipg" = getelementptr { ptr, ptr }, ptr %"p'ipl", i64 0, i32 1
; CHECK: %"ep'ipl" = load ptr, ptr %"gep'ipg", align 8
; CHECK: %[[OLD:[0-9]+]] = load double, ptr %"ep'ipl", align 8
; CHECK: %[[NEW:[0-9]+]] = fadd fast double %[[OLD]],
; CHECK: store double %[[NEW]], ptr %"ep'ipl", align 8

; CHECK-LABEL: define internal { double } @diffef_self_mut(double %x, double %differeturn)
; CHECK: store ptr %"a'ipa", ptr @self_mut_shadow, align 8
; CHECK: %"p'ipl" = load ptr, ptr @self_mut_shadow, align 8
