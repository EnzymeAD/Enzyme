; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Shadows of Fortran module variables, declared in a separate module and
; paired with the variables by a table in C:
;
;   module mo_cfg                      ! existing code, unchanged
;     real(8) :: divdamp_fac = 0.0025d0
;   end module
;   module mo_ad_shadows               ! new file
;     real(8) :: divdamp_fac = 0d0
;   end module
;
;   extern char _QMmo_cfgEdivdamp_fac, _QMmo_ad_shadowsEdivdamp_fac;
;   void *__enzyme_shadow_globals[][2] = {
;     {&_QMmo_cfgEdivdamp_fac, &_QMmo_ad_shadowsEdivdamp_fac}};
;
; Without the table, forward mode replaces the shadow of such a global (an
; external global with an initializer, as flang emits module variables) by
; a zeroed local, and a seed stored in the shadow is lost.

@_QMmo_cfgEdivdamp_fac = dso_local local_unnamed_addr global double 2.500000e-03, align 8
@_QMmo_ad_shadowsEdivdamp_fac = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@__enzyme_shadow_globals = dso_local global [1 x [2 x ptr]] [[2 x ptr] [ptr @_QMmo_cfgEdivdamp_fac, ptr @_QMmo_ad_shadowsEdivdamp_fac]], align 16

define double @scale(double %x) {
entry:
  %f = load double, ptr @_QMmo_cfgEdivdamp_fac, align 8
  %mul = fmul double %f, %x
  ret double %mul
}

define double @dscale(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @scale, double %x, double %dx)
  ret double %r
}

declare double @__enzyme_fwddiff(...)

; CHECK-NOT: @__enzyme_shadow_globals
; CHECK: @_QMmo_cfgEdivdamp_fac = dso_local local_unnamed_addr global double 2.500000e-03, align 8, !enzyme_shadow ![[md:[0-9]+]]
; CHECK-NOT: @__enzyme_shadow_globals

; CHECK: define internal double @fwddiffescale(double %x, double %"x'")
; CHECK: load double, ptr @_QMmo_ad_shadowsEdivdamp_fac

; CHECK: ![[md]] = !{ptr @_QMmo_ad_shadowsEdivdamp_fac}
