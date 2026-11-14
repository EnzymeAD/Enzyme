; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -preserve-nvvm -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme" -enzyme-preopt=false -S | FileCheck %s

; Verify that preserve-nvvm turns an __enzyme_shadow_global pair into
; !enzyme_shadow metadata, keeps both globals in @llvm.compiler.used, and
; reads the declared shadow for an external initialized global in forward mode.

; CHECK-NOT: @__enzyme_shadow_global_divdamp_fac
; CHECK: @_QMmo_cfgEdivdamp_fac = global double 2.500000e-03, !enzyme_shadow ![[MD:.+]]
; CHECK-NOT: @__enzyme_shadow_global_divdamp_fac
; CHECK: @llvm.compiler.used = appending global [2 x ptr] [ptr @_QMmo_ad_shadowsEdivdamp_fac, ptr @_QMmo_cfgEdivdamp_fac], section "llvm.metadata"
@_QMmo_cfgEdivdamp_fac = global double 2.500000e-03
@_QMmo_ad_shadowsEdivdamp_fac = global double 0.000000e+00
@__enzyme_shadow_global_divdamp_fac = global [2 x ptr] [ptr @_QMmo_cfgEdivdamp_fac, ptr @_QMmo_ad_shadowsEdivdamp_fac]

define double @scale(double %x) {
entry:
  %f = load double, ptr @_QMmo_cfgEdivdamp_fac
  %mul = fmul double %f, %x
  ret double %mul
}

; CHECK-LABEL: define internal double @fwddiffescale(
; CHECK-SAME: double %{{.+}}, double %{{.+}})
; CHECK: load double, ptr @_QMmo_ad_shadowsEdivdamp_fac
define double @dscale(double %x, double %dx) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @scale, double %x, double %dx)
  ret double %r
}

declare double @__enzyme_fwddiff(...)

; CHECK: ![[MD]] = !{ptr @_QMmo_ad_shadowsEdivdamp_fac}
