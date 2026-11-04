; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S 2>&1 | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S 2>&1 | FileCheck %s

; Clang 20 and later attach the TBAA tag of errno (an integer access) to
; calls of libm functions such as tanh and sqrt. The tag describes memory the
; callee may write, not the returned double, so type analysis must consume it
; silently instead of reporting an unknown TBAA call.

; CHECK-NOT: unknown tbaa call instruction

declare double @tanh(double) #0
declare double @sqrt(double) #0
declare double @__enzyme_autodiff(...)

define double @activation(double %x) {
entry:
  %t = call double @tanh(double %x), !tbaa !4
  %s = call double @sqrt(double %x), !tbaa !4
  %r = fmul double %t, %s
  ret double %r
}

; CHECK-LABEL: define double @caller(
define double @caller(double %x) {
entry:
  ; CHECK: call { double } @diffeactivation(double %x, double 1.000000e+00)
  %d = call double (...) @__enzyme_autodiff(ptr @activation, double %x)
  ret double %d
}

; CHECK-LABEL: define internal { double } @diffeactivation(
; CHECK: call fast double @sqrt(double %x)
; CHECK: call fast double @tanh(double %x)

attributes #0 = { nounwind }

!0 = !{!"Simple C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!"int", !1, i64 0}
!4 = !{!2, !2, i64 0}
