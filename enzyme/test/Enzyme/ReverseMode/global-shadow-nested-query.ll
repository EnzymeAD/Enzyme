; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S -o %t.ll && %lli %t.ll; fi

; Forward over reverse with one context per level. The inner gradient of g
; lives in g's inner shadow, and its tangent, the mixed term d2f/dg dx, in
; the outer shadow of that shadow, which a nested query reaches. For
; f = g x^3 at x = 2, g = 2: d2f/dx2 = 6 g x = 24, df/dg = x^3 = 8 and
; d2f/dg dx = 3 x^2 = 12.

@g = global double 2.000000e+00, align 8
@inner = private constant i32 1, !enzyme_context !0
@outer = private constant i32 1, !enzyme_context !0
@enzyme_context = external global i32

declare double @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)
declare ptr @__enzyme_shadow(ptr, ptr, i32)

define double @f(double %x) {
entry:
  %g = load double, ptr @g, align 8
  %x2 = fmul double %x, %x
  %x3 = fmul double %x2, %x
  %r = fmul double %g, %x3
  ret double %r
}

define double @df(double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, ptr @enzyme_context, ptr @inner, double %x)
  ret double %r
}

define i32 @main() {
entry:
  %dg = call ptr @__enzyme_shadow(ptr @inner, ptr @g, i32 0)
  %ddg = call ptr @__enzyme_shadow(ptr @outer, ptr %dg, i32 0)
  store double 0.0, ptr %dg, align 8
  store double 0.0, ptr %ddg, align 8
  %ddx = call double (...) @__enzyme_fwddiff(ptr @df, ptr @enzyme_context, ptr @outer, double 2.0, double 1.0)
  %vdg = load double, ptr %dg, align 8
  %vddg = load double, ptr %ddg, align 8
  %c1 = fcmp une double %ddx, 2.400000e+01
  %c2 = fcmp une double %vdg, 8.000000e+00
  %c3 = fcmp une double %vddg, 1.200000e+01
  %c12 = or i1 %c1, %c2
  %bad = or i1 %c12, %c3
  %ret = zext i1 %bad to i32
  ret i32 %ret
}

!0 = !{}

; CHECK: @g.ad.inner.ad.outer = private global double 0.000000e+00
; CHECK: @g.ad.inner = private global double 0.000000e+00, align 8, !enzyme_shadows ![[innershadows:[0-9]+]]
; CHECK: @g = global double 2.000000e+00, align 8, !enzyme_shadows

; CHECK: define i32 @main()
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double 0.000000e+00, ptr @g.ad.inner, align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr @g.ad.inner.ad.outer, align 8

; CHECK: ![[innershadows]] = !{![[entry:[0-9]+]]}
; CHECK: ![[entry]] = !{ptr @outer, ptr @g.ad.inner.ad.outer}
