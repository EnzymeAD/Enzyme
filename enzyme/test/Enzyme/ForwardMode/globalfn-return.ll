; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; The forward derivative of a function called through a pointer returns the
; primal result next to its shadow, since the caller may use both: here %r
; feeds the product and its derivative.

@global = private unnamed_addr constant [1 x ptr] [ptr @square]

define double @square(double %x) {
entry:
  %mul = fmul double %x, %x
  ret double %mul
}

define double @mulglobal(double %x, i64 %idx) {
entry:
  %arrayidx = getelementptr inbounds [1 x ptr], ptr @global, i64 0, i64 %idx
  %fp = load ptr, ptr %arrayidx, align 8
  %r = call double %fp(double %x)
  %m = fmul double %r, %x
  ret double %m
}

define double @derivative(double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(ptr nonnull @mulglobal, double %x, double 1.0, i64 0)
  ret double %0
}

define double @cube(double %x) {
entry:
  %sq = fmul double %x, %x
  %mul = fmul double %sq, %x
  ret double %mul
}

define double @mulselect(double %x, i1 %c) {
entry:
  %fp = select i1 %c, ptr @square, ptr @cube
  %r = call double %fp(double %x)
  %m = fmul double %r, %x
  ret double %m
}

define [2 x double] @derivative2(double %x, i1 %c) {
entry:
  %0 = tail call [2 x double] (...) @__enzyme_fwddiff(ptr nonnull @mulselect, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 2.0, i1 %c)
  ret [2 x double] %0
}

declare double @__enzyme_fwddiff(...)

; CHECK: @"_enzyme_forward_square'" = internal constant ptr @fwddiffesquare
; CHECK: @"_enzyme_forward2_square'" = internal constant ptr @fwddiffe2square
; CHECK: @"_enzyme_forward2_cube'" = internal constant ptr @fwddiffe2cube

; CHECK: define internal double @fwddiffemulglobal(double %x, double %"x'", i64 %idx)
; CHECK:   %[[res:.+]] = call {{(fast )?}}{ double, double } %{{.+}}(double %x, double %"x'")
; CHECK-NEXT:   %[[r:.+]] = extractvalue { double, double } %[[res]], 0
; CHECK-NEXT:   %[[dr:.+]] = extractvalue { double, double } %[[res]], 1
; CHECK-NEXT:   %[[a:.+]] = fmul fast double %[[dr]], %x
; CHECK-NEXT:   %[[b:.+]] = fmul fast double %"x'", %[[r]]
; CHECK-NEXT:   %[[c:.+]] = fadd fast double %[[a]], %[[b]]
; CHECK-NEXT:   ret double %[[c]]

; CHECK: define internal { double, double } @fwddiffesquare(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK:   %[[i0:.+]] = insertvalue { double, double } {{(undef|poison)}}, double %mul, 0
; CHECK-NEXT:   %[[i1:.+]] = insertvalue { double, double } %[[i0]], double %{{.+}}, 1
; CHECK-NEXT:   ret { double, double } %[[i1]]

; CHECK: define internal [2 x double] @fwddiffe2mulselect(double %x, [2 x double] %"x'", i1 %c)
; CHECK:   %[[res2:.+]] = call {{(fast )?}}{ double, [2 x double] } %{{.+}}(double %x, [2 x double] %"x'")
; CHECK:   %[[r2:.+]] = extractvalue { double, [2 x double] } %[[res2]], 0
; CHECK:   %[[dx0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %{{.+}} = fmul fast double %[[dx0]], %[[r2]]

; CHECK: define internal { double, [2 x double] } @fwddiffe2square(double %x, [2 x double] %"x'")
; CHECK:   ret { double, [2 x double] }

; CHECK: define internal { double, [2 x double] } @fwddiffe2cube(double %x, [2 x double] %"x'")
; CHECK:   ret { double, [2 x double] }
