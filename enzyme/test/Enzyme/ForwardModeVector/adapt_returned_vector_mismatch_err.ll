; RUN: if [ %llvmver -ge 15 ]; then not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S 2>&1 | FileCheck %s; fi

define double @f(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

declare { double, [3 x double] } @__enzyme_fwddiff(...)

define { double, [3 x double] } @test(double %x, double %dx1, double %dx2, double %dx3) {
  %r = call { double, [3 x double] } (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_width", i64 3, metadata !"enzyme_dup", double %x, double %dx1, double %dx2, double %dx3)
  ret { double, [3 x double] } %r
}

; CHECK: Enzyme: Cannot cast return type of gradient [3 x double]
; CHECK-SAME: to desired type { double, [3 x double] }
