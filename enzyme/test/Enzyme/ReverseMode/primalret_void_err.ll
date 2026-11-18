; RUN: if [ %llvmver -lt 16 ]; then not %opt < %s %loadEnzyme -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: not %opt < %s %newLoadEnzyme -passes="enzyme" -S 2>&1 | FileCheck %s

; Requesting enzyme_primal_return on a void-returning function in reverse mode
; must emit a structured diagnostic instead of creating an alloca of void type.

define void @scale(double* %x, double* %y) {
entry:
  %v = load double, double* %x, align 8
  %s = fmul double %v, %v
  store double %s, double* %y, align 8
  ret void
}

declare void @__enzyme_autodiff(...)

define void @dscale(double* %x, double* %dx, double* %y, double* %dy) {
entry:
  call void (...) @__enzyme_autodiff(void (double*, double*)* @scale, metadata !"enzyme_primal_return", metadata !"enzyme_dup", double* %x, double* %dx, metadata !"enzyme_dupnoneed", double* %y, double* %dy)
  ret void
}

; CHECK: Enzyme: Requested primal result of void-returning function type void ({{.*}}) scale
