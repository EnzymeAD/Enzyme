; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm" -S | FileCheck %s

; preserve-nvvm marks libdevice math wrappers noinline so Enzyme can match
; them by name. A wrapper that is a single forwarding call plus a return must
; inherit the callee's attributes (notably speculatable and memory(none)),
; since inlining will no longer supply them and FunctionAttrs never infers
; speculatable.

define double @__nv_fabs(double %x) #0 {
  %r = call double @llvm.nvvm.fabs.f64(double %x)
  ret double %r
}

; llvm.nvvm.sqrt.* is not declared speculatable upstream, but is treated as
; such here since it has no side effects or UB.
define double @__nv_sqrt(double %x) #0 {
  %r = call double @llvm.nvvm.sqrt.rn.d(double %x)
  ret double %r
}

define float @__nv_sqrtf(float %x) #0 {
  %r = call float @llvm.nvvm.sqrt.rn.ftz.f(float %x)
  ret float %r
}

; A wrapper that does not forward its arguments must not be attributed.
define double @__nv_fmax(double %x, double %y) #0 {
  %r = call double @llvm.nvvm.fmax.d(double %y, double %x)
  ret double %r
}

; A wrapper with extra work must not be attributed.
define double @__nv_cbrt(double %x) #0 {
  %r = call double @llvm.nvvm.sqrt.rn.d(double %x)
  %s = fadd double %r, 1.0
  ret double %s
}

declare double @llvm.nvvm.fabs.f64(double) #1
declare double @llvm.nvvm.fmax.d(double, double) #1
declare double @llvm.nvvm.sqrt.rn.d(double) #2
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #2

attributes #0 = { alwaysinline nounwind }
attributes #1 = { nofree nosync nounwind speculatable willreturn readnone }
attributes #2 = { nofree nosync nounwind willreturn readnone }

; CHECK: define double @__nv_fabs(double %x) #[[FABS:[0-9]+]]
; CHECK: define double @__nv_sqrt(double %x) #[[SQRT:[0-9]+]]
; CHECK: define float @__nv_sqrtf(float %x) #[[SQRTF:[0-9]+]]
; CHECK: define double @__nv_fmax(double %x, double %y) #[[FMAX:[0-9]+]]
; CHECK: define double @__nv_cbrt(double %x) #[[CBRT:[0-9]+]]
; CHECK: attributes #[[FABS]] = { {{.*}}noinline{{.*}}speculatable{{.*}}"enzyme_math"="fabs"
; CHECK: attributes #[[SQRT]] = { {{.*}}noinline{{.*}}speculatable{{.*}}"enzyme_math"="sqrt"
; CHECK: attributes #[[SQRTF]] = { {{.*}}noinline{{.*}}speculatable{{.*}}"enzyme_math"="sqrtf"
; CHECK: attributes #[[FMAX]] = { noinline nounwind "enzyme_math"="fmax"
; CHECK: attributes #[[CBRT]] = { noinline nounwind "enzyme_math"="cbrt"
