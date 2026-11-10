; A pre-link pipeline that defers differentiation to the link compiles a unit as
; if a derivative another unit asks for may need its code, even when the unit
; itself asks nothing of Enzyme. Here f lives in a unit without a request and
; calls a stand-in for libdevice's sin, whose body is a polynomial: the
; derivative has to come from the rule for sin (a call to cos), which needs the
; call to be kept and recognized in the pre-link run, and not from the body.

; REQUIRES: nvptx-registered-target
; RUN: rm -rf %t && split-file %s %t
; RUN: if [ %llvmver -ge 20 ]; then %opt %loadLLDEnzymeOpt -passes="lto-pre-link<O2>" %t/def.ll -o %t/def.bc && %opt %loadLLDEnzymeOpt -passes="lto-pre-link<O2>" %t/use.ll -o %t/use.bc && llvm-link %t/use.bc %t/def.bc -o %t/linked.bc && %opt %loadLLDEnzymeOpt -passes="lto<O2>" %t/linked.bc -S | FileCheck %s; fi

; ThinLTO through the same phases, with the two units merged by llvm-link in
; place of a thin link.
; RUN: if [ %llvmver -ge 20 ]; then %opt %loadLLDEnzymeOpt -passes="thinlto-pre-link<O2>" %t/def.ll -o %t/def.thin.bc && %opt %loadLLDEnzymeOpt -passes="thinlto-pre-link<O2>" %t/use.ll -o %t/use.thin.bc && llvm-link %t/use.thin.bc %t/def.thin.bc -o %t/linked.thin.bc && %opt %loadLLDEnzymeOpt -passes="thinlto<O2>" %t/linked.thin.bc -S | FileCheck %s; fi

; CHECK-LABEL: define double @df(
; CHECK: call fast double @llvm.cos.f64(double %x)
; CHECK: ret double

;--- def.ll
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define internal double @__nv_sin(double %x) {
  %x2 = fmul double %x, %x
  %x3 = fmul double %x2, %x
  %t = fdiv double %x3, 6.000000e+00
  %r = fsub double %x, %t
  ret double %r
}

define double @f(double %x) {
  %s = call double @__nv_sin(double %x)
  %r = fmul double %s, %x
  ret double %r
}

;--- use.ll
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @f(double)

define double @df(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

declare double @__enzyme_autodiff(...)
