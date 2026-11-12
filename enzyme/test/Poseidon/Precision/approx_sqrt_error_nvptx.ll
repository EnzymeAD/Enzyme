; RUN: rm -rf %t && mkdir -p %t
; RUN: sed '/^target triple/d; s/"sm_120"/"x86-64"/' %s > %t/host.ll
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/approx_sqrt -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/nvptx -poseidon-print "-poseidon-apply-rewrites=PT0=Funcs FP64(0%) + FP32(100%)" -S 2>&1 | FileCheck --check-prefix=NVPTX %s
; RUN: %opt < %t/host.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/approx_sqrt -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/host -poseidon-print "-poseidon-apply-rewrites=PT0=Funcs FP64(0%) + FP32(100%)" -S 2>&1 | FileCheck --check-prefix=HOST %s

; sqrt(x*x + y*y) over x, y in [1, 2], once for NVPTX and once for the host.
; Retyped to FP32, the sqrt call is the approximate square root on NVPTX and
; sqrtf on the host, so the same candidates are modelled with that root's
; error bound on NVPTX only: about 1.86e-7 of a root near 2.1, weighted by the
; gradient sum 100, adds 4e-5 to the accuracy cost of every candidate that
; retypes the sqrt, and nothing to one that keeps it in FP64.

target triple = "nvptx64-nvidia-cuda"

define double @tester(double %x, double %y) #0 {
entry:
  %xx = fmul double %x, %x
  %yy = fmul double %y, %y
  %s = fadd double %xx, %yy
  %r = call double @sqrt(double %s)
  ret double %r
}

define double @test_opt(double %x, double %y) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @sqrt(double)
declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="sm_120" }

; NVPTX: 4.946043e-05 {{.*}} All FP64(0%) + FP32(100%)
; NVPTX: 3.922927e-06 {{.*}} All FP64(30%) + FP32(70%)
; NVPTX: 5.005415e-05 {{.*}} Funcs FP64(0%) + FP32(100%)
; NVPTX-LABEL: define {{.*}} @test_opt(
; NVPTX: call float @llvm.nvvm.sqrt.approx.f(

; HOST: 6.096452e-06 {{.*}} All FP64(0%) + FP32(100%)
; HOST: 3.922927e-06 {{.*}} All FP64(30%) + FP32(70%)
; HOST: 5.263638e-06 {{.*}} Funcs FP64(0%) + FP32(100%)
; HOST-LABEL: define {{.*}} @test_opt(
; HOST: call float @sqrtf(
