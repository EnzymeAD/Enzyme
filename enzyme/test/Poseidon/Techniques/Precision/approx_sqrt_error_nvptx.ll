; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/approx_sqrt \
; DEFINE:   -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-precision-expansions=false \
; DEFINE:   -poseidon-precision-three-tier=false -poseidon-print "-poseidon-pin=P0=Funcs FP64(0%) + FP32(100%)" -S
; RUN: rm -rf %t && mkdir %t
; RUN: sed -e '/^target triple/d' -e 's/"target-cpu"="sm_120"/"target-cpu"="x86-64"/' %s > %t/host.ll
; RUN: %{opt} < %s -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/nvptx 2>&1 | FileCheck --check-prefix=NVPTX %s
; RUN: %{opt} < %t/host.ll -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/host 2>&1 | FileCheck --check-prefix=HOST %s

; sqrt(x*x + y*y) over x, y in [1, 2], for NVPTX and for the host (the module
; without its triple). Retyped to FP32 the root is the approximate square root
; on NVPTX and sqrtf on the host, so only on NVPTX does every candidate that
; retypes it carry that root's error bound (about 4e-5 of accuracy cost).

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
