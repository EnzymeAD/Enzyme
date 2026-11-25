; DEFINE: %{opt} = %loadPoseidon -poseidon-profile-use=%S/../../Inputs/fma_site_profile \
; DEFINE:   -poseidon-enable-rewrite=true -poseidon-herbie-binary=/nonexistent/herbie \
; DEFINE:   -poseidon-enable-precision=false -poseidon-print -S
; RUN: rm -rf %t && split-file %s %t

; Herbie ranks its rewrite candidates by a platform cost table, so a cached
; result belongs to the platform it was searched under and its cache entry is
; named after it. The Herbie binary is absent: no compile here may need it.

; On an NVIDIA GPU, over one entry seeded under cuda-sm120, the same
; architecture reads it (NV-MATCH); an sm_90 cost model looks under its own
; name, finds no answer and, with no Herbie to compute one, stops rather than
; replaying the other device's answer (NV-OTHERARCH); a CUDA cost model with no platform file
; beside it stops the compile naming the file and the command that writes it
; (NV-NOPLATFORM); the same platform name with other costs does not read it
; either, since the platform's contents are part of the key (NV-RECALIBRATED).
; RUN: mkdir %t/nv %t/nv/cache %t/nv/nocm %t/nv/recal
; RUN: cp %S/../../Inputs/fma_opt_cache/herbie_default_958ABA9C1BC0AE00.json %t/nv/cache/herbie_cuda-sm120_45B4DF20BF2EFFF8.json
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/nv/nocm/
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %cost_models/cm_sm_120_synthetic.herbie.rkt %t/nv/recal/ && echo ";; recalibrated" >> %t/nv/recal/cm_sm_120_synthetic.herbie.rkt
; RUN: %opt < %t/kernel.ll %{opt} -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/nv/cache 2>&1 | FileCheck --check-prefix=NV-MATCH %s
; RUN: not %opt < %t/sm90.ll %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%cost_models/cm_sm_90_synthetic.csv -poseidon-cache=%t/nv/cache 2>&1 | FileCheck --check-prefix=NV-OTHERARCH %s
; RUN: not %opt < %t/kernel.ll %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/nv/nocm/cm_sm_120_synthetic.csv -poseidon-cache=%t/nv/nocm-cache 2>&1 | FileCheck --check-prefix=NV-NOPLATFORM %s
; RUN: not %opt < %t/kernel.ll %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/nv/recal/cm_sm_120_synthetic.csv -poseidon-cache=%t/nv/cache 2>&1 | FileCheck --check-prefix=NV-RECALIBRATED %s

; NV-MATCH: [poseidon] Using cached Herbie output from {{.*}}herbie_cuda-sm120_45B4DF20BF2EFFF8.json
; NV-MATCH: define double @preprocess_tester(double %x, double %y, double %z)

; NV-OTHERARCH-NOT: [poseidon] Using cached Herbie output
; NV-OTHERARCH: Poseidon: no Herbie answer for subgraph 0 of preprocess_tester is cached in {{.*}}cache/herbie_cuda-sm90_{{[0-9A-F]+}}.json, and there is no Herbie at /nonexistent/herbie to compute it;

; NV-NOPLATFORM: error: {{.*}}Poseidon: the Herbie platform of this device's cost model, {{.*}}cm_sm_120_synthetic.herbie.rkt, does not exist; generate it with 'poseidon-calibrate --only herbie-platform --out {{.*}}cm_sm_120_synthetic.csv'

; NV-RECALIBRATED-NOT: [poseidon] Using cached Herbie output
; NV-RECALIBRATED: Poseidon: no Herbie answer for subgraph 0 of preprocess_tester is cached in {{.*}}cache/herbie_cuda-sm120_{{[0-9A-F]+}}.json, and there is no Herbie at /nonexistent/herbie to compute it;

; AMD GPU, the module retargeted to gfx942: poseidon-calibrate generates the
; Herbie platform of an AMD GPU's cost model (hip-gfx942, hip-gfx90a) beside it,
; a compile against it searches under that platform and keys its Herbie cache
; by it (AMD-MATCH reads an entry seeded under hip-gfx942), and without the
; platform file the compile stops rather than search under Herbie's default
; costs (AMD-NOPLATFORM). The cost models are the sm_120 fixture with its
; native_arch rewritten. Needs the AMDGPU target.
; RUN: %if amdgpu-registered-target %{ mkdir %t/amd %t/amd/cm %t/amd/nocm %t/amd/cache %}
; RUN: %if amdgpu-registered-target %{ sed 's/^# native_arch=sm_120$/# native_arch=gfx942/' %cost_models/cm_sm_120_synthetic.csv > %t/amd/cm/cm_gfx942_Fixture.csv %}
; RUN: %if amdgpu-registered-target %{ cp %t/amd/cm/cm_gfx942_Fixture.csv %t/amd/nocm/ %}
; RUN: %if amdgpu-registered-target %{ %poseidon_calibrate --only herbie-platform --out %t/amd/cm/cm_gfx942_Fixture.csv | FileCheck --check-prefix=AMD-CALIBRATE %s %}
; RUN: %if amdgpu-registered-target %{ FileCheck --check-prefix=AMD-PLATFORM %s < %t/amd/cm/cm_gfx942_Fixture.herbie.rkt %}
; RUN: %if amdgpu-registered-target %{ sed 's/^# native_arch=sm_120$/# native_arch=gfx90a/' %cost_models/cm_sm_120_synthetic.csv > %t/amd/cm/cm_gfx90a_Fixture.csv %}
; RUN: %if amdgpu-registered-target %{ %poseidon_calibrate --only herbie-platform --out %t/amd/cm/cm_gfx90a_Fixture.csv > /dev/null %}
; RUN: %if amdgpu-registered-target %{ FileCheck --check-prefix=AMD-PLATFORM90A %s < %t/amd/cm/cm_gfx90a_Fixture.herbie.rkt %}
; RUN: %if amdgpu-registered-target %{ cp %S/../../Inputs/fma_opt_cache/herbie_default_958ABA9C1BC0AE00.json %t/amd/cache/herbie_hip-gfx942_9E5CC014DF84A1D1.json %}
; RUN: %if amdgpu-registered-target %{ %opt < %t/amdgpu.ll %{opt} -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-cost-model=%t/amd/cm/cm_gfx942_Fixture.csv -poseidon-cache=%t/amd/cache 2>&1 | FileCheck --check-prefix=AMD-MATCH %s %}
; RUN: %if amdgpu-registered-target %{ not %opt < %t/amdgpu.ll %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/amd/nocm/cm_gfx942_Fixture.csv -poseidon-cache=%t/amd/nocm-cache 2>&1 | FileCheck --check-prefix=AMD-NOPLATFORM %s %}

; AMD-CALIBRATE: poseidon-calibrate: wrote {{.*}}cm_gfx942_Fixture.herbie.rkt

; AMD-PLATFORM: ;; HIP gfx942 platform, generated from a Poseidon GPU cost model CSV.
; AMD-PLATFORM: ;; Target device: Fixture
; AMD-PLATFORM: (module hip-gfx942

; AMD-PLATFORM90A: ;; HIP gfx90a platform, generated from a Poseidon GPU cost model CSV.
; AMD-PLATFORM90A: (module hip-gfx90a

; AMD-MATCH: [poseidon] Using cached Herbie output from {{.*}}herbie_hip-gfx942_9E5CC014DF84A1D1.json
; AMD-MATCH: define double @preprocess_tester(double %x, double %y, double %z)

; AMD-NOPLATFORM: error: {{.*}}Poseidon: the Herbie platform of this device's cost model, {{.*}}nocm/cm_gfx942_Fixture.herbie.rkt, does not exist; generate it with 'poseidon-calibrate --only herbie-platform --out {{.*}}nocm/cm_gfx942_Fixture.csv'

;--- kernel.ll
source_filename = "herbie_platform_key.cu"
target triple = "nvptx64-nvidia-cuda"

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="sm_120" }

;--- sm90.ll
source_filename = "herbie_platform_key.cu"
target triple = "nvptx64-nvidia-cuda"

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="sm_90" }

;--- amdgpu.ll
source_filename = "herbie_platform_key.hip"
target triple = "amdgcn-amd-amdhsa"

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="gfx942" }
