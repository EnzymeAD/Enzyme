; RUN: rm -rf %t && mkdir -p %t/cache %t/nocm
; RUN: cp %S/Inputs/fma_opt_cache/cachedHerbieOutput_default_f9f2bd660c90d989_0_0.txt %t/cache/cachedHerbieOutput_cuda-sm120_f9f2bd660c90d989_0_0.txt
; RUN: cp %S/Inputs/fma_opt_cache/cachedHerbieOutput_default_f9f2bd660c90d989_0_0.txt.input %t/cache/cachedHerbieOutput_cuda-sm120_f9f2bd660c90d989_0_0.txt.input
; RUN: sed "s/sm_120/sm_90/" %s > %t/sm90.ll
; RUN: cp %S/Inputs/cm_gpu_fixture_x1e6.csv %t/nocm/
; RUN: cp -r %t/cache %t/stale && echo "cuda-sm120 0000000000000000" > %t/stale/cachedHerbieOutput_cuda-sm120_f9f2bd660c90d989_0_0.txt.platform

; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/cache -S 2>&1 | FileCheck --check-prefix=MATCH %s
; RUN: %opt < %t/sm90.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_sm90_x1e6.csv -poseidon-cache=%t/cache -S 2>&1 | FileCheck --check-prefix=OTHERARCH %s
; RUN: (%opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/nocm/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/nocm-cache -S 2>&1 || true) | FileCheck --check-prefix=NOPLATFORM %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/stale -S 2>&1 | FileCheck --check-prefix=RECALIBRATED %s
; REQUIRES: poseidon

; Herbie ranks its rewrite candidates by a platform cost table, so a cached
; result belongs to the platform it was searched under and its cache entry is
; named after it. Four runs over one seeded entry, keyed cuda-sm120:
;   MATCH        the same architecture reads it,
;   OTHERARCH    an sm_90 cost model does not, and runs Herbie instead (which
;                here cannot start, and says so, rather than silently
;                replaying the other device's answer),
;   NOPLATFORM   a CUDA cost model with no platform file beside it aborts the
;                compile naming the file and the command that writes it,
;   RECALIBRATED the same platform NAME with a different cost digest is reused
;                and reported.
; The Herbie binary is deliberately absent: no arm of this test may need it.

; ModuleID = 'herbie_platform_key.cu'
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

; MATCH: Using cached Herbie output from {{.*}}cachedHerbieOutput_cuda-sm120_f9f2bd660c90d989_0_0.txt
; MATCH: define double @preprocess_tester(double %x, double %y, double %z)

; OTHERARCH-NOT: Using cached Herbie output
; OTHERARCH: [poseidon] Herbie subgraph 0 of preprocess_tester
; OTHERARCH: Execution failed

; NOPLATFORM: LLVM ERROR: Poseidon: the Herbie algebraic search needs the platform generated from this device's cost model, and {{.*}}cm_gpu_fixture_x1e6.herbie.rkt does not exist. Generate it with 'poseidon-calibrate --only herbie-platform --out {{.*}}cm_gpu_fixture_x1e6.csv'

; RECALIBRATED: WARNING: cached Herbie output {{.*}}cachedHerbieOutput_cuda-sm120_f9f2bd660c90d989_0_0.txt was searched under platform 'cuda-sm120' with cost digest 0000000000000000
; RECALIBRATED: Using cached Herbie output from {{.*}}cachedHerbieOutput_cuda-sm120_f9f2bd660c90d989_0_0.txt
