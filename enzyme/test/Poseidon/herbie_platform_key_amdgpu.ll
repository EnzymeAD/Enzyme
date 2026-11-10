; RUN: rm -rf %t && mkdir -p %t/cm %t/nocm %t/cache
; RUN: sed 's/^# native_arch=sm_120$/# native_arch=gfx942/' %S/Inputs/cm_gpu_fixture_x1e6.csv > %t/cm/cm_gfx942_Fixture.csv
; RUN: cp %t/cm/cm_gfx942_Fixture.csv %t/nocm/
; RUN: %poseidon_calibrate --only herbie-platform --out %t/cm/cm_gfx942_Fixture.csv | FileCheck --check-prefix=CALIBRATE %s
; RUN: FileCheck --check-prefix=PLATFORM %s < %t/cm/cm_gfx942_Fixture.herbie.rkt
; RUN: sed 's/^# native_arch=sm_120$/# native_arch=gfx90a/' %S/Inputs/cm_gpu_fixture_x1e6.csv > %t/cm/cm_gfx90a_Fixture.csv
; RUN: %poseidon_calibrate --only herbie-platform --out %t/cm/cm_gfx90a_Fixture.csv > /dev/null
; RUN: FileCheck --check-prefix=PLATFORM90A %s < %t/cm/cm_gfx90a_Fixture.herbie.rkt
; RUN: cp %S/Inputs/fma_opt_cache/cachedHerbieOutput_default_06F0ECDBFA1A458D.txt %t/cache/cachedHerbieOutput_hip-gfx942_E38949A1AAABB9CA.txt

; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/cm/cm_gfx942_Fixture.csv -poseidon-cache=%t/cache -S 2>&1 | FileCheck --check-prefix=MATCH %s
; RUN: (%opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%t/nocm/cm_gfx942_Fixture.csv -poseidon-cache=%t/nocm-cache -S 2>&1 || true) | FileCheck --check-prefix=NOPLATFORM %s
; REQUIRES: poseidon, amdgpu-registered-target

; An AMD GPU's cost model is a GPU's: poseidon-calibrate generates its Herbie
; platform (named hip-gfx942) beside it, a compile against it searches under
; that platform and keys its Herbie cache by it (MATCH reads an entry seeded
; under hip-gfx942), and without the platform file the compile aborts rather
; than search under Herbie's default costs (NOPLATFORM). The model is the
; sm_120 lit fixture with its native_arch rewritten (also to gfx90a, whose
; platform is hip-gfx90a); the Herbie binary is deliberately absent.

; ModuleID = 'herbie_platform_key.hip'
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

; CALIBRATE: [calibrate] wrote {{.*}}cm_gfx942_Fixture.herbie.rkt

; PLATFORM: ;; HIP gfx942 platform, generated from a Poseidon GPU cost model CSV.
; PLATFORM: ;; Target device: Fixture
; PLATFORM: (module hip-gfx942

; PLATFORM90A: ;; HIP gfx90a platform, generated from a Poseidon GPU cost model CSV.
; PLATFORM90A: (module hip-gfx90a

; MATCH: Using cached Herbie output from {{.*}}cachedHerbieOutput_hip-gfx942_E38949A1AAABB9CA.txt
; MATCH: define double @preprocess_tester(double %x, double %y, double %z)

; NOPLATFORM: LLVM ERROR: Poseidon: the Herbie algebraic search needs the platform generated from this device's cost model, and {{.*}}nocm/cm_gfx942_Fixture.herbie.rkt does not exist. Generate it with 'poseidon-calibrate --only herbie-platform --out {{.*}}nocm/cm_gfx942_Fixture.csv'
