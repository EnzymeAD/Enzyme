; RUN: rm -rf %t && mkdir -p %t
; RUN: sed 's/double 1.0e-2)/double 1.0e-2, metadata !"poseidon_confidence", double 5.0e-1)/' %s > %t/site.ll
; RUN: sed 's/metadata !"poseidon_tau", double 1.0e-2/metadata !"poseidon_confidence", double 9.0e-1/' %s > %t/notol.ll
; RUN: sed 's/double 1.0e-2)/double 1.0e-2, metadata !"poseidon_confidence", double 1.5e+0)/' %s > %t/badsite.ll

; The flag alone: the default level, named as coming from the flag.
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c1 -poseidon-print -S 2>&1 | FileCheck --check-prefix=DEFAULT %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c2 -poseidon-print -poseidon-confidence=0.5 -S 2>&1 | FileCheck --check-prefix=FLAG %s

; The site key wins over the flag, and says so.
; RUN: %opt %t/site.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c3 -poseidon-print -poseidon-confidence=0.8 -S 2>&1 | FileCheck --check-prefix=SITE %s

; A level outside (0, 1] is refused, by the flag and at the site, each naming
; what carried it.
; RUN: not --crash %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c4 -poseidon-confidence=1.5 -S 2>&1 | FileCheck --check-prefix=BADFLAG %s
; RUN: not %opt %t/badsite.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c5 -S 2>&1 | FileCheck --check-prefix=BADSITE %s

; A confidence level with no tolerance at the same call is refused, naming both
; keys.
; RUN: not %opt %t/notol.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c6 -S 2>&1 | FileCheck --check-prefix=NOTOL %s
; REQUIRES: poseidon

; -poseidon-confidence and the marker key poseidon_confidence are the two ways
; the confidence level of a site's accuracy target is written: the fraction of
; the sampled inputs a matrix product's modelled error must clear the target
; on. This site holds no matrix product, so the level changes nothing it emits;
; what is checked here is that both spellings are accepted, that the site value
; wins over the flag, and that the three refusals name what carried the value.

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @site(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", double %y, double %ys, double %z, metadata !"poseidon_tau", double 1.0e-2)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; DEFAULT: [poseidon] preprocess_tester{{[^ ]*}}: accuracy target confidence 9.500000e-01 (-poseidon-confidence)
; DEFAULT-NOT: overrides -poseidon-confidence

; FLAG: [poseidon] preprocess_tester{{[^ ]*}}: accuracy target confidence 5.000000e-01 (-poseidon-confidence)

; SITE: [poseidon] preprocess_tester{{[^ ]*}}: the site's own confidence 5.000000e-01 overrides -poseidon-confidence=8.000000e-01
; SITE: [poseidon] preprocess_tester{{[^ ]*}}: accuracy target confidence 5.000000e-01 (site)

; BADFLAG: Poseidon: -poseidon-confidence=1.500000 is outside (0, 1]

; BADSITE: error: {{.*}}Poseidon: poseidon_confidence 1.500000 is outside (0, 1]

; NOTOL: error: {{.*}}Poseidon: poseidon_confidence needs a poseidon_tau at the same call; a confidence level is the level this site's own accuracy target is read at
