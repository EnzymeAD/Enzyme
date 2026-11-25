; DEFINE: %{opt} = %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= \
; DEFINE:   -S -o /dev/null
; RUN: rm -rf %t && mkdir -p %t/site_id %t/truncated %t/field %t/dropped
; RUN: sed '1i SiteId = 2147483648' %S/../Inputs/fma_opt_profiles/tester.fpprofile > %t/site_id/tester.fpprofile
; RUN: not %{opt} -poseidon-profile-use=%t/site_id 2>&1 | FileCheck --check-prefix=SITE-ID %s
; RUN: sed '$d' %S/../Inputs/fma_opt_profiles/tester.fpprofile > %t/truncated/tester.fpprofile
; RUN: not %{opt} -poseidon-profile-use=%t/truncated 2>&1 | FileCheck --check-prefix=TRUNCATED %s
; RUN: sed '0,/Exec = 100/s//Exec = many/' %S/../Inputs/fma_opt_profiles/tester.fpprofile > %t/field/tester.fpprofile
; RUN: not %{opt} -poseidon-profile-use=%t/field 2>&1 | FileCheck --check-prefix=FIELD %s
; RUN: sed '0,/MaxRes = .*/s//&\n\tSumValue = 1.0/' %S/../Inputs/fma_opt_profiles/tester.fpprofile > %t/dropped/tester.fpprofile
; RUN: not %{opt} -poseidon-profile-use=%t/dropped 2>&1 | FileCheck --check-prefix=DROPPED %s

; A profile that does not parse stops the compile at the site, naming the file
; and the line; so does a line the format does not have.

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

attributes #0 = { "target-cpu"="x86-64" }

; SITE-ID: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}site_id/tester.fpprofile:1: SiteId 2147483648 is out of range; re-run profile generation.
; TRUNCATED: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}truncated/tester.fpprofile:20: the record of slot 1 ends before its Operand[1] line; re-run profile generation.
; FIELD: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}field/tester.fpprofile:8: Exec many is not a number; re-run profile generation.
; DROPPED: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}dropped/tester.fpprofile:5: expected the SumSens line of slot 0; re-run profile generation.
