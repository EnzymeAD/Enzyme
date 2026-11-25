; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S
; RUN: rm -rf %t && mkdir -p %t/empty %t/metric_only
; RUN: echo "qoi 1" > %t/metric_only/metric.txt
; RUN: sed '/^define double @tester/,/^}/s|%mul = fmul fast double %add, %z|%extra = fmul fast double %add, %add\n  %mul = fmul fast double %extra, %z|' %s > %t/drift.ll
;
; RUN: %{opt} < %s -poseidon-profile-use=%S/../Inputs/never_executed -pass-remarks-missed=poseidon -poseidon-explain=%t/never -poseidon-print -o %t/never.ll 2> %t/never.err
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/never.err %t/never | FileCheck --check-prefix=NEVER-JSON %s
; RUN: FileCheck --check-prefix=NEVER --implicit-check-not=warning --implicit-check-not=Warning %s < %t/never.err
; RUN: FileCheck --check-prefix=NEVER-REMARK %s < %t/never.err
; RUN: FileCheck --check-prefix=LEFT %s < %t/never.ll
; RUN: %{opt} < %s -poseidon-profile-use=%S/../Inputs/never_executed -poseidon-strict-mode -o /dev/null 2>&1 | count 0
; RUN: not %{opt} < %t/drift.ll -poseidon-profile-use=%S/../Inputs/never_executed -o /dev/null 2>&1 | FileCheck --check-prefix=DRIFT %s
;
; RUN: not %{opt} < %s -poseidon-profile-use=%t/empty -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY %s
; RUN: not %{opt} < %s -poseidon-profile-use=%t/metric_only -o /dev/null 2>&1 | FileCheck --check-prefix=METRIC %s
;
; RUN: %{opt} < %s -poseidon-profile-use=%S/../Inputs/other_sites -pass-remarks-missed=poseidon -poseidon-explain=%t/absent -o %t/absent.ll 2>&1 | tee %t/absent.err | FileCheck --check-prefix=ABSENT %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/absent.err %t/absent | FileCheck --check-prefix=ABSENT-JSON %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/absent.ll
; RUN: not %{opt} < %s -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-strict-mode -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=STRICT %s
;
; RUN: %{opt} < %s -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-output-profiles=%S/../Inputs/never_executed -poseidon-print -o %t/out_never.ll 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-NEVER --implicit-check-not=warning --implicit-check-not=Warning %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/out_never.ll
; RUN: not %{opt} < %s -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-output-profiles=%S/../Inputs/fma_opt_profiles,%t/missing -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-NODIR %s
; RUN: not %{opt} < %s -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-output-profiles=%t/empty -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-EMPTY %s
; RUN: %{opt} < %s -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-output-profiles=%S/../Inputs/other_sites -o %t/out_absent.ll 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-ABSENT %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/out_absent.ll
; RUN: not %{opt} < %s -poseidon-profile-use=%S/../Inputs/fma_opt_profiles -poseidon-output-profiles=%S/../Inputs/other_sites -poseidon-strict-mode -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-STRICT %s

; Why a site has no profile to solve from decides what profile use does. A
; site the profiling run instrumented and never executed has a profile that
; holds its header only, and is left as written with a note. A site the
; profiled program did not contain has no profile in a directory that holds
; others, and is left as written with a warning, or refused under strict mode.
; A directory that holds no profile is refused. Every -poseidon-output-profiles
; directory is read by the same rules.

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

; NEVER: [poseidon] tester: the profiling run instrumented this site and never executed it ({{.*}}never_executed/tester.fpprofile); left unchanged
; NEVER-REMARK: remark: <unknown>:0:0: Poseidon left the site unchanged: the profiling run instrumented this site and never executed it
; NEVER-JSON: preprocess_tester: never_executed; the profiling run instrumented this site and never executed it{{$}}

; The site calls the body, which stays as written.
; LEFT-LABEL: define double @tester(
; LEFT-NEXT: entry:
; LEFT-NEXT: %add = fadd fast double %x, %y
; LEFT-NEXT: %mul = fmul fast double %add, %z
; LEFT-NEXT: ret double %mul
; LEFT-LABEL: define double @test_opt(
; LEFT-NEXT: entry:
; LEFT-NEXT: [[R:%.*]] = call double @tester(double %x, double %y, double %z)
; LEFT-NEXT: ret double [[R]]

; DRIFT: error: <unknown>:0:0: in function preprocess_tester {{.*}}: Poseidon: the profile of preprocess_tester was recorded against a different canonical form (profile CanonicalHash f9f2bd660c90d989, this compile {{[0-9a-f]+}}). Its slot indices do not name the same instructions; re-run profile generation.

; EMPTY: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/empty (-poseidon-profile-use) holds no .fpprofile file; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; METRIC: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/metric_only (-poseidon-profile-use) holds no .fpprofile file; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote

; ABSENT: warning: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile at {{.*}}other_sites/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
; ABSENT: remark: <unknown>:0:0: Poseidon left the site unchanged: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed)
; ABSENT-JSON: preprocess_tester: unprofiled; the profiled program did not contain this site

; STRICT: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile of tester at {{.*}}other_sites/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed), and -poseidon-strict-mode does not leave a site unchanged

; OUT-NEVER: [poseidon] tester: the profiling run instrumented this site and never executed it ({{.*}}never_executed/tester.fpprofile); left unchanged
; OUT-NODIR: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/missing (-poseidon-output-profiles) does not exist; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; OUT-EMPTY: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/empty (-poseidon-output-profiles) holds no .fpprofile file; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; OUT-ABSENT: warning: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile at {{.*}}other_sites/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
; OUT-STRICT: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile of tester at {{.*}}other_sites/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed), and -poseidon-strict-mode does not leave a site unchanged
