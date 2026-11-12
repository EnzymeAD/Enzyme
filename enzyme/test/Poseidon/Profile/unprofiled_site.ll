; RUN: rm -rf %t && mkdir -p %t/empty %t/metric_only %t/mixed
; RUN: echo "qoi 1" > %t/metric_only/metric.txt
; RUN: cp %S/../Inputs/other_sites/preprocess_other_poseidon_body.fpprofile %t/mixed/
; RUN: cp %S/../Inputs/other_sites_format1/preprocess_other_poseidon_body.fpprofile %t/mixed/preprocess_older_poseidon_body.fpprofile
; RUN: sed '/^define double @tester/,/^}/s|%mul = fmul fast double %add, %z|%extra = fmul fast double %add, %add\n  %mul = fmul fast double %extra, %z|' %s > %t/drift.ll
;
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/never_executed -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -pass-remarks-missed=poseidon -S -o %t/never.ll 2> %t/never.err
; RUN: FileCheck --check-prefix=NEVER --implicit-check-not=warning --implicit-check-not=Warning %s < %t/never.err
; RUN: FileCheck --check-prefix=NEVER-REMARK %s < %t/never.err
; RUN: FileCheck --check-prefix=LEFT %s < %t/never.ll
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/never_executed -poseidon-strict-mode -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=NEVER --implicit-check-not=warning --implicit-check-not=Warning %s
; RUN: not %opt < %t/drift.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/never_executed -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=DRIFT %s
;
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/missing -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=NODIR %s
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/empty -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY %s
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/metric_only -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=METRIC %s
;
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -pass-remarks-missed=poseidon -S -o %t/absent.ll 2> %t/absent.err
; RUN: FileCheck --check-prefix=ABSENT %s < %t/absent.err
; RUN: FileCheck --check-prefix=LEFT %s < %t/absent.ll
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-strict-mode -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=STRICT %s
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/mixed -poseidon-strict-mode -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=MIXED %s
;
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/other_sites_format1 -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -pass-remarks-missed=poseidon -S -o %t/format1.ll 2> %t/format1.err
; RUN: FileCheck --check-prefix=FORMAT1 %s < %t/format1.err
; RUN: FileCheck --check-prefix=LEFT %s < %t/format1.ll
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/other_sites_format1 -poseidon-strict-mode -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=FORMAT1-STRICT %s
;
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash -poseidon-output-profiles=%S/../Inputs/never_executed -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o %t/out_never.ll 2>&1 | FileCheck --check-prefix=OUT-NEVER --implicit-check-not=warning --implicit-check-not=Warning %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/out_never.ll
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash -poseidon-output-profiles=%S/../Inputs/canonical_hash,%t/missing -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=OUT-NODIR %s
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash -poseidon-output-profiles=%t/empty -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=OUT-EMPTY %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash -poseidon-output-profiles=%S/../Inputs/other_sites -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o %t/out_absent.ll 2>&1 | FileCheck --check-prefix=OUT-ABSENT %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/out_absent.ll
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash -poseidon-output-profiles=%S/../Inputs/other_sites -poseidon-strict-mode -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=OUT-STRICT %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash -poseidon-output-profiles=%S/../Inputs/other_sites_format1 -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o %t/out_format1.ll 2>&1 | FileCheck --check-prefix=OUT-FORMAT1 %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/out_format1.ll
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/canonical_hash -poseidon-output-profiles=%S/../Inputs/other_sites_format1 -poseidon-strict-mode -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S -o /dev/null 2>&1 | FileCheck --check-prefix=OUT-FORMAT1-STRICT %s

; Why a site has no profile to solve from decides what profile use does. A
; site the profiling run instrumented and never executed has a profile that
; holds its header only, and is left as written with a note. A site the
; profiled program did not contain has no profile in a directory that holds
; others, and is left as written with a warning, or refused under strict mode.
; A directory none of whose profiles names its format (ProfileFormat) was
; written before the runtime left a profile for a site it never executed, so a
; site absent from it may simply not have run: it is left as written with a
; warning that says so, and refused under strict mode, which leaves a site
; unchanged only when its profile says it did not run; one such profile beside
; one that names its format does not make the directory that old. A directory that holds
; no profile at all was not written by a profiling run and is refused. Every
; -poseidon-output-profiles directory is read by the same rules.

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

; NEVER: [poseidon] tester: the profiling run instrumented this site and never executed it ({{.*}}never_executed/preprocess_tester.fpprofile); left unchanged
; NEVER-NOT: [poseidon]
; NEVER-REMARK: remark: {{.*}}Poseidon left the site unchanged: the profiling run instrumented this site and never executed it

; The site calls the body as written.
; LEFT-LABEL: define double @test_opt(
; LEFT-NEXT: entry:
; LEFT-NEXT: call double @tester(double %x, double %y, double %z)

; DRIFT: Poseidon: the profile of preprocess_tester was recorded against a different canonical form (profile CanonicalHash f9f2bd660c90d989, this compile {{[0-9a-f]+}})

; NODIR: error: {{.*}}Poseidon: the profile directory {{.*}}missing (-poseidon-profile-use) does not exist; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; EMPTY: error: {{.*}}Poseidon: the profile directory {{.*}}empty (-poseidon-profile-use) holds no .fpprofile file; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; METRIC: error: {{.*}}Poseidon: the profile directory {{.*}}metric_only (-poseidon-profile-use) holds no .fpprofile file; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote

; ABSENT: warning: {{.*}}Poseidon: no profile at {{.*}}other_sites/preprocess_tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
; ABSENT: [poseidon] tester: the profiled program did not contain this site (no profile at {{.*}}other_sites/preprocess_tester.fpprofile); left unchanged
; ABSENT: remark: {{.*}}Poseidon left the site unchanged: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed)

; STRICT: error: {{.*}}Poseidon: no profile of tester at {{.*}}other_sites/preprocess_tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed), and -poseidon-strict-mode does not leave a site unchanged

; FORMAT1: warning: {{.*}}Poseidon: no profile at {{.*}}other_sites_format1/preprocess_tester.fpprofile: the profile directory was written before the profiling runtime recorded sites that never ran, so this site may simply not have run (or the profiled program did not contain it); leaving the site unchanged
; FORMAT1: [poseidon] tester: the profile directory was written before the profiling runtime recorded sites that never ran, so this site may simply not have run (or the profiled program did not contain it) (no profile at {{.*}}other_sites_format1/preprocess_tester.fpprofile); left unchanged
; FORMAT1: remark: {{.*}}Poseidon left the site unchanged: the profile directory was written before the profiling runtime recorded sites that never ran, so this site may simply not have run (or the profiled program did not contain it)

; FORMAT1-STRICT: error: {{.*}}Poseidon: no profile of tester at {{.*}}other_sites_format1/preprocess_tester.fpprofile: the profile directory was written before the profiling runtime recorded sites that never ran, so it cannot tell whether this site did not run or the profiled program did not contain it, and -poseidon-strict-mode does not leave a site unchanged; generate the profile again with this Poseidon

; MIXED: error: {{.*}}Poseidon: no profile of tester at {{.*}}mixed/preprocess_tester.fpprofile: the profiled program did not contain this site

; OUT-NEVER: [poseidon] tester: the profiling run instrumented this site and never executed it ({{.*}}never_executed/preprocess_tester.fpprofile); left unchanged
; OUT-NODIR: error: {{.*}}Poseidon: the profile directory {{.*}}missing (-poseidon-output-profiles) does not exist; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; OUT-EMPTY: error: {{.*}}Poseidon: the profile directory {{.*}}empty (-poseidon-output-profiles) holds no .fpprofile file; no profiling run wrote into it. Run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; OUT-ABSENT: warning: {{.*}}Poseidon: no profile at {{.*}}other_sites/preprocess_tester.fpprofile: the profiled program did not contain this site
; OUT-ABSENT: [poseidon] tester: the profiled program did not contain this site (no profile at {{.*}}other_sites/preprocess_tester.fpprofile); left unchanged
; OUT-STRICT: error: {{.*}}Poseidon: no profile of tester at {{.*}}other_sites/preprocess_tester.fpprofile: the profiled program did not contain this site
; OUT-FORMAT1: warning: {{.*}}Poseidon: no profile at {{.*}}other_sites_format1/preprocess_tester.fpprofile: the profile directory was written before the profiling runtime recorded sites that never ran
; OUT-FORMAT1: [poseidon] tester: the profile directory was written before the profiling runtime recorded sites that never ran, so this site may simply not have run (or the profiled program did not contain it) (no profile at {{.*}}other_sites_format1/preprocess_tester.fpprofile); left unchanged
; OUT-FORMAT1-STRICT: error: {{.*}}Poseidon: no profile of tester at {{.*}}other_sites_format1/preprocess_tester.fpprofile: the profile directory was written before the profiling runtime recorded sites that never ran
