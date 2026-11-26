; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache= -S
; DEFINE: %{fma} = %S/../Inputs/fma_site_profile/tester.fpprofile
; RUN: rm -rf %t && split-file %s %t && mkdir -p %t/empty %t/metric_only %t/never_executed %t/site_id %t/truncated %t/field %t/dropped %t/squares %t/mixed %t/traffic %t/traffic_width %t/traffic_array %t/traffic_twice
; RUN: echo "qoi 1" > %t/metric_only/metric.txt
; RUN: sed 's/^  %mul = fmul fast double %add, %z$/  %extra = fmul fast double %add, %add\n  %mul = fmul fast double %extra, %z/' %t/tester.ll > %t/drift.ll
; RUN: printf 'SiteId = 0\nCanonicalHash = f9f2bd660c90d989\nExecuted = 0\n\n' > %t/never_executed/tester.fpprofile
; RUN: %if enzyme %{ %opt < %t/tester.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize" -enzyme-preopt=false -poseidon-profile-generate -S | FileCheck %s --check-prefix=GEN %}
; RUN: %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-early-prune -poseidon-print 2>&1 | FileCheck %s --check-prefix=MATCH --implicit-check-not=error: --implicit-check-not=CanonicalHash
; RUN: not %{opt} < %t/drift.ll -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-early-prune 2>&1 | FileCheck %s --check-prefix=DRIFT
; RUN: sed '1i SiteId = 2147483648' %{fma} > %t/site_id/tester.fpprofile
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/site_id 2>&1 | FileCheck --check-prefix=SITE-ID %s
; RUN: sed '$d' %{fma} > %t/truncated/tester.fpprofile
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/truncated 2>&1 | FileCheck --check-prefix=TRUNCATED %s
; RUN: sed '0,/Exec = 100/s//Exec = many/' %{fma} > %t/field/tester.fpprofile
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/field 2>&1 | FileCheck --check-prefix=FIELD %s
; RUN: sed '0,/MaxRes = .*/s//&\n\tSumValue = 1.0/' %{fma} > %t/dropped/tester.fpprofile
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/dropped 2>&1 | FileCheck --check-prefix=DROPPED %s
; RUN: sed 's/^\(.SumAbsGrad = .*\)$/\1\n\tSumSqGrad = 6.25\n\tSumSqSens = 25.0/' %{fma} > %t/squares/tester.fpprofile
; RUN: sed '0,/SumSqSens/{/SumSq/d}' %t/squares/tester.fpprofile > %t/mixed/tester.fpprofile
; RUN: %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/squares -poseidon-print 2>&1 | FileCheck --check-prefix=SQUARES %s
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/mixed 2>&1 | FileCheck --check-prefix=MIXED %s
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/squares -poseidon-output-profiles=%S/../Inputs/fma_site_profile 2>&1 | FileCheck --check-prefix=SQUARES-OUT %s
; RUN: sed '1i TrafficArray = 0 0 -1\nTraffic = 0 8 4 2 32 16\nTraffic = 1 0 3 0 12 0' %{fma} > %t/traffic/tester.fpprofile
; RUN: %{opt} < %t/tester.ll -poseidon-profile-use=%t/traffic -poseidon-early-prune -poseidon-print 2>&1 | FileCheck %s --check-prefix=MATCH --implicit-check-not=error: --implicit-check-not=CanonicalHash
; RUN: sed '1i Traffic = 1 8 0 4096 0' %{fma} > %t/traffic_width/tester.fpprofile
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/traffic_width 2>&1 | FileCheck --check-prefix=TRAFFIC-FIELDS %s
; RUN: sed '1i TrafficArray = 0 -2 0' %{fma} > %t/traffic_array/tester.fpprofile
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/traffic_array 2>&1 | FileCheck --check-prefix=TRAFFIC-ARRAY %s
; RUN: sed '1i Traffic = 0 8 1 0 8 0\nTraffic = 0 8 1 0 8 0' %{fma} > %t/traffic_twice/tester.fpprofile
; RUN: not %{opt} < %t/tester.ll -o /dev/null -poseidon-profile-use=%t/traffic_twice 2>&1 | FileCheck --check-prefix=TRAFFIC-TWICE %s
; RUN: %{opt} < %t/tester.ll -poseidon-profile-use=%t/never_executed -pass-remarks-missed=poseidon -poseidon-explain=%t/never -poseidon-print -o %t/never.ll 2> %t/never.err
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/never.err %t/never | FileCheck --check-prefix=NEVER-JSON %s
; RUN: FileCheck --check-prefix=NEVER --implicit-check-not=warning --implicit-check-not=Warning %s < %t/never.err
; RUN: FileCheck --check-prefix=NEVER-REMARK %s < %t/never.err
; RUN: FileCheck --check-prefix=LEFT %s < %t/never.ll
; RUN: %{opt} < %t/tester.ll -poseidon-profile-use=%t/never_executed -poseidon-strict-mode -o /dev/null 2>&1 | count 0
; RUN: not %{opt} < %t/drift.ll -poseidon-profile-use=%t/never_executed -o /dev/null 2>&1 | FileCheck --check-prefix=DRIFT %s
; RUN: not %{opt} < %t/tester.ll -poseidon-profile-use=%t/empty -o /dev/null 2>&1 | FileCheck --check-prefix=EMPTY %s
; RUN: not %{opt} < %t/tester.ll -poseidon-profile-use=%t/metric_only -o /dev/null 2>&1 | FileCheck --check-prefix=METRIC %s
; RUN: %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -pass-remarks-missed=poseidon -poseidon-explain=%t/absent -o %t/absent.ll 2>&1 | tee %t/absent.err | FileCheck --check-prefix=ABSENT %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/absent.err %t/absent | FileCheck --check-prefix=ABSENT-JSON %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/absent.ll
; RUN: not %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -poseidon-strict-mode -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=STRICT %s
; RUN: %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-output-profiles=%t/never_executed -poseidon-print -o %t/out_never.ll 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-NEVER --implicit-check-not=warning --implicit-check-not=Warning %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/out_never.ll
; RUN: not %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-output-profiles=%S/../Inputs/fma_site_profile,%t/missing -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-NODIR %s
; RUN: not %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-output-profiles=%t/empty -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-EMPTY %s
; RUN: %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-output-profiles=%S/../Inputs/unrelated_site_profile -o %t/out_absent.ll 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-ABSENT %s
; RUN: FileCheck --check-prefix=LEFT %s < %t/out_absent.ll
; RUN: not %{opt} < %t/tester.ll -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-output-profiles=%S/../Inputs/unrelated_site_profile -poseidon-strict-mode -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=OUT-STRICT %s

; Profile use of one site. The profile carries a digest of the canonical body
; (its slot indices are positions in it): the generating compile registers it, a
; body that produces it is solved, one that does not (drift.ll) is refused; the
; memory traffic lines are read with the header. A profile that does not parse,
; or has a line the format lacks, stops the compile naming the file and line.
; The squared sums SumSqGrad and SumSqSens are read in every record or in none
; (squares, mixed), and an output profile has them when the site's profile does;
; S is then their sum. A site without a profile to solve from: never executed
; (header only) is left as written with a note; absent from a directory of
; others, left with a warning or refused under strict mode; a directory with no
; profile is refused. -poseidon-output-profiles directories follow the same
; rules.

; GEN: @poseidon.site.preprocess_tester = private unnamed_addr constant [7 x i8] c"tester\00"
; GEN: @poseidon.static.preprocess_tester = private unnamed_addr constant [45 x i8] c"SiteId = 0\0ACanonicalHash = f9f2bd660c90d989\0A\00"
; GEN: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @poseidon.register_static.preprocess_tester, ptr null }]
; GEN-LABEL: define internal void @poseidon.register_static.preprocess_tester()
; GEN-NEXT: entry:
; GEN-NEXT: call void @__poseidon_register_static(ptr @poseidon.site.preprocess_tester, ptr @poseidon.static.preprocess_tester)
; GEN-NEXT: ret void
; S is the largest result the fixture records; the site calls the body it kept.
; MATCH-NOT: recorded against a different canonical form
; MATCH: [poseidon] preprocess_tester: tau=5.000000e-01 S=1.500000e+02 A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00
; MATCH-LABEL: define double @test_opt(
; MATCH-NEXT: entry:
; MATCH-NEXT: [[R:%.*]] = call double @tester(double %x, double %y, double %z)
; MATCH-NEXT: ret double [[R]]
; SITE-ID: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}site_id/tester.fpprofile:1: SiteId 2147483648 is out of range; re-run profile generation
; TRUNCATED: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}truncated/tester.fpprofile:20: the record of slot 1 ends before its Operand[1] line; re-run profile generation
; FIELD: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}field/tester.fpprofile:8: Exec many is not a number; re-run profile generation
; DROPPED: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}dropped/tester.fpprofile:5: expected the SumSens line of slot 0; re-run profile generation
; SQUARES: [poseidon] preprocess_tester: tau=5.000000e-01 S=5.000000e+01 A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00
; MIXED: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}mixed/tester.fpprofile:17: the record of slot 1 records the squared sums SumSqGrad and SumSqSens, and the records before it do not; re-run profile generation
; SQUARES-OUT: error: {{.*}}preprocess_tester{{.*}}: Poseidon: output profile {{.*}}fma_site_profile/tester.fpprofile does not record the squared sums SumSqGrad and SumSqSens, and the site's profile does; both must come from the same profiling binary{{$}}
; TRAFFIC-FIELDS: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}traffic_width/tester.fpprofile:1: Traffic 1 8 0 4096 0 is not 6 numbers; re-run profile generation{{$}}
; TRAFFIC-ARRAY: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}traffic_array/tester.fpprofile:1: TrafficArray 0 -2 0 is not <record> <parameter> <offset>; re-run profile generation{{$}}
; TRAFFIC-TWICE: error: {{.*}}preprocess_tester{{.*}}: Poseidon: {{.*}}traffic_twice/tester.fpprofile:2: traffic record 0 is recorded twice; re-run profile generation{{$}}
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
; DRIFT: error: <unknown>:0:0: in function preprocess_tester {{.*}}: Poseidon: the profile of preprocess_tester was recorded against a different canonical form (profile CanonicalHash f9f2bd660c90d989, this compile {{[0-9a-f]+}}); re-run profile generation
; EMPTY: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/empty (-poseidon-profile-use) holds no .fpprofile file; run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; METRIC: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/metric_only (-poseidon-profile-use) holds no .fpprofile file; run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; ABSENT: warning: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile at {{.*}}unrelated_site_profile/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
; ABSENT: remark: <unknown>:0:0: Poseidon left the site unchanged: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed)
; ABSENT-JSON: preprocess_tester: unprofiled; the profiled program did not contain this site
; STRICT: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile of tester at {{.*}}unrelated_site_profile/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed), and -poseidon-strict-mode does not leave a site unchanged
; OUT-NEVER: [poseidon] tester: the profiling run instrumented this site and never executed it ({{.*}}never_executed/tester.fpprofile); left unchanged
; OUT-NODIR: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/missing (-poseidon-output-profiles) does not exist; run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; OUT-EMPTY: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: the profile directory {{.*}}/empty (-poseidon-output-profiles) holds no .fpprofile file; run the program built with -poseidon-profile-generate first, or pass the directory it wrote
; OUT-ABSENT: warning: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile at {{.*}}unrelated_site_profile/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
; OUT-STRICT: error: <unknown>:0:0: in function test_opt {{.*}}: Poseidon: no profile of tester at {{.*}}unrelated_site_profile/tester.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed), and -poseidon-strict-mode does not leave a site unchanged

;--- tester.ll
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
