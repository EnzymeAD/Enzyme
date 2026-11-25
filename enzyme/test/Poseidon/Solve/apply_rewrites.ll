; DEFINE: %{regime} = env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %S/../Inputs/regime_split/tester.ll \
; DEFINE:   %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; DEFINE: %{pass} = %loadPoseidon -passes="poseidon,poseidon-finalize"
; DEFINE: %{passfn} = %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)"
; DEFINE: %{joint} = -poseidon-profile-use=%S/../Inputs/joint_profiles
; DEFINE: %{pt} = -poseidon-enable-herbie=false -poseidon-enable-pt=true
; DEFINE: %{cm} = -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; RUN: rm -rf %t && split-file %s %t

; -poseidon-apply-rewrites selects candidates by name: an algebraic rewrite by
; its id or its expression, a precision candidate by its description, each at
; a site named by the function the marker optimizes.

; regime.ll: the replayed Herbie answer lists a regime split first and the
; exact rewrite second. The solve drops the split and its report numbers the
; exact rewrite R0_1, its place in Herbie's list. With a table cached, applying
; R0_1 by hand names the same rewrite, although it is at position 0 of the list
; pricing leaves.
; RUN: mkdir %t/cached
; RUN: %{regime} -poseidon-cost-budget=1e9 -poseidon-print -poseidon-cache=%t/cached -S 2>&1 | FileCheck --check-prefix=CACHED-SOLVE %s
; RUN: %{regime} -poseidon-apply-rewrites=R0_1 -poseidon-print -poseidon-cache=%t/cached -S 2>&1 | FileCheck --check-prefix=CACHED-APPLY %s

; CACHED-SOLVE: dropped (unpriced arm)
; CACHED-SOLVE: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; CACHED-APPLY: Selecting R0_1: (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) -> (/.f64 (*.f64 v1 v1)
; CACHED-APPLY: define double @preprocess_tester
; CACHED-APPLY-NOT: select
; CACHED-APPLY: ret double

; A rewrite named by its expression is applied; the regime split is a name of
; no candidate, because pricing dropped it, and that is what the error says.
; RUN: mkdir %t/named
; RUN: %{regime} -poseidon-cache=%t/named/c "-poseidon-apply-rewrites=tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))" -S -poseidon-print 2>&1 | FileCheck --check-prefix=NAMED %s
; RUN: not %{regime} -poseidon-cache=%t/named/c "-poseidon-apply-rewrites=R0=(if.f64 (<=.f64 v1 #s(literal -4999/5000 binary64)) #s(literal 0 binary64) (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)))" -S -o /dev/null 2>&1 | FileCheck --check-prefix=NAMED-DROPPED %s
; RUN: not %{regime} -poseidon-cache=%t/named/c "-poseidon-apply-rewrites=R0=(-.f64 v0 v1)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=NAMED-UNKNOWN %s

; NAMED: [poseidon] Selecting tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)): (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) -> (/.f64 (*.f64 v1 v1)
; NAMED: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; NAMED: [poseidon] pin: tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; NAMED-LABEL: define double @preprocess_tester
; NAMED-NOT: select
; NAMED: ret double
; NAMED-DROPPED: error: {{.*}}Poseidon: -poseidon-apply-rewrites: tester: R0 has no candidate named '(if.f64 (<=.f64 v1 #s(literal -4999/5000 binary64)) {{.*}}': pricing dropped it, because a regime-split arm is reached by fewer than {{[0-9]+}} samples

; NAMED-UNKNOWN: error: {{.*}}Poseidon: -poseidon-apply-rewrites: tester: R0 has no candidate named '(-.f64 v0 v1)'. Its candidates are:
; NAMED-UNKNOWN-NEXT: {{^}}  (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}

; The value a replay from the cached table prints applies what the replay
; applied.
; RUN: %{regime} -poseidon-cache=%t/named/c -poseidon-cost-budget=1e9 -poseidon-print -S -o /dev/null 2>&1 | FileCheck --check-prefix=NAMED-SOLVE %s
; RUN: %{regime} -poseidon-cache=%t/named/c -poseidon-cost-budget=1e9 -poseidon-print -S -o %t/named/replay.ll 2> %t/named/replay.err
; RUN: FileCheck --check-prefix=NAMED-REPLAY %s < %t/named/replay.err
; RUN: %{regime} -poseidon-cache=%t/named/c -poseidon-print "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/named/replay.err | paste -sd, -)" -S -o %t/named/pinned.ll 2> %t/named/pinned.err
; RUN: diff %t/named/replay.ll %t/named/pinned.ll
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: ' %t/named/replay.err > %t/named/replay.picks
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: ' %t/named/pinned.err > %t/named/pinned.picks
; RUN: diff %t/named/replay.picks %t/named/pinned.picks

; NAMED-SOLVE: DP table for preprocess_tester cached to
; NAMED-SOLVE: [poseidon] pin: tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}

; NAMED-REPLAY: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; NAMED-REPLAY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; NAMED-REPLAY: [poseidon] pin: tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}

; sites.ll, two sites with the profiles of Inputs/joint_profiles: an item
; without a site applies at every site.
; RUN: mkdir %t/sites %t/sites/mangled %t/sites/onemangled
; RUN: %opt %t/sites.ll %{pass} %{joint} %{pt} %{cm} -poseidon-cache=%t/sites/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(50%%) + FP32(50%%),tester_b:PT0=All FP64(70%%) + FP32(30%%)" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=SITES %s
; RUN: %opt %t/sites.ll %{pass} %{joint} %{pt} -poseidon-joint-dp %{cm} -poseidon-cache=%t/sites/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(50%%) + FP32(50%%),tester_b:PT0=All FP64(70%%) + FP32(30%%)" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=SITES %s
; RUN: %opt %t/sites.ll %{pass} %{joint} %{pt} %{cm} -poseidon-cache=%t/sites/c "-poseidon-apply-rewrites=PT0=All FP64(0%%) + FP32(100%%)" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-EVERY %s
; RUN: not %opt %t/sites.ll %{pass} %{joint} %{pt} %{cm} -poseidon-cache=%t/sites/c "-poseidon-apply-rewrites=tester_c:PT0=All FP64(0%%) + FP32(100%%)" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NOSITE %s
; RUN: not %opt %t/sites.ll %{pass} %{joint} %{pt} %{cm} -poseidon-cache=%t/sites/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(10%%) + FP32(90%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONAME %s
; RUN: not %opt %t/sites.ll %{pass} %{joint} %{pt} %{cm} -poseidon-cache=%t/sites/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(50%%) + FP32(50%%),PT0_0" -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-TWICE %s

; SITES: [poseidon] Selecting tester_a:PT0=All FP64(50%) + FP32(50%): All FP64(50%) + FP32(50%)
; SITES: Applying solution for CS: All FP64(50%) + FP32(50%) (#1)
; SITES: [poseidon] pin: tester_a:PT0=All FP64(50%) + FP32(50%){{$}}
; SITES: Applying solution for CS: All FP64(70%) + FP32(30%) (#2)
; SITES: [poseidon] pin: tester_b:PT0=All FP64(70%) + FP32(30%){{$}}

; SITES-EVERY: [poseidon] pin: tester_a:PT0=All FP64(0%) + FP32(100%){{$}}
; SITES-EVERY: [poseidon] pin: tester_b:PT0=All FP64(0%) + FP32(100%){{$}}

; SITES-NOSITE: [poseidon] pin: {{$}}
; SITES-NOSITE: [poseidon] pin: {{$}}
; SITES-NOSITE: error: Poseidon: -poseidon-apply-rewrites item 'tester_c:PT0=All FP64(0%) + FP32(100%)' names the site 'tester_c', which is no site of this module. Its sites are:
; SITES-NOSITE-NEXT: {{^}}  tester_a{{$}}
; SITES-NOSITE-NEXT: {{^}}  tester_b{{$}}

; SITES-NONAME: error: {{.*}}Poseidon: -poseidon-apply-rewrites: tester_a: PT0 has no candidate named 'All FP64(10%) + FP32(90%)'. Its candidates are:
; SITES-NONAME-NEXT: {{^}}  All FP64(0%) + FP32(100%){{$}}
; SITES-NONAME-NEXT: {{^}}  All FP64(50%) + FP32(50%){{$}}

; SITES-TWICE: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'PT0_0': CS 0 already has a PT applied; give one item per unit

; Two sites whose symbols demangle to one name, tester(double, double) and
; tester(double, int): the name is ambiguous and each symbol names its own
; site. With only one of them mangled, the name is that site's.
; RUN: sed -e 's/tester_a/_Z6testerdd/g' -e 's/tester_b/_Z6testerdi/g' %t/sites.ll > %t/sites/mangled/m.ll
; RUN: cp %S/../Inputs/joint_profiles/preprocess_tester_a.fpprofile %t/sites/mangled/preprocess__Z6testerdd.fpprofile
; RUN: cp %S/../Inputs/joint_profiles/preprocess_tester_b.fpprofile %t/sites/mangled/preprocess__Z6testerdi.fpprofile
; RUN: not %opt %t/sites/mangled/m.ll %{pass} -poseidon-profile-use=%t/sites/mangled %{pt} %{cm} -poseidon-cache=%t/sites/mangled/c "-poseidon-apply-rewrites=tester:PT0=All FP64(0%%) + FP32(100%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-AMBIGUOUS %s
; RUN: %opt %t/sites/mangled/m.ll %{pass} -poseidon-profile-use=%t/sites/mangled %{pt} %{cm} -poseidon-cache=%t/sites/mangled/c "-poseidon-apply-rewrites=_Z6testerdi:PT0=All FP64(70%%) + FP32(30%%)" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-SYMBOL %s
; RUN: sed -e 's/tester_a/_Z6testerdd/g' %t/sites.ll > %t/sites/onemangled/m.ll
; RUN: cp %S/../Inputs/joint_profiles/preprocess_tester_a.fpprofile %t/sites/onemangled/preprocess__Z6testerdd.fpprofile
; RUN: cp %S/../Inputs/joint_profiles/preprocess_tester_b.fpprofile %t/sites/onemangled/preprocess_tester_b.fpprofile
; RUN: %opt %t/sites/onemangled/m.ll %{pass} -poseidon-profile-use=%t/sites/onemangled %{pt} %{cm} -poseidon-cache=%t/sites/onemangled/c "-poseidon-apply-rewrites=tester:PT0=All FP64(50%%) + FP32(50%%)" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-BASENAME %s

; SITES-AMBIGUOUS: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'tester:PT0=All FP64(0%) + FP32(100%)' names the site 'tester', which more than one site of this module answers to; give one of them by its symbol:
; SITES-AMBIGUOUS-NEXT: {{^}}  _Z6testerdd (tester(double, double)){{$}}
; SITES-AMBIGUOUS-NEXT: {{^}}  _Z6testerdi (tester(double, int)){{$}}

; SITES-SYMBOL: [poseidon] pin: {{$}}
; SITES-SYMBOL: Applying solution for CS: All FP64(70%) + FP32(30%) (#2)
; SITES-SYMBOL: [poseidon] pin: _Z6testerdi:PT0=All FP64(70%) + FP32(30%){{$}}

; SITES-BASENAME: Applying solution for CS: All FP64(50%) + FP32(50%) (#1)
; SITES-BASENAME: [poseidon] pin: _Z6testerdd:PT0=All FP64(50%) + FP32(50%){{$}}
; SITES-BASENAME: [poseidon] pin: {{$}}

; The values a solve prints, joined, apply what it applied: under a budget of
; -0.04 the per-site solves pick a different candidate at each site, and the
; joint solve spends it on tester_b alone.
; RUN: %opt %t/sites.ll %{passfn} %{joint} %{pt} %{cm} -poseidon-cache=%t/sites/site -poseidon-cost-budget=-0.04 -S -poseidon-print -o %t/sites/site.ll 2> %t/sites/site.err
; RUN: FileCheck --check-prefix=SITES-SITE-SOLVE %s < %t/sites/site.err
; RUN: %opt %t/sites.ll %{passfn} %{joint} %{pt} %{cm} -poseidon-cache=%t/sites/site -poseidon-print "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/sites/site.err | paste -sd, -)" -S -o %t/sites/site.pinned.ll 2> %t/sites/site.pinned.err
; RUN: diff %t/sites/site.ll %t/sites/site.pinned.ll
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: ' %t/sites/site.err > %t/sites/site.picks
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: ' %t/sites/site.pinned.err > %t/sites/site.pinned.picks
; RUN: diff %t/sites/site.picks %t/sites/site.pinned.picks
; RUN: %opt %t/sites.ll %{passfn} %{joint} %{pt} -poseidon-joint-dp %{cm} -poseidon-cache=%t/sites/joint -poseidon-cost-budget=-0.04 -S -poseidon-print -o %t/sites/joint.ll 2> %t/sites/joint.err
; RUN: FileCheck --check-prefix=SITES-JOINT-SOLVE %s < %t/sites/joint.err
; RUN: %opt %t/sites.ll %{passfn} %{joint} %{pt} -poseidon-joint-dp %{cm} -poseidon-cache=%t/sites/joint -poseidon-print "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/sites/joint.err | paste -sd, -)" -S -o %t/sites/joint.pinned.ll 2> %t/sites/joint.pinned.err
; RUN: diff %t/sites/joint.ll %t/sites/joint.pinned.ll
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: ' %t/sites/joint.err > %t/sites/joint.picks
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: ' %t/sites/joint.pinned.err > %t/sites/joint.pinned.picks
; RUN: diff %t/sites/joint.picks %t/sites/joint.pinned.picks

; SITES-SITE-SOLVE: [poseidon] pin: tester_a:PT0=All FP64(0%) + FP32(100%){{$}}
; SITES-SITE-SOLVE: [poseidon] pin: tester_b:PT0=All FP64(70%) + FP32(30%){{$}}

; SITES-JOINT-SOLVE: [poseidon] pin: {{$}}
; SITES-JOINT-SOLVE: [poseidon] pin: tester_b:PT0=All FP64(70%) + FP32(30%){{$}}

;--- sites.ll
define double @tester_a(double %x, double %y) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %x
  ret double %mul
}
define double @tester_b(double %x, double %y) #0 {
entry:
  %mul = fmul fast double %x, %y
  %sub = fsub fast double %mul, %y
  %div = fdiv fast double %sub, %x
  ret double %div
}
define double @site_a(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_a, double %x, double %y)
  ret double %0
}
define double @site_b(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_b, double %x, double %y)
  ret double %0
}
declare double @__poseidon_fp_optimize(double (double, double)*, ...)
attributes #0 = { "target-cpu"="x86-64" }
