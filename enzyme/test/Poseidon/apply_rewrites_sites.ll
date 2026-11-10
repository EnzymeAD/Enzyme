; RUN: rm -rf %t && mkdir -p %t/mangled %t/onemangled
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(50%%) + FP32(50%%),tester_b:PT0=All FP64(70%%) + FP32(30%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(50%%) + FP32(50%%),tester_b:PT0=All FP64(70%%) + FP32(30%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=PT0=All FP64(0%%) + FP32(100%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=EVERY %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=tester_c:PT0=All FP64(0%%) + FP32(100%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=NOSITE %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(10%%) + FP32(90%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=NONAME %s
; RUN: not --crash %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c "-poseidon-apply-rewrites=tester_a:PT0=All FP64(50%%) + FP32(50%%),PT0_0" -S -o /dev/null 2>&1 | FileCheck --check-prefix=TWICE %s

; Two sites whose symbols demangle to one name, tester(double, double) and
; tester(double, int): the name is ambiguous and each symbol names its own
; site. With only one of them mangled, the name is that site's.
; RUN: sed -e 's/tester_a/_Z6testerdd/g' -e 's/tester_b/_Z6testerdi/g' %s > %t/mangled/m.ll
; RUN: cp %S/Inputs/joint_profiles/preprocess_tester_a.fpprofile %t/mangled/preprocess__Z6testerdd.fpprofile
; RUN: cp %S/Inputs/joint_profiles/preprocess_tester_b.fpprofile %t/mangled/preprocess__Z6testerdi.fpprofile
; RUN: not %opt %t/mangled/m.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/mangled -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/mangled/c "-poseidon-apply-rewrites=tester:PT0=All FP64(0%%) + FP32(100%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=AMBIGUOUS %s
; RUN: %opt %t/mangled/m.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/mangled -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/mangled/c "-poseidon-apply-rewrites=_Z6testerdi:PT0=All FP64(70%%) + FP32(30%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=SYMBOL %s
; RUN: sed -e 's/tester_a/_Z6testerdd/g' %s > %t/onemangled/m.ll
; RUN: cp %S/Inputs/joint_profiles/preprocess_tester_a.fpprofile %t/onemangled/preprocess__Z6testerdd.fpprofile
; RUN: cp %S/Inputs/joint_profiles/preprocess_tester_b.fpprofile %t/onemangled/preprocess_tester_b.fpprofile
; RUN: %opt %t/onemangled/m.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/onemangled -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/onemangled/c "-poseidon-apply-rewrites=tester:PT0=All FP64(50%%) + FP32(50%%)" -S -o /dev/null 2>&1 | FileCheck --check-prefix=BASENAME %s

; The values a solve prints, joined, apply what it applied: under a budget of
; -64 the per-site solves pick a different candidate at each site, and the
; joint solve spends it on tester_b alone.
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/site -poseidon-cost-budget=-0.04 -S -o %t/site.ll 2> %t/site.err
; RUN: FileCheck --check-prefix=SITE-SOLVE %s < %t/site.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/site "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/site.err | paste -sd, -)" -S -o %t/site.pinned.ll 2> %t/site.pinned.err
; RUN: diff %t/site.ll %t/site.pinned.ll
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: ' %t/site.err > %t/site.picks
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: ' %t/site.pinned.err > %t/site.pinned.picks
; RUN: diff %t/site.picks %t/site.pinned.picks
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/joint -poseidon-cost-budget=-0.04 -S -o %t/joint.ll 2> %t/joint.err
; RUN: FileCheck --check-prefix=JOINT-SOLVE %s < %t/joint.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/joint "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/joint.err | paste -sd, -)" -S -o %t/joint.pinned.ll 2> %t/joint.pinned.err
; RUN: diff %t/joint.ll %t/joint.pinned.ll
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: ' %t/joint.err > %t/joint.picks
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: ' %t/joint.pinned.err > %t/joint.pinned.picks
; RUN: diff %t/joint.picks %t/joint.pinned.picks

; -poseidon-apply-rewrites names precision candidates by their description and
; sites by the function the marker optimizes; an item without a site applies
; at every site. The two sites and profiles are joint_dp_two_sites.ll's.

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

; SITES: [poseidon] Selecting tester_a:PT0=All FP64(50%) + FP32(50%): All FP64(50%) + FP32(50%)
; SITES: Applying solution for CS: All FP64(50%) + FP32(50%) (#1)
; SITES: [poseidon] pin: tester_a:PT0=All FP64(50%) + FP32(50%){{$}}
; SITES: Applying solution for CS: All FP64(70%) + FP32(30%) (#2)
; SITES: [poseidon] pin: tester_b:PT0=All FP64(70%) + FP32(30%){{$}}

; EVERY: [poseidon] pin: tester_a:PT0=All FP64(0%) + FP32(100%){{$}}
; EVERY: [poseidon] pin: tester_b:PT0=All FP64(0%) + FP32(100%){{$}}

; NOSITE: [poseidon] pin: {{$}}
; NOSITE: [poseidon] pin: {{$}}
; NOSITE: LLVM ERROR: [poseidon] -poseidon-apply-rewrites item 'tester_c:PT0=All FP64(0%) + FP32(100%)' names the site 'tester_c', which is no site of this module. Its sites are:
; NOSITE-NEXT: {{^}}  tester_a{{$}}
; NOSITE-NEXT: {{^}}  tester_b{{$}}

; NONAME: LLVM ERROR: [poseidon] tester_a: PT0 has no candidate named 'All FP64(10%) + FP32(90%)'. Its candidates are:
; NONAME-NEXT: {{^}}  All FP64(0%) + FP32(100%){{$}}
; NONAME-NEXT: {{^}}  All FP64(50%) + FP32(50%){{$}}

; TWICE: LLVM ERROR: [poseidon] CS 0 already has a PT applied (duplicate ID 'PT0_0')

; AMBIGUOUS: LLVM ERROR: [poseidon] -poseidon-apply-rewrites item 'tester:PT0=All FP64(0%) + FP32(100%)' names the site 'tester', which more than one site of this module answers to; give one of them by its symbol:
; AMBIGUOUS-NEXT: {{^}}  _Z6testerdd (tester(double, double)){{$}}
; AMBIGUOUS-NEXT: {{^}}  _Z6testerdi (tester(double, int)){{$}}

; SYMBOL: [poseidon] pin: {{$}}
; SYMBOL: Applying solution for CS: All FP64(70%) + FP32(30%) (#2)
; SYMBOL: [poseidon] pin: _Z6testerdi:PT0=All FP64(70%) + FP32(30%){{$}}

; BASENAME: Applying solution for CS: All FP64(50%) + FP32(50%) (#1)
; BASENAME: [poseidon] pin: _Z6testerdd:PT0=All FP64(50%) + FP32(50%){{$}}
; BASENAME: [poseidon] pin: {{$}}

; SITE-SOLVE: [poseidon] pin: tester_a:PT0=All FP64(0%) + FP32(100%){{$}}
; SITE-SOLVE: [poseidon] pin: tester_b:PT0=All FP64(70%) + FP32(30%){{$}}

; JOINT-SOLVE: [poseidon] pin: {{$}}
; JOINT-SOLVE: [poseidon] pin: tester_b:PT0=All FP64(70%) + FP32(30%){{$}}
