; RUN: rm -rf %t && mkdir -p %t
; RUN: sed 's/@tester_a, double %x, double %y)/@tester_a, double %x, double %y, metadata !"poseidon_tau", double 1.0e-02)/' %S/joint_dp_two_sites.ll > %t/a.ll
; RUN: sed 's/@tester_b, double %x, double %y)/@tester_b, double %x, double %y, metadata !"poseidon_tau", double 1.0e-03)/' %t/a.ll > %t/ab_diff.ll
; RUN: sed 's/@tester_b, double %x, double %y)/@tester_b, double %x, double %y, metadata !"poseidon_tau", double 1.0e-02)/' %t/a.ll > %t/ab_same.ll
; RUN: not %opt %t/a.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c1 -S -o /dev/null 2>&1 | FileCheck --check-prefix=NONE %s
; RUN: not %opt %t/ab_diff.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c2 -S -o /dev/null 2>&1 | FileCheck --check-prefix=DIFF %s
; RUN: %opt %t/a.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c3 -poseidon-tau=1e-2 -S -o %t/flag.out.ll 2> %t/flag.err
; RUN: %opt %t/ab_same.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/c4 -S -o %t/site.out.ll 2> %t/site.err
; RUN: FileCheck --check-prefix=SOLVE %s < %t/flag.err
; RUN: FileCheck --check-prefix=SOLVE %s < %t/site.err
; RUN: sed -e 's/, metadata !"poseidon_tau", double 1.000000e-02//' -e '/^; ModuleID/d' -e '/^source_filename/d' %t/flag.out.ll > %t/flag.cmp.ll
; RUN: sed -e 's/, metadata !"poseidon_tau", double 1.000000e-02//g' -e '/^; ModuleID/d' -e '/^source_filename/d' %t/site.out.ll > %t/site.cmp.ll
; RUN: diff %t/flag.cmp.ll %t/site.cmp.ll
; REQUIRES: poseidon

; One shared budget is spent against one accuracy target, so the sites of a
; -poseidon-joint-dp solve must mean the same one: a site's own target, else
; -poseidon-tau, else none. Different targets are a compile error naming each
; site; equal ones solve as one site would, wherever they are written.

; NONE: error: {{.*}}Poseidon: the sites of one -poseidon-joint-dp solve have different accuracy targets: tester_a (1.000000e-02) tester_b (none); give them one target
; DIFF: error: {{.*}}Poseidon: the sites of one -poseidon-joint-dp solve have different accuracy targets: tester_a (1.000000e-02) tester_b (1.000000e-03); give them one target
; SOLVE: [poseidon] preprocess_tester_a,preprocess_tester_b: tau=1.000000e-02
