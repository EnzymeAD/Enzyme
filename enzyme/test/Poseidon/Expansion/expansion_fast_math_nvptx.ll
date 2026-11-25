; DEFINE: %{opt} = %opt %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-cost-budget=-1e-9 -S
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/tail call double @llvm.fmuladd/tail call contract double @llvm.fmuladd/' %S/../Inputs/ds_expansion/ds_fms.ll > %t/contract.ll
; RUN: sed 's/tail call double @llvm.fmuladd/tail call fast double @llvm.fmuladd/; s/= fneg double/= fneg fast double/' %S/../Inputs/ds_expansion/ds_fms.ll > %t/fast.ll
; RUN: sed 's/^attributes #0 = { /attributes #0 = { "unsafe-fp-math"="true" "no-nans-fp-math"="true" "no-infs-fp-math"="true" "approx-func-fp-math"="true" "no-signed-zeros-fp-math"="true" /' %S/../Inputs/ds_expansion/ds_fms.ll > %t/attr.ll
; RUN: %{opt} %S/../Inputs/ds_expansion/ds_fms.ll -poseidon-cache=%t/plain -poseidon-print -o %t/plain.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s
; RUN: %{opt} %t/contract.ll -poseidon-cache=%t/contract -poseidon-print -o %t/contract.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s
; RUN: %{opt} %t/fast.ll -poseidon-cache=%t/fast -poseidon-print -o %t/fast.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s
; RUN: %{opt} %t/attr.ll -poseidon-cache=%t/attr -poseidon-print -o %t/attr.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s
; RUN: %{opt} %t/contract.ll -poseidon-cache=%t/strict -poseidon-strict-mode -poseidon-print -o %t/strict.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s
; RUN: FileCheck --check-prefix=BODY %s < %t/fast.out.ll
; RUN: sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/plain.out.ll > %t/plain.body.ll
; RUN: sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/contract.out.ll | diff %t/plain.body.ll -
; RUN: sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/fast.out.ll | diff %t/plain.body.ll -
; RUN: sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/attr.out.ll | diff %t/plain.body.ll -
; RUN: sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/strict.out.ll | diff %t/plain.body.ll -
; UNSUPPORTED: global-fp-contract

; Fast-math flags on the fmuladd chain of Inputs/ds_expansion (the canonical
; form that names the site's profile ignores them). The materializer emits the
; expansion's operations without fast-math flags. From LLVM 24 neither
; 'contract' nor a fast-math function attribute reaches an operation without
; flags, so a site with 'contract', with 'fast', or with every fast-math
; attribute (strict mode included) gets its expansion, and the emitted function
; is the one emitted for the site without them.

; APPLIED-NOT: NOT PROPOSED
; APPLIED: Applying solution for CS: All FP64(0%) + Expansion2(100%)

; BODY-LABEL: define {{.*}} @preprocess_fms_body(
; BODY-NOT: {{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}
; BODY: call float @llvm.fma.f32(
; BODY-NOT: {{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}
; BODY: fadd double %{{.+}}, %{{.+}}, !poseidon.ds.join
; BODY-NOT: {{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}
; BODY: ret void
