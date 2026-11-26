; DEFINE: %{opt} = %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-algebraic=false \
; DEFINE:   -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-tau=1e-12 -poseidon-print -S -o /dev/null
; DEFINE: %{prof} = -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune
; RUN: rm -rf %t && mkdir %t

; The accuracy memo keeps one entry per priced realization with its key; an entry
; of another seed or compiler is recomputed, naming the part.
; RUN: mkdir %t/entry
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/c 2>&1 | FileCheck --check-prefixes=ENTRY-COMPUTE,ENTRY-PRICED --implicit-check-not=recomputing %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/c 2>&1 | FileCheck --check-prefix=ENTRY-PRICED --implicit-check-not='[poseidon] domain error' --implicit-check-not=recomputing %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/one -poseidon-threads=1 2>&1 | FileCheck --check-prefixes=ENTRY-COMPUTE,ENTRY-PRICED --implicit-check-not=recomputing %s
; RUN: diff -r -x 'launches_*' %t/entry/c %t/entry/one
; RUN: cp -r %t/entry/c %t/entry/seed && sed -i 's/^\( *"seed": "\)[0-9]*"/\11"/' %t/entry/seed/matmul_accuracy_*.json
; RUN: cat %t/entry/seed/matmul_accuracy_*.json | FileCheck --check-prefix=ENTRY-EDITED-SEED --implicit-check-not='"seed"' %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/seed -poseidon-print 2>&1 | FileCheck --check-prefixes=ENTRY-SEED,ENTRY-COMPUTE,ENTRY-PRICED --implicit-check-not="different seed" %s
; RUN: cp -r %t/entry/c %t/entry/compiler && sed -i 's/^\( *"compiler": "\)[0-9A-F]* /\10123456789ABCDEF /' %t/entry/compiler/matmul_accuracy_*.json
; RUN: cat %t/entry/compiler/matmul_accuracy_*.json | FileCheck --check-prefix=ENTRY-EDITED-COMPILER --implicit-check-not='"compiler"' %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/compiler -poseidon-print 2>&1 | FileCheck --check-prefixes=ENTRY-COMPILER,ENTRY-COMPUTE,ENTRY-PRICED --implicit-check-not="different compiler" %s

; ENTRY-EDITED-SEED-COUNT-13: "seed": "1"
; ENTRY-EDITED-COMPILER-COUNT-13: "compiler": "0123456789ABCDEF LLVM {{[^"]+}}",

; ENTRY-SEED-COUNT-13: [poseidon] accuracy memo {{.*}}matmul_accuracy_{{[0-9A-F]+}}.json was computed for a different seed; recomputing it
; ENTRY-COMPILER-COUNT-13: [poseidon] accuracy memo {{.*}}matmul_accuracy_{{[0-9A-F]+}}.json was computed for a different compiler; recomputing it

; ENTRY-COMPUTE: [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=3.137966e-04

; ENTRY-PRICED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; ENTRY-PRICED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; ENTRY-PRICED: #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=3.137966e-04
; ENTRY-PRICED: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7){{$}}

; Entries that do not parse are recomputed to the same domain errors and pick.
; RUN: mkdir -p %t/key/seeded
; RUN: ls %t/entry/c | FileCheck --check-prefix=KEY-MEMO %s
; RUN: for f in %t/entry/c/matmul_accuracy_*; do echo '0 0' > %t/key/seeded/$(basename $f); done
; RUN: %{opt} %{prof} -poseidon-cache=%t/key/seeded 2>&1 | FileCheck --check-prefix=ENTRY-PRICED %s

; KEY-MEMO: matmul_accuracy_
