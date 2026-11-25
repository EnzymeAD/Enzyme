; DEFINE: %{opt} = %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; DEFINE:   -poseidon-tau=1e-12 -poseidon-print -S -o /dev/null
; DEFINE: %{prof} = -poseidon-profile-use=%S/../../Inputs/matmul_site_profile
; RUN: rm -rf %t && mkdir %t

; The accuracy memo of the product of Inputs/stale_cache/matmul_device.ll, one
; entry per priced realization, each stored with its key.

; An entry holds its key's parts. The compile after the first finds all 13
; entries under the same key and computes nothing; a single-threaded compile
; writes the same entries. An entry whose stored key differs in one part is
; recomputed, and the compile names the part: the sampling seed, and the
; compiler, here another build of the pass.
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

; Every entry the first compile writes, planted in an empty cache under the
; same name with the text 0 0 (the numbers of an FP16 product accurate to every
; tolerance), does not parse and is recomputed: the second compile computes the
; same domain errors and pick as the first.
; RUN: mkdir %t/key %t/key/seeded
; RUN: %{opt} %{prof} -poseidon-cache=%t/key/first 2>&1 | FileCheck --check-prefixes=KEY-FIRST,KEY-PRICED %s
; RUN: ls %t/key/first | FileCheck --check-prefix=KEY-MEMO %s
; RUN: for f in %t/key/first/matmul_accuracy_*; do echo '0 0' > %t/key/seeded/$(basename $f); done
; RUN: %{opt} %{prof} -poseidon-cache=%t/key/seeded 2>&1 | FileCheck --check-prefix=KEY-PRICED %s

; KEY-FIRST: [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=3.137966e-04

; KEY-MEMO: matmul_accuracy_

; KEY-PRICED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; KEY-PRICED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; KEY-PRICED: #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=3.137966e-04
; KEY-PRICED: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7){{$}}
