; REQUIRES: scev-trunc-iv-nowrap
; DEFINE: %{opt} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; DEFINE:   -poseidon-tau=1e-12 -poseidon-print -S -o /dev/null
; DEFINE: %{prof} = -poseidon-profile-use=%S/../Inputs/raise_tcec
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
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/c 2>&1 | FileCheck --check-prefix=ENTRY-PRICED --implicit-check-not=DOMERR --implicit-check-not=recomputing %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/one -poseidon-threads=1 2>&1 | FileCheck --check-prefixes=ENTRY-COMPUTE,ENTRY-PRICED --implicit-check-not=recomputing %s
; RUN: diff -r -x 'launches_*' %t/entry/c %t/entry/one
; RUN: cp -r %t/entry/c %t/entry/seed && sed -i 's/^\( *"seed": "\)[0-9]*"/\11"/' %t/entry/seed/matmulacc_*.json
; RUN: cat %t/entry/seed/matmulacc_*.json | FileCheck --check-prefix=ENTRY-EDITED-SEED --implicit-check-not='"seed"' %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/seed 2>&1 | FileCheck --check-prefixes=ENTRY-SEED,ENTRY-COMPUTE,ENTRY-PRICED --implicit-check-not="different seed" %s
; RUN: cp -r %t/entry/c %t/entry/compiler && sed -i 's/^\( *"compiler": "\)[0-9A-F]* /\10123456789ABCDEF /' %t/entry/compiler/matmulacc_*.json
; RUN: cat %t/entry/compiler/matmulacc_*.json | FileCheck --check-prefix=ENTRY-EDITED-COMPILER --implicit-check-not='"compiler"' %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/entry/compiler 2>&1 | FileCheck --check-prefixes=ENTRY-COMPILER,ENTRY-COMPUTE,ENTRY-PRICED --implicit-check-not="different compiler" %s

; ENTRY-EDITED-SEED-COUNT-13: "seed": "1"
; ENTRY-EDITED-COMPILER-COUNT-13: "compiler": "0123456789ABCDEF LLVM {{[^"]+}}",

; ENTRY-SEED-COUNT-13: [poseidon] accuracy memo {{.*}}matmulacc_{{[0-9A-F]+}}.json was computed for a different seed; recomputing it.
; ENTRY-COMPILER-COUNT-13: [poseidon] accuracy memo {{.*}}matmulacc_{{[0-9A-F]+}}.json was computed for a different compiler; recomputing it.

; ENTRY-COMPUTE: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04

; ENTRY-PRICED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; ENTRY-PRICED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; ENTRY-PRICED: #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=3.137966e-04
; ENTRY-PRICED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err 1.840864e-13 <= 1.000000e-12 at confidence 9.500000e-01)

; Every entry the first compile writes, planted in an empty cache under the
; same name with the numbers 0 0 (an FP16 product accurate to every tolerance),
; is a miss because it does not carry its key: the second compile computes the
; same domain errors and pick as the first.
; RUN: mkdir %t/key %t/key/seeded
; RUN: %{opt} %{prof} -poseidon-cache=%t/key/first 2>&1 | FileCheck --check-prefixes=KEY-FIRST,KEY-PRICED %s
; RUN: ls %t/key/first | FileCheck --check-prefix=KEY-MEMO %s
; RUN: for f in %t/key/first/matmulacc_*; do echo '0 0' > %t/key/seeded/$(basename $f); done
; RUN: %{opt} %{prof} -poseidon-cache=%t/key/seeded 2>&1 | FileCheck --check-prefix=KEY-PRICED %s

; KEY-FIRST: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04

; KEY-MEMO: matmulacc_

; KEY-PRICED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; KEY-PRICED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; KEY-PRICED: #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=3.137966e-04
; KEY-PRICED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err 1.840864e-13 <= 1.000000e-12 at confidence 9.500000e-01)

; The 13 results of this compile in the format without keys
; (matmulacc_<hash of the inputs>.txt), each planted with the numbers 0 0, are
; never read from the cache directory: the first compile computes every
; result. Named by -poseidon-import-accuracy-memo, the six of models that are
; unchanged are taken over, so the FP16 dispatch is picked, and stored under
; their keys: the compile after it, without the flag, finds all 13 in the cache
; and computes none. The import compile computes the seven of the Ozaki-II
; model, which no planted entry was computed by.
; RUN: mkdir %t/import %t/import/old
; RUN: for h in 3A0C4F59C6907801 3A0C4F59CB4F208F 3A0C4F59CB4F20B1 83F790B2E595C48 83F790B2E595C4A 83F790B2E595C4C 83F790B2E595C50 83F790B2E595C54 83F790B2E595C58 83F790B2E595C5C D060729D8D884DBF D060729D8D8C25C2 D060729D8D92C0C4; do echo '0 0' > %t/import/old/matmulacc_$h.txt; done
; RUN: %{opt} %{prof} -poseidon-cache=%t/import/old 2>&1 | FileCheck --check-prefix=IMPORT-COMPUTED --implicit-check-not=imported %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/import/new -poseidon-import-accuracy-memo=%t/import/old > %t/import/import.log 2>&1
; RUN: FileCheck --check-prefixes=IMPORT,IMPORT-PLANTED --implicit-check-not="DOMERR matmul" --implicit-check-not="imported accuracy memo" %s < %t/import/import.log
; RUN: FileCheck --check-prefix=IMPORT-NEWER %s < %t/import/import.log
; RUN: %{opt} %{prof} -poseidon-cache=%t/import/new 2>&1 | FileCheck --check-prefix=IMPORT-PLANTED --implicit-check-not=DOMERR --implicit-check-not=imported --implicit-check-not=recomputing %s

; IMPORT-COMPUTED: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04
; IMPORT-COMPUTED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; IMPORT-COMPUTED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; IMPORT-COMPUTED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32

; IMPORT-COUNT-6: [poseidon] imported accuracy memo {{.*}}old{{/|\\}}matmulacc_{{[0-9A-F]+}}.txt as {{.*}}new{{/|\\}}matmulacc_{{[0-9A-F]+}}.json

; Each DOMERR line of the import compile, in order, and no other.
; IMPORT-NEWER-NOT: DOMERR
; IMPORT-NEWER: DOMERR ozakiII nm=8
; IMPORT-NEWER-NOT: DOMERR
; IMPORT-NEWER: DOMERR ozakiII nm=9
; IMPORT-NEWER-NOT: DOMERR
; IMPORT-NEWER: DOMERR ozakiII nm=10
; IMPORT-NEWER-NOT: DOMERR
; IMPORT-NEWER: DOMERR ozakiII nm=11
; IMPORT-NEWER-NOT: DOMERR
; IMPORT-NEWER: DOMERR ozakiII nm=12
; IMPORT-NEWER-NOT: DOMERR
; IMPORT-NEWER: DOMERR ozakiII nm=13
; IMPORT-NEWER-NOT: DOMERR
; IMPORT-NEWER: DOMERR ozakiII nm=14
; IMPORT-NEWER-NOT: DOMERR

; IMPORT-PLANTED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=0.000000e+00
; IMPORT-PLANTED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; IMPORT-PLANTED: #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=0.000000e+00
; IMPORT-PLANTED: Poseidon error-budget: matmul -> direct-dispatch cublas f16/f32

; A cache directory holding results in the format without keys is not read
; without -poseidon-import-accuracy-memo, so a compile that computes in front
; of it says so, once, naming the directory and the flag. With the flag given,
; or with no such file, it says nothing.
; RUN: mkdir %t/legacy %t/legacy/legacy %t/legacy/flagged %t/legacy/clean
; RUN: echo '0 0' > %t/legacy/legacy/matmulacc_0123456789ABCDEF.txt && cp %t/legacy/legacy/matmulacc_0123456789ABCDEF.txt %t/legacy/flagged/
; RUN: %{opt} %{prof} -poseidon-cache=%t/legacy/legacy 2>&1 | FileCheck --check-prefixes=LEGACY-NOTE,LEGACY-COMPUTED --implicit-check-not='matmulacc_*.txt' %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/legacy/flagged -poseidon-import-accuracy-memo=%t/legacy/flagged 2>&1 | FileCheck --check-prefix=LEGACY-COMPUTED --implicit-check-not='matmulacc_*.txt' %s
; RUN: %{opt} %{prof} -poseidon-cache=%t/legacy/clean 2>&1 | FileCheck --check-prefix=LEGACY-COMPUTED --implicit-check-not='matmulacc_*.txt' %s

; LEGACY-NOTE: [poseidon] {{.*}}legacy holds matrix accuracy results stored before the memo kept their keys (matmulacc_*.txt), which are not read; computing them again. -poseidon-import-accuracy-memo={{.*}}legacy takes them over instead.

; LEGACY-COMPUTED: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04
; LEGACY-COMPUTED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32
