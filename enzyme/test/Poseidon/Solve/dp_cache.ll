; DEFINE: %{pass} = %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true -S -o /dev/null
; DEFINE: %{cm} = -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; DEFINE: %{profiles} = -poseidon-profile-use=%S/../Inputs/pow_term/profiles
; DEFINE: %{out0} = -poseidon-profile-use=%S/../Inputs/pow_term/out0
; RUN: rm -rf %t && mkdir %t

; The DP table cache of the site below. %t/noscale.ll is the module without
; the call of scale_a.
; RUN: sed -e '/%ys = call double @scale_a/d' -e 's/fmul double %k, %ys/fmul double %k, %y/' -e '/^declare double @scale_[ab]/d' %s > %t/noscale.ll

; A table is cached per site with the key of everything it is solved from.
; Solved under profile A and then, against the same cache, under profile B of
; the same canonical form (Inputs/pow_term/out0, recorded with only the first
; output seeded), the site is solved again with the reason, and picks what a
; solve under B in an empty cache picks; a third compile reads the table back.
; A flag the table depends on (-poseidon-num-samples) re-solves too; one that
; only selects from the table (the budget, -poseidon-tau-cheapest) does not. A
; table in another cache format, or one that does not parse, is reported,
; solved again and overwritten.
; RUN: mkdir %t/key %t/key/old %t/key/bad
; RUN: %opt %t/noscale.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c 2>&1 | FileCheck --check-prefix=KEY-A %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c 2>&1 | FileCheck --check-prefixes=KEY-B,KEY-PICK-B %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/fresh 2>&1 | FileCheck --check-prefix=KEY-PICK-B %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c 2>&1 | FileCheck --check-prefixes=KEY-HIT,KEY-PICK-B %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c -poseidon-num-samples=512 2>&1 | FileCheck --check-prefix=KEY-SAMPLES %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.4 -poseidon-cache=%t/key/c -poseidon-num-samples=512 -poseidon-tau-cheapest=false 2>&1 | FileCheck --check-prefix=KEY-HIT %s
; RUN: echo '{"costModelFingerprint": "0", "candidateKeys": true, "functions": {"preprocess_tester": {"costToAccuracyMap": {"0": 0}, "costToSolutionMap": {"0": []}}}}' > %t/key/old/table.json
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/old 2>&1 | FileCheck --check-prefixes=KEY-OLD,KEY-PICK-B %s
; RUN: echo '{"functions": {"preprocess_tester": nan}}' > %t/key/bad/table.json
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/bad 2>&1 | FileCheck --check-prefixes=KEY-BAD,KEY-PICK-B %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/bad 2>&1 | FileCheck --check-prefixes=KEY-HIT,KEY-PICK-B %s

; Each run's lines in the order they are printed.
; KEY-A-NOT: DP cache entry
; KEY-HIT-NOT: DP cache entry
; KEY-B: [poseidon] DP cache entry for preprocess_tester was solved for a different profile; re-solving.
; KEY-SAMPLES: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-num-samples; re-solving.
; KEY-OLD: [poseidon] DP cache entry for preprocess_tester was written in another cache format; re-solving.
; KEY-BAD: [poseidon] DP cache entry for preprocess_tester is unreadable ({{.*}}table.json: {{.*}}); re-solving.
; KEY-A: Solving the DP for preprocess_tester.
; KEY-B: Solving the DP for preprocess_tester.
; KEY-SAMPLES: Solving the DP for preprocess_tester.
; KEY-OLD: Solving the DP for preprocess_tester.
; KEY-BAD: Solving the DP for preprocess_tester.
; KEY-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; KEY-A: DP table for preprocess_tester cached to file.
; KEY-B: DP table for preprocess_tester cached to file.
; KEY-OLD: DP table for preprocess_tester cached to file.
; KEY-BAD: DP table for preprocess_tester cached to file.
; KEY-HIT-NOT: cached to file
; KEY-A: Applying solution for CS: All FP64(70%) + FP32(30%)
; KEY-PICK-B: Applying solution for CS: Funcs FP64(70%) + FP32(30%)

; Each compile starts from the first one's cache and changes one input of the
; table (a profile byte, a cost-model row, a solve flag, a constant of the
; body, the function it calls): it names that input and solves again. An
; unchanged compile reads the table back.
; RUN: mkdir %t/parts %t/parts/p
; RUN: sed '0,/Exec = 64/s//Exec = 65/' %S/../Inputs/pow_term/profiles/preprocess_tester.fpprofile > %t/parts/p/preprocess_tester.fpprofile
; RUN: sed 's/^fadd,float,1$/fadd,float,2/' %cost_models/cm_cpu_native.csv > %t/parts/cm.csv
; RUN: sed 's/fmul double %%p, 1.000000e-20/fmul double %%p, 2.000000e-20/' %s > %t/parts/const.ll
; RUN: sed 's/call double @scale_a(/call double @scale_b(/' %s > %t/parts/callee.ll
; RUN: not cmp -s %S/../Inputs/pow_term/profiles/preprocess_tester.fpprofile %t/parts/p/preprocess_tester.fpprofile
; RUN: not cmp -s %cost_models/cm_cpu_native.csv %t/parts/cm.csv
; RUN: not cmp -s %s %t/parts/const.ll
; RUN: not cmp -s %s %t/parts/callee.ll
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/base 2>&1 | FileCheck --check-prefixes=PARTS-FRESH,PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/same
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/same 2>&1 | FileCheck --check-prefix=PARTS-HIT %s
; RUN: cp -r %t/parts/base %t/parts/profile
; RUN: %opt %s %{pass} -poseidon-profile-use=%t/parts/p %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/profile 2>&1 | FileCheck --check-prefixes=PARTS-PROFILE,PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/costmodel
; RUN: %opt %s %{pass} %{profiles} -poseidon-cost-model=%t/parts/cm.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/costmodel 2>&1 | FileCheck --check-prefixes=PARTS-COSTMODEL,PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/flag
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/flag -poseidon-random-seed=7 2>&1 | FileCheck --check-prefixes=PARTS-FLAG,PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/const
; RUN: %opt %t/parts/const.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/const 2>&1 | FileCheck --check-prefixes=PARTS-BODY,PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/callee
; RUN: %opt %t/parts/callee.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/callee 2>&1 | FileCheck --check-prefixes=PARTS-BODY,PARTS-SOLVED %s

; PARTS-FRESH-NOT: DP cache entry
; PARTS-HIT-NOT: DP cache entry
; PARTS-PROFILE: [poseidon] DP cache entry for preprocess_tester was solved for a different profile; re-solving.
; PARTS-COSTMODEL: [poseidon] DP cache entry for preprocess_tester was solved for a different cost model; re-solving.
; PARTS-FLAG: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-random-seed; re-solving.
; PARTS-BODY: [poseidon] DP cache entry for preprocess_tester was solved for a different function body; re-solving.
; PARTS-SOLVED: Solving the DP for preprocess_tester.
; PARTS-SOLVED: DP table for preprocess_tester cached to file.
; PARTS-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; PARTS-HIT-NOT: Solving the DP

; A cached table whose key has parts this compile does not names the first of
; them by name, whatever order the stored key object holds them in.
; RUN: mkdir %t/extra
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/extra/c 2>&1 | FileCheck --check-prefix=EXTRA-SOLVED %s
; RUN: sed 's/"key": {/"key": {"zz extra": "1", "aa extra": "1", "mm extra": "1",/' %t/extra/c/table.json > %t/extra/table.json
; RUN: mv %t/extra/table.json %t/extra/c/table.json
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/extra/c 2>&1 | FileCheck --check-prefixes=EXTRA,EXTRA-SOLVED %s

; EXTRA: [poseidon] DP cache entry for preprocess_tester was solved for a different aa extra; re-solving.
; EXTRA-SOLVED: DP table for preprocess_tester cached to file.

; A table entry whose key names no compiler is a miss. The cached table gets
; other numbers (the point the solve picks made the worst) and loses every key
; part naming the compiler; the next compile solves again.
; RUN: mkdir %t/compiler
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/compiler/c 2>&1 | FileCheck --check-prefixes=COMPILER-SOLVE,COMPILER-PICK %s
; RUN: sed -e '/^ *"compiler[^"]*":/d' -e 's/^\( *"-536870912": \)[0-9][0-9.e+-]*,$/\11,/' %t/compiler/c/table.json > %t/compiler/table.json && mv %t/compiler/table.json %t/compiler/c/table.json
; RUN: FileCheck --check-prefix=COMPILER-EDITED %s < %t/compiler/c/table.json
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/compiler/c 2>&1 | FileCheck --check-prefixes=COMPILER-FOREIGN,COMPILER-PICK %s

; COMPILER-EDITED: "-536870912": 1,

; COMPILER-SOLVE: Solving the DP for preprocess_tester.
; COMPILER-SOLVE: DP table for preprocess_tester cached to file.

; COMPILER-FOREIGN-NOT: Loaded DP tables from cache
; COMPILER-FOREIGN: [poseidon] DP cache entry for preprocess_tester {{.*}}; re-solving.
; COMPILER-FOREIGN: Solving the DP for preprocess_tester.

; COMPILER-PICK: Applying solution for CS: Funcs FP64(70%) + FP32(30%) (#10)

; A table cut off in the middle (an interrupted write) is reported, solved
; again and overwritten; the compile after that reads it back.
; RUN: mkdir %t/trunc
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=TRUNC-SOLVE,TRUNC-PICK %s
; RUN: head -c $(( $(wc -c < %t/trunc/c/table.json) / 2 )) %t/trunc/c/table.json > %t/trunc/half.json && mv %t/trunc/half.json %t/trunc/c/table.json
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=TRUNC,TRUNC-SOLVE,TRUNC-PICK %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=TRUNC-HIT,TRUNC-PICK %s

; TRUNC: [poseidon] DP cache entry for preprocess_tester is unreadable ({{.*}}table.json: {{.*}}); re-solving.
; TRUNC-SOLVE: Solving the DP for preprocess_tester.
; TRUNC-SOLVE: DP table for preprocess_tester cached to file.
; TRUNC-HIT-NOT: DP cache entry
; TRUNC-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; TRUNC-HIT-NOT: cached to file
; TRUNC-PICK: Applying solution for CS: Funcs FP64(70%) + FP32(30%) (#10)

define void @tester(double %T, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %l = call double @llvm.log.f64(double %T)
  %nl = fneg double %l
  %e = call double @llvm.fma.f64(double %nl, double 1.790000e-02, double -1.484800e+00)
  %p = call double @llvm.pow.f64(double %T, double %e)
  %k = fmul double %p, 1.000000e-20
  %ys = call double @scale_a(double %y)
  %r1 = fmul double %k, %ys
  %h = fmul double %nl, %y
  %r2 = fadd double %h, %T
  store double %r1, ptr %o1, align 8
  store double %r2, ptr %o2, align 8
  ret void
}

define void @site(double %T, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %T, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare double @scale_a(double)
declare double @scale_b(double)
declare double @llvm.log.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.pow.f64(double, double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }
