; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/array_cores/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/array_cores/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t -S -o /dev/null 2>&1 | FileCheck --check-prefix=FIRST %s
; RUN: rm %t/cachedHerbieOutput_*
; RUN: env FAKE_HERBIE_ARRAY=timeout %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/array_cores/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/array_cores/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t -S -o /dev/null 2>&1 | FileCheck --check-prefix=DRIFT %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/array_cores/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/array_cores/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=100000000 -poseidon-cache=%t -S -o /dev/null 2>&1 | FileCheck --check-prefix=HIT %s
; REQUIRES: poseidon

; A DP table is keyed on the candidates it was solved over as well as on the
; inputs that decide them, because Herbie can answer the same cores
; differently: here its results are deleted and the array core times out on
; the search that replaces them, so the outputs are sent separately and the
; site has other candidates under unchanged inputs. The table is solved again
; rather than replayed against candidates it never priced.

define void @tester(double %x, double %y, ptr %o1, ptr %o2, ptr %o3) #0 {
entry:
  %xx = fmul double %x, %x
  %yy = fmul double %y, %y
  %s = fadd double %xx, %yy
  %r = call double @llvm.sqrt.f64(double %s)
  %d = fsub double %r, %x
  %h = fmul double %d, 5.000000e-01
  %z = fmul double %xx, 2.500000e+00
  store double %d, ptr %o1, align 8
  store double %h, ptr %o2, align 8
  store double %z, ptr %o3, align 8
  ret void
}

define void @site(double %x, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2, ptr %o3, ptr %do3) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2, metadata !"enzyme_dup", ptr %o3, ptr %do3)
  ret void
}

declare double @llvm.sqrt.f64(double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; FIRST: Solving the DP for preprocess_tester.
; FIRST: Applying solution for (let* ((t1 (- (sqrt (+ (* v0 v0) (* v1 v1))) v0))) (array t1 (* t1 0.5)))

; DRIFT: array core returned nothing
; DRIFT: [poseidon] DP cache entry for preprocess_tester was solved for a different candidate set; re-solving.
; DRIFT: Solving the DP for preprocess_tester.
; DRIFT-NOT: (array t1
; DRIFT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)

; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; HIT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)
