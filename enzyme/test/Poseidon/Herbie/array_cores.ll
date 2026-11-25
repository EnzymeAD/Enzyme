; DEFINE: %{opt} = %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/array_cores/profiles \
; DEFINE:   -poseidon-enable-herbie=true \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/array_cores/fake-herbie.sh \
; DEFINE:   -poseidon-herbie-processes=1 -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/arrays -poseidon-herbie-arrays -poseidon-print 2>&1 | FileCheck --check-prefix=ARRAY %s
; RUN: env FAKE_HERBIE_ARRAY=timeout %{opt} -poseidon-cache=%t/fallback -poseidon-herbie-arrays -poseidon-print 2>&1 | FileCheck --check-prefix=FALLBACK %s
; RUN: %{opt} -poseidon-cache=%t/off -poseidon-herbie-arrays=false 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: not %{opt} -poseidon-cache=%t/grouping -poseidon-herbie-arrays -poseidon-herbie-array-grouping=Inputs 2>&1 \
; RUN:   | FileCheck --check-prefix=GROUPING %s

; Both outputs read sqrt(x*x + y*y) - x. With -poseidon-herbie-arrays the
; subgraph goes to Herbie as one array core per precision, the shared subterm
; let*-bound and each element weighted by its output's sensitivity. The array
; rewrite covers both outputs and is lowered with the shared subterm computed
; once. %z reads x alone, so it is no element of that array; its per-output
; cores go to Herbie in the same invocation as the array cores. An array core
; that returns nothing is replaced by the per-output cores sent with
; -poseidon-herbie-arrays=false.

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

; ARRAY: fake-herbie: {{.*}} --num-enodes 8000
; ARRAY-NEXT: (FPCore (v0 v1) {{.*}} :precision binary64 {{.*}} :herbie-weights (1.500000e+00 5.000000e-01) :name "0" (let* ((t1 (- (sqrt (+ (* v0 v0) (* v1 v1))) v0))) (array t1 (* t1 0.5))))
; ARRAY-NEXT: (FPCore (v0 v1) {{.*}} :precision binary32 {{.*}} :name "1" (let* ((t1
; ARRAY-NEXT: (FPCore (v0) {{.*}} :name "2" (* (* v0 v0) 2.5))
; ARRAY-NEXT: (FPCore (v0) {{.*}} :name "3" (* (* v0 v0) 2.5))
; ARRAY-NOT: fake-herbie:
; ARRAY: Applying solution for (let* {{.*}} --(0)-> (array<binary64:binary64>
; ARRAY-NOT: Applying solution for
; ARRAY: define void @preprocess_tester
; ARRAY: call double @hypot
; ARRAY-NOT: call double @hypot
; ARRAY: %[[D:[a-z.0-9]+]] = fdiv double
; ARRAY-NEXT: fmul double %[[D]], 5.000000e-01
; ARRAY: store double %[[D]], ptr %o1

; FALLBACK: fake-herbie:
; FALLBACK-NEXT: (FPCore (v0 v1) {{.*}} (let* ((t1
; FALLBACK: array core returned nothing for %d = fsub double %r, %x{{.*}}; sending it separately
; FALLBACK: array core returned nothing for %h = fmul double %d, 5.000000e-01{{.*}}; sending it separately
; FALLBACK: fake-herbie:
; FALLBACK-NEXT: (FPCore (v0 v1) {{.*}} :name "0" (- (sqrt (+ (* v0 v0) (* v1 v1))) v0))
; FALLBACK: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)->
; FALLBACK: Applying solution for (* (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) 0.5) --(0)->

; OFF-NOT: (array
; OFF: fake-herbie: {{.*}} --num-enodes 8000
; OFF-NEXT: (FPCore (v0 v1) {{.*}} :name "0" (- (sqrt (+ (* v0 v0) (* v1 v1))) v0))
; OFF-NOT: (array

; The grouping is spelled exactly; any other spelling is refused.
; GROUPING: Poseidon: -poseidon-herbie-array-grouping=Inputs is not 'inputs' or 'executions'
