; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/fmod_operator -poseidon-early-prune -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.5 -poseidon-cache=%t/pt -o %t/pt.ll 2>&1 | FileCheck --check-prefix=PT %s
; RUN: FileCheck --check-prefix=PT-IR %s < %t/pt.ll
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/fmod_operator %{opt} -poseidon-enable-algebraic=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/herbie -o %t/herbie.ll 2>&1 | FileCheck --check-prefix=HERBIE %s
; RUN: FileCheck --check-prefix=HERBIE-IR %s < %t/herbie.ll
; RUN: grep -v '^fmod,' %cost_models/cm_cpu_native.csv > %t/nofmod.csv
; RUN: not %{opt} -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-cost-model=%t/nofmod.csv -poseidon-cache=%t/nofmod -o /dev/null 2>&1 | FileCheck --check-prefix=NOFMOD %s
; RUN: grep -v '^fma,' %cost_models/cm_cpu_native.csv > %t/nofma.csv
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/fmod_operator not %{opt} -poseidon-enable-algebraic=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%t/nofma.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/nofma -o /dev/null 2>&1 | FileCheck --check-prefix=NOFMA %s

; An frem is fmod to Herbie, the accuracy model and the cost model (its fmod
; row, also for the frem a rewrite's fmod becomes); a model without the row is
; an error at the site that needs it. Profiles: host profiler, x = 0.5 + i/4,
; y = 0.5 + (i % 7)/4, i < 64.

define void @tester_frem(double %x, double %y, ptr %o) #0 {
entry:
  %m = fmul nnan double %x, %y
  %r = frem nnan double %m, 2.500000e+00
  %s = fadd nnan double %r, %x
  store double %s, ptr %o, align 8
  ret void
}

define void @tester_fma(double %x, double %y, ptr %o) #0 {
entry:
  %m = fmul nnan double %x, %y
  %s = fadd nnan double %m, %x
  store double %s, ptr %o, align 8
  ret void
}

define void @site_frem(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_frem, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

define void @site_fma(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_fma, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)
attributes #0 = { "target-cpu"="x86-64" }

; fmul 4 + fmod 8 + fadd 4 in double; 1 + 2 + 1 and three casts in float.
; PT: [poseidon] Optimizing preprocess_tester_frem
; PT: Initial ComputationCost: 1.600000e+01
; PT: [poseidon] precision candidate 'All FP64(0%) + FP32(100%)': initial=1.600000e+01 cand=7.000000e+00 exec=64 dcost=-5.760000e+02 dacc=3.098434e-05{{$}}
; PT: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; PT: [poseidon] pin: tester_frem:P0=All FP64(0%) + FP32(100%)
; PT-IR-LABEL: define void @preprocess_tester_frem(
; PT-IR-NEXT: entry:
; PT-IR-NEXT: [[XF:%.+]] = fptrunc {{.*}}double %x to float
; PT-IR-NEXT: [[YF:%.+]] = fptrunc {{.*}}double %y to float
; PT-IR-NEXT: [[M:%.+]] = fmul nnan float [[XF]], [[YF]]{{$}}
; PT-IR-NEXT: [[R:%.+]] = frem nnan float [[M]], 2.500000e+00
; PT-IR-NEXT: [[S:%.+]] = fadd nnan float [[R]], [[XF]]{{$}}
; PT-IR-NEXT: [[SD:%.+]] = fpext float [[S]] to double
; PT-IR-NEXT: store double [[SD]], ptr %o,
; PT-IR-NEXT: ret void

; HERBIE: :name "0" (+ (fmod (* v0 v1) 2.5) v0))
; The rewrite of tester_fma costs fma 4 + fmod 8 over fmul 4 + fadd 4.
; HERBIE: [poseidon] Optimizing preprocess_tester_fma
; HERBIE: {{^}}-3.731043e-14 2.560000e+02 1.500000e+00 9.843750e-01 (fmod.f64 (fma.f64 v0 v1 v0) #s(literal 1000 binary64)){{$}}
; HERBIE: Applying solution for (+ (* v0 v1) v0) --(0)-> (fmod.f64 (fma.f64 v0 v1 v0) #s(literal 1000 binary64)){{$}}
; HERBIE: [poseidon] pin: tester_fma:A0=(fmod.f64 (fma.f64 v0 v1 v0) #s(literal 1000 binary64)){{$}}
; HERBIE-IR-LABEL: define void @preprocess_tester_fma(
; HERBIE-IR: [[FMA:%.+]] = tail call nnan double @llvm.fma.f64(double %x, double %y, double %x)
; HERBIE-IR-NEXT: [[FMOD:%.+]] = frem nnan double [[FMA]], 1.000000e+03
; HERBIE-IR-NEXT: store double [[FMOD]], ptr %o,
; HERBIE-IR-NEXT: ret void

; NOFMOD: error: {{.*}}preprocess_tester_frem{{.*}}Poseidon: cost model {{.*}}nofmod.csv has no row 'fmod,double'; measure it with poseidon-calibrate{{$}}

; NOFMA: error: <unknown>:0:0: in function preprocess_tester_fma {{.*}}: Poseidon: cost model {{.*}}nofma.csv has no row 'fma,double'; measure it with poseidon-calibrate{{$}}
