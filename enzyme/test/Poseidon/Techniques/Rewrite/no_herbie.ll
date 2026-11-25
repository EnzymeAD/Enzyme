; DEFINE: %{flags} = -poseidon-profile-use=%S/../../Inputs/joint_profiles -poseidon-early-prune \
; DEFINE:   -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; RUN: rm -rf %t && mkdir %t && printf 'log\nnone\n' > %t/answers
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{flags} -poseidon-herbie-binary=%t/no-herbie -poseidon-cache=%t/c -pass-remarks-missed=poseidon -disable-output 2>&1 | FileCheck --check-prefix=DEFAULT --implicit-check-not="no algebraic rewrites" --implicit-check-not=warning: --implicit-check-not=error: %s
; RUN: %clang -O1 -c -x ir --target=x86_64-unknown-linux-gnu %s -o /dev/null %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/joint_profiles -mllvm -poseidon-early-prune -mllvm -poseidon-enable-precision=false -mllvm -poseidon-cost-model=%cost_models/cm_cpu_native.csv -mllvm -poseidon-herbie-binary=%t/no-herbie -mllvm -poseidon-cache=%t/c -Rpass-missed=poseidon 2>&1 | FileCheck --check-prefix=DEFAULT --implicit-check-not="no algebraic rewrites" --implicit-check-not=warning: --implicit-check-not=error: %s
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{flags} -poseidon-herbie-binary=%t/no-herbie -poseidon-cache=%t/c -poseidon-strict-mode -disable-output 2>&1 | FileCheck --check-prefix=STRICT %s
; RUN: env FAKE_HERBIE_DIR=%t %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" %{flags} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/fake -pass-remarks-missed=poseidon -disable-output 2>&1 | FileCheck --check-prefix=HERBIE --implicit-check-not="no algebraic rewrites" %s

; Algebraic rewriting left to its default is on when a Herbie can run. With
; none, the compile proposes no rewrites and says so once, in a
; missed-optimization remark at the first site it solves, not in a warning or
; an error. Under -poseidon-strict-mode the missing Herbie is an error, as is
; asking for rewrites with -poseidon-enable-rewrite (herbie_cannot_run.ll).

; DEFAULT: remark: {{.*}}Poseidon proposes no algebraic rewrites: there is no Herbie at {{.*}}/no-herbie{{( \[-Rpass-missed=poseidon\])?$}}

; STRICT: error: {{.*}}Poseidon: no Herbie answer for subgraph 0 of preprocess_tester_a is cached in {{.*}}, and there is no Herbie at {{.*}}/no-herbie to compute it; copy here the -poseidon-cache directory of a compile that ran Herbie, or turn algebraic rewriting off with -poseidon-enable-rewrite=0{{$}}

; HERBIE: fake-herbie: report

target triple = "x86_64-unknown-linux-gnu"

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
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_a, double %x, double %y)
  ret double %0
}

define double @site_b(double %x, double %y) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_b, double %x, double %y)
  ret double %0
}

declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }
