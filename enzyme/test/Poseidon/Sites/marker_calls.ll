; DEFINE: %{opt} = %loadPoseidon -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; RUN: rm -rf %t && split-file %s %t
; RUN: cat %t/tester.ll %t/nothing.ll > %t/nothing_full.ll
; RUN: cat %t/tester.ll %t/two.ll > %t/two_full.ll
; RUN: cat %t/tester.ll %t/guid.ll > %t/guid_full.ll
; RUN: %opt < %t/noaction.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck --check-prefix=NOACTION %s
; RUN: %opt < %t/nothing_full.ll -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" %{opt} -poseidon-enable-precision=false -poseidon-cache= -S | FileCheck --check-prefix=NOTHING %s
; RUN: %opt < %t/two_full.ll -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" %{opt} -poseidon-enable-precision=true -poseidon-precision-expansions=false -poseidon-cache=%t/two -S -poseidon-print 2>&1 | FileCheck --check-prefix=TWO %s
; RUN: %if guid-metadata %{ %opt < %t/guid_full.ll -passes="assign-guid,poseidon,poseidon-finalize" %{opt} -poseidon-enable-precision=true -poseidon-precision-expansions=false -poseidon-cache=%t/guid -S | FileCheck --check-prefix=GUID %s %}

; Marker calls of __poseidon_fp_optimize on a host function.

; Without a profile action the marker calls the function.
; NOACTION: define double @test_profile(double %x, double %y)
; NOACTION-NEXT: entry:
; NOACTION-NEXT:   %[[fadd:.+]] = call double @tester(double %x, double %y)
; NOACTION-NEXT:   ret double %[[fadd]]
; NOACTION-NEXT: }

; With nothing applied the wrapper calls the original body and the clone is
; dropped. enzyme_not_overwritten modifies the marker that follows it, on
; either side of it, as Enzyme's handleArguments decodes it.
; NOTHING: define double @test_opt(double %x, double %y, double %z, double %xs, double %ys)
; NOTHING-NEXT: entry:
; NOTHING-NEXT:   %[[call:.+]] = call double @tester(double %x, double %y, double %z)
; NOTHING-NEXT:   ret double %[[call]]
; NOTHING-NOT: @preprocess_tester

; One body reached from two marker calls (a marker in a template body
; instantiated for several kernels): both sites are served by the one profile
; the run recorded for the body, and the second reuses the first one's solve.
; TWO-NOT: no profile at
; TWO-NOT: left unchanged
; TWO: [[BODY:preprocess_tester[^ :]*]]: tau=1.000000e-02
; TWO: tester: same body and target as an earlier site; reusing its solve ([[BODY]])
; TWO-LABEL: define double @site_b(
; TWO: call {{.*}}@[[BODY]](

; Where GUIDs were assigned before the pass (a full-LTO link of pre-link
; bitcode), the clone gets the GUID of its own name, not the copied !guid.
; GUID: define double @tester({{.*}}!guid ![[ORIGINAL:[0-9]+]]
; GUID: define {{.*}}@preprocess_tester({{.*}}!guid ![[CLONE:[0-9]+]]
; GUID-DAG: ![[ORIGINAL]] = !{i64 -2743526145958358539}
; GUID-DAG: ![[CLONE]] = !{i64 3151465777686795207}

;--- noaction.ll
define double @tester(double %x, double %y) {
entry:
  %0 = fadd fast double %x, %y
  ret double %0
}

define double @test_profile(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y, metadata !"poseidon_tau", double 1.0e-6)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)

;--- tester.ll
define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

;--- nothing.ll
define double @test_opt(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, metadata !"enzyme_not_overwritten", metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", metadata !"enzyme_not_overwritten", double %y, double %ys, double %z, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

;--- two.ll
define double @site_a(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", double %y, double %ys, double %z, metadata !"poseidon_tau", double 1.0e-2)
  ret double %0
}

define double @site_b(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", double %y, double %ys, double %z, metadata !"poseidon_tau", double 1.0e-2)
  ret double %0
}

;--- guid.ll
define double @site(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", double %y, double %ys, double %z, metadata !"poseidon_tau", double 1.0e-2)
  ret double %0
}
