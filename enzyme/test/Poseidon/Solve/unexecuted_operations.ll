; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/unexecuted_operations \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-tau=1e-4 -pass-remarks=poseidon -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-loose-coverage -poseidon-cache=%t/loose -o %t/loose.ll 2>&1 | FileCheck --check-prefix=LOOSE %s
; RUN: FileCheck --check-prefix=IR %s < %t/loose.ll
; RUN: not %{opt} -poseidon-cache=%t/strict -o /dev/null 2>&1 | FileCheck --check-prefix=STRICT %s

; The profiling run took only the scalar branch, so the two operations of the
; matrix branch have no profile record. With -poseidon-loose-coverage they are
; left as written, as an unexecuted matrix product is, and the subgraph of the
; executed operations around them ends at them: priced from the profile alone,
; it is rewritten. Without the flag such an operation is an error, as for a
; matrix product.
;
; Profile: this module through the host profiler with
; Inputs/unexecuted_operations/driver.c.

define void @tester(double %x, double %y, i32 %k, ptr %m, ptr %o) #0 {
entry:
  %a = fmul double %x, %y
  %b = fadd double %a, %x
  %matrix = icmp ne i32 %k, 0
  br i1 %matrix, label %mat, label %scal

mat:
  %mv = load double, ptr %m, align 8, !tbaa !0
  %u = fmul double %b, %mv
  %v = fadd double %u, %a
  store double %v, ptr %o, align 8, !tbaa !0
  ret void

scal:
  %w = fmul double %b, %y
  %z = fadd double %w, %x
  store double %z, ptr %o, align 8, !tbaa !0
  ret void
}

define void @site(double %x, double %y, i32 %k, ptr %m, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, i32 %k, metadata !"enzyme_const", ptr %m, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}

; LOOSE: [poseidon] preprocess_tester: 2 floating-point operation(s) the profiling run never executed are left as written (no source location): no profile gives their ranges, cost or error, so the subgraphs of the executed operations end at them and the error a rewrite of those passes on to them is not modelled
; LOOSE: remark: <unknown>:0:0: Poseidon runs all 4 operations in float;
; LOOSE: [poseidon] pin: tester:PT0=All FP64(0%) + FP32(100%)

; IR-LABEL: define void @preprocess_tester(
; IR: fmul float
; IR: fadd float
; IR: mat:
; IR: %u = fmul double
; IR-NEXT: %v = fadd double
; IR: scal:
; IR-NEXT: fmul float
; IR-NEXT: fadd float

; STRICT: error: {{.*}}Poseidon: the profile does not cover this operation: the profiling run never executed it. Profile an input that runs it, or pass -poseidon-loose-coverage to leave it as written.
