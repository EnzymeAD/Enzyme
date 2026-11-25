; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/unmodelled_operand -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-budget=-0.3 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-print -poseidon-cache=%t -S -o %t/out.ll 2>&1 | FileCheck --check-prefix=SOLVE %s
; RUN: FileCheck --check-prefix=IR %s < %t/out.ll

; An instruction with an operand the accuracy model has no node for, an undef
; or poison value, a constant expression or an argument of a type other than
; float and double, is not an operation of an FP subgraph: it is neither
; profiled nor rewritten, and its result is an input of the operations that use
; it.
;
; Profiles: this module through the host profiler, driven by
;   for (int i = 0; i < 64; ++i) {
;     double x = 0.5 + i * 0.25, y = 0.5 + (i % 7) * 0.25;
;     double o, d = 1.0, u, du = 0.0;
;     site_undef(x, y, &o, &d, &u, &du);
;     d = 1.0;
;     site_ce(x, y, &o, &d);
;     d = 1.0;
;     site_powi(x, y, 1 + i % 3, &o, &d);
;   }

@g = global i64 0

define void @tester_undef(double %x, double %y, ptr %o, ptr %u) #0 {
entry:
  %a = fadd double %x, %y
  %b = fmul double %a, %a
  %p = call double @llvm.pow.f64(double %b, double undef)
  store double %p, ptr %u, align 8
  %s = fmul double %b, %x
  %t = fadd double %s, %y
  store double %t, ptr %o, align 8
  ret void
}

define void @tester_ce(double %x, double %y, ptr %o) #0 {
entry:
  %a = fadd double %x, %y
  %b = fmul double %a, %a
  %p = fmul double %b, bitcast (i64 ptrtoint (ptr @g to i64) to double)
  %s = fmul double %p, %x
  %t = fadd double %s, %y
  store double %t, ptr %o, align 8
  ret void
}

define void @tester_powi(double %x, double %y, i32 %n, ptr %o) #0 {
entry:
  %a = fadd double %x, %y
  %b = fmul double %a, %a
  %p = call double @llvm.powi.f64.i32(double %b, i32 %n)
  %s = fmul double %p, %x
  %t = fadd double %s, %y
  store double %t, ptr %o, align 8
  ret void
}

define void @site_undef(double %x, double %y, ptr %o, ptr %do, ptr %u, ptr %du) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_undef, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do, metadata !"enzyme_dup", ptr %u, ptr %du)
  ret void
}

define void @site_ce(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_ce, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

define void @site_powi(double %x, double %y, i32 %n, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_powi, double %x, double %y, i32 %n, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

declare double @llvm.pow.f64(double, double)
declare double @llvm.powi.f64.i32(double, i32)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; SOLVE: [poseidon] Optimizing preprocess_tester_undef
; SOLVE: Found a subgraph with 4 operations and 2 inputs and 2 outputs
; SOLVE: Applying solution for precision change All FP64(0%) + FP32(100%)
; SOLVE: [poseidon] Optimizing preprocess_tester_ce
; SOLVE: Found a subgraph with 2 operations and 3 inputs and 1 outputs
; SOLVE: Applying solution for precision change All FP64(0%) + FP32(100%)
; SOLVE: [poseidon] Optimizing preprocess_tester_powi
; SOLVE: Found a subgraph with 2 operations and 3 inputs and 1 outputs
; SOLVE: Applying solution for precision change All FP64(0%) + FP32(100%)

; IR-LABEL: define void @preprocess_tester_undef(
; IR: [[B:%.+]] = fpext float %{{.+}} to double
; IR-NEXT: call double @llvm.pow.f64(double [[B]], double undef)
; IR-LABEL: define void @preprocess_tester_ce(
; IR: [[B:%.+]] = fpext float %{{.+}} to double
; IR-NEXT: [[P:%.+]] = fmul double [[B]], bitcast (i64 ptrtoint (ptr @g to i64) to double)
; IR-NEXT: fptrunc double [[P]] to float
; IR-LABEL: define void @preprocess_tester_powi(
; IR: [[B:%.+]] = fpext float %{{.+}} to double
; IR-NEXT: [[P:%.+]] = tail call double @llvm.powi.f64.i32(double [[B]], i32 %n)
; IR-NEXT: fptrunc double [[P]] to float
