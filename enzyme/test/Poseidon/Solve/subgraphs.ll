; DEFINE: %{chain} = %opt %t/chain.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/subgraph_split -poseidon-early-prune -poseidon-enable-algebraic=false \
; DEFINE:   -poseidon-enable-precision=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=-0.05 -poseidon-print -S
; DEFINE: %{unexec} = %opt %t/unexecuted.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/unexecuted_operations -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-tau=1e-4 -pass-remarks=poseidon \
; DEFINE:   -pass-remarks-missed=poseidon -S
; FP subgraph formation: chain splitting, operands with no model node, and
; operations the profile never executed.
; RUN: rm -rf %t && split-file %s %t

; Chain splitting: %t has three users in the FP subgraph and four operations
; upstream used by nothing else, so the 9-operation subgraph splits at %t into
; two units exactly when both thresholds (3 uses, 4 ops) are met; one more of
; either leaves it whole and the module byte-identical. Profile: host profiler
; on x = 1 + i/8, y = 0.5 + (i % 7)/4, i < 64.

; RUN: %{chain} -poseidon-cache=%t/default -o %t/default.ll 2>&1 | FileCheck --check-prefix=NOSPLIT %s
; RUN: %{chain} -poseidon-cache=%t/split -poseidon-subgraph-split-uses=3 -poseidon-subgraph-split-ops=4 -o %t/split.ll 2>&1 | FileCheck --check-prefix=SPLIT %s
; RUN: %{chain} -poseidon-cache=%t/uses4 -poseidon-subgraph-split-uses=4 -poseidon-subgraph-split-ops=4 -o %t/uses4.ll 2>&1 | FileCheck --check-prefix=NOSPLIT %s
; RUN: diff %t/default.ll %t/uses4.ll
; RUN: %{chain} -poseidon-cache=%t/ops5 -poseidon-subgraph-split-uses=3 -poseidon-subgraph-split-ops=5 -o %t/ops5.ll 2>&1 | FileCheck --check-prefix=NOSPLIT %s
; RUN: diff %t/default.ll %t/ops5.ll

; NOSPLIT: Found a subgraph with 9 operations and 2 inputs and 2 outputs
; NOSPLIT-NOT: Bottleneck:
; NOSPLIT: [poseidon] After splitting, have 1 subgraphs in preprocess_tester
; NOSPLIT: Initial AccuracyCost: 8.837436e-11
; NOSPLIT-NEXT: Initial ComputationCost: 3.600000e+01
; NOSPLIT: Total candidate compositions: 1.000000e+01
; NOSPLIT: Minimum accuracy cost within budget: 2.672498e-03
; NOSPLIT-NEXT: Computation cost budget used: -5.555556e-02
; NOSPLIT: Applying solution for precision change All FP64(80%) + FP32(20%) (#7)

; SPLIT: Found a subgraph with 9 operations and 2 inputs and 2 outputs
; SPLIT: Bottleneck:   %t = fsub double %c, %y
; SPLIT-NEXT: Num of operations that would be moved: 4 (>=4)
; SPLIT-NEXT: Num of internal uses: 3 (>=3)
; SPLIT-NEXT: Operations that would be moved:
; SPLIT-NEXT: %t = fsub double %c, %y
; SPLIT-NEXT: %c = fmul double %b, %b
; SPLIT-NEXT: %b = fadd double %a, %x
; SPLIT-NEXT: %a = fmul double %x, %y
; SPLIT: [poseidon] After splitting, have 2 subgraphs in preprocess_tester
; SPLIT: Initial AccuracyCost: 5.730790e-11
; SPLIT-NEXT: Initial ComputationCost: 1.600000e+01
; SPLIT: Initial AccuracyCost: 2.614053e-11
; SPLIT-NEXT: Initial ComputationCost: 2.000000e+01
; SPLIT: Total candidate compositions: 3.000000e+01
; SPLIT: Minimum accuracy cost within budget: 2.672498e-03
; SPLIT-NEXT: Computation cost budget used: -5.555556e-02
; SPLIT: Applying solution for precision change All FP64(60%) + FP32(40%) (#3)

; An operation with an operand the accuracy model has no node for (undef, a
; constant expression, a non-FP argument) is left out of the FP subgraph and
; its result is an input. Profiles: host profiler, x = 0.5 + i/4, y = 0.5 +
; (i % 7)/4, n = 1 + i % 3, i < 64.
; RUN: %opt %t/unmodelled.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/unmodelled_operand -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-cost-budget=-0.3 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-print -poseidon-cache=%t -S -o %t/out.ll 2>&1 | FileCheck --check-prefix=UMSOLVE %s
; RUN: FileCheck --check-prefix=UMIR %s < %t/out.ll

; UMSOLVE: [poseidon] Optimizing preprocess_tester_undef
; UMSOLVE: Found a subgraph with 4 operations and 2 inputs and 2 outputs
; UMSOLVE: Applying solution for precision change All FP64(0%) + FP32(100%)
; UMSOLVE: [poseidon] Optimizing preprocess_tester_ce
; UMSOLVE: Found a subgraph with 2 operations and 3 inputs and 1 outputs
; UMSOLVE: Applying solution for precision change All FP64(0%) + FP32(100%)
; UMSOLVE: [poseidon] Optimizing preprocess_tester_powi
; UMSOLVE: Found a subgraph with 2 operations and 3 inputs and 1 outputs
; UMSOLVE: Applying solution for precision change All FP64(0%) + FP32(100%)

; UMIR-LABEL: define void @preprocess_tester_undef(
; UMIR: [[B:%.+]] = fpext float %{{.+}} to double
; UMIR-NEXT: call double @llvm.pow.f64(double [[B]], double undef)
; UMIR-LABEL: define void @preprocess_tester_ce(
; UMIR: [[B:%.+]] = fpext float %{{.+}} to double
; UMIR-NEXT: [[P:%.+]] = fmul double [[B]], bitcast (i64 ptrtoint (ptr @g to i64) to double)
; UMIR-NEXT: fptrunc double [[P]] to float
; UMIR-LABEL: define void @preprocess_tester_powi(
; UMIR: [[B:%.+]] = fpext float %{{.+}} to double
; UMIR-NEXT: [[P:%.+]] = tail call double @llvm.powi.f64.i32(double [[B]], i32 %n)
; UMIR-NEXT: fptrunc double [[P]] to float

; Operations the profiling run never executed (the matrix branch) are left as
; written with a remark naming their line; the executed subgraph around them is
; rewritten. Profile: host profiler, scalar branch only.
; RUN: %{unexec} -poseidon-cache=%t/loose -poseidon-print -o %t/loose.ll 2>&1 | FileCheck --check-prefix=LOOSE %s
; RUN: FileCheck --check-prefix=UXIR %s < %t/loose.ll

; LOOSE: remark: matrix.c:1:0: Poseidon left 2 floating-point operation(s) as written that the profiling run never executed (matrix.c:9); profile an input that runs them{{$}}
; LOOSE: [poseidon] preprocess_tester: tau=1.000000e-04 S=3.818938e+03 A0=2.850360e-11 selected cost=-4.375000e-01 accCost=2.794345e-02 rel=7.317073e-06
; LOOSE-NEXT: [poseidon] preprocess_tester: memory traffic not priced, the cost model has no stream rows; the arithmetic decides
; LOOSE-NEXT: remark: <unknown>:0:0: Poseidon runs all 4 operations in float; modelled error 7.5e-15 -> 7.3e-06, cost 1 -> 0.5625 (-44%) [P0_0]
; LOOSE: [poseidon] pin: tester:P0=All FP64(0%) + FP32(100%)

; UXIR-LABEL: define void @preprocess_tester(
; UXIR-NEXT: entry:
; UXIR-NEXT: [[XF:%.+]] = fptrunc double %x to float
; UXIR-NEXT: [[YF:%.+]] = fptrunc double %y to float
; UXIR-NEXT: [[A:%.+]] = fmul float [[XF]], [[YF]]{{$}}
; UXIR-NEXT: [[B:%.+]] = fadd float [[A]], [[XF]]{{$}}
; UXIR: [[R:%.+]] = phi double [ [[ZD:%.+]], %scal ], [ %v, %mat ]
; UXIR-NEXT: store double [[R]], ptr %o
; UXIR: mat:
; UXIR-NEXT: [[BD:%.+]] = fpext float [[B]] to double
; UXIR-NEXT: [[AD:%.+]] = fpext float [[A]] to double
; UXIR-NEXT: %mv = load double, ptr %m
; UXIR-NEXT: %u = fmul double %mv, [[BD]],
; UXIR-NEXT: %v = fadd double %u, [[AD]],
; UXIR: scal:
; UXIR-NEXT: [[W:%.+]] = fmul float [[B]], [[YF]]{{$}}
; UXIR-NEXT: [[Z:%.+]] = fadd float [[W]], [[XF]]{{$}}
; UXIR-NEXT: [[ZD]] = fpext float [[Z]] to double

;--- chain.ll
define void @tester(double %x, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %a = fmul double %x, %y
  %b = fadd double %a, %x
  %c = fmul double %b, %b
  %t = fsub double %c, %y
  %u1 = fmul double %t, %x
  %u2 = fadd double %t, %y
  %u3 = fdiv double %t, %x
  %r1 = fadd double %u1, %u2
  %r2 = fmul double %u3, %u2
  store double %r1, ptr %o1, align 8
  store double %r2, ptr %o2, align 8
  ret void
}

define void @site(double %x, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)
attributes #0 = { "target-cpu"="x86-64" }

;--- unmodelled.ll
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

;--- unexecuted.ll
define void @tester(double %x, double %y, i32 %k, ptr %m, ptr %o) #0 !dbg !5 {
entry:
  %a = fmul double %x, %y
  %b = fadd double %a, %x
  %matrix = icmp ne i32 %k, 0
  br i1 %matrix, label %mat, label %scal

mat:
  %mv = load double, ptr %m, align 8, !tbaa !0
  %u = fmul double %b, %mv, !dbg !8
  %v = fadd double %u, %a, !dbg !8
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
!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!9}
!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = distinct !DICompileUnit(language: DW_LANG_C11, file: !6, emissionKind: LineTablesOnly)
!5 = distinct !DISubprogram(name: "tester", scope: !6, file: !6, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !4)
!6 = !DIFile(filename: "matrix.c", directory: "")
!7 = !DISubroutineType(types: !{})
!8 = !DILocation(line: 9, column: 7, scope: !5)
!9 = !{i32 2, !"Debug Info Version", i32 3}
