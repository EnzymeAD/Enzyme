; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/unexecuted_operations \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-tau=1e-4 -pass-remarks=poseidon -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/loose -o %t/loose.ll 2>&1 | FileCheck --check-prefix=LOOSE %s
; RUN: FileCheck --check-prefix=IR %s < %t/loose.ll

; The profiling run took only the scalar branch, so the two operations of the
; matrix branch have no profile record. They are left as written, as an
; unexecuted matrix product is, with a note naming their source line, and the
; subgraph of the executed operations around them ends at them: priced from the
; profile alone, it is rewritten.
;
; Profile: this module through the host profiler, driven by
;   for (int i = 0; i < 64; ++i) {
;     double x = 1.0 + i * 0.125, y = 0.5 + (i % 7) * 0.25, o, d = 1.0;
;     site(x, y, 0, 0, &o, &d);
;   }

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

; LOOSE: [poseidon] preprocess_tester: 2 floating-point operation(s) the profiling run never executed are left as written (matrix.c:9): no profile gives their ranges, cost or error, so the subgraphs of the executed operations end at them and the error a rewrite of those passes on to them is not modelled
; LOOSE: [poseidon] preprocess_tester: tau=1.000000e-04 S=3.818938e+03 A0=2.850360e-11 selected cost=-4.375000e-01 accCost=2.794345e-02 rel=7.317073e-06
; LOOSE-NEXT: remark: <unknown>:0:0: Poseidon runs all 4 operations in float; modelled error 7.5e-15 -> 7.3e-06, cost 1 -> 0.5625 (-44%) [PT0_0]
; LOOSE: [poseidon] pin: tester:PT0=All FP64(0%) + FP32(100%)

; IR-LABEL: define void @preprocess_tester(
; IR-NEXT: entry:
; IR-NEXT: [[XF:%.+]] = fptrunc double %x to float
; IR-NEXT: [[YF:%.+]] = fptrunc double %y to float
; IR-NEXT: [[A:%.+]] = fmul float [[XF]], [[YF]]{{$}}
; IR-NEXT: [[B:%.+]] = fadd float [[A]], [[XF]]{{$}}
; IR: [[R:%.+]] = phi double [ [[ZD:%.+]], %scal ], [ %v, %mat ]
; IR-NEXT: store double [[R]], ptr %o
; IR: mat:
; IR-NEXT: [[BD:%.+]] = fpext float [[B]] to double
; IR-NEXT: [[AD:%.+]] = fpext float [[A]] to double
; IR-NEXT: %mv = load double, ptr %m
; IR-NEXT: %u = fmul double %mv, [[BD]],
; IR-NEXT: %v = fadd double %u, [[AD]],
; IR: scal:
; IR-NEXT: [[W:%.+]] = fmul float [[B]], [[YF]]{{$}}
; IR-NEXT: [[Z:%.+]] = fadd float [[W]], [[XF]]{{$}}
; IR-NEXT: [[ZD]] = fpext float [[Z]] to double

