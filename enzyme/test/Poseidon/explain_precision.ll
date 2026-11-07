; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/explain/precision -poseidon-enable-herbie=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/a -poseidon-tau=1e-6 -pass-remarks=poseidon -pass-remarks-missed=poseidon -pass-remarks-analysis=poseidon -poseidon-explain=%t/a -disable-output 2>&1 | FileCheck --check-prefix=FLOAT %s
; RUN: FileCheck --check-prefix=JSON %s < %t/a/preprocess_tester.explain.json
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/explain/precision -poseidon-enable-herbie=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/b -poseidon-tau=1e-12 -pass-remarks-missed=poseidon -disable-output 2>&1 | FileCheck --check-prefix=KEPT %s
; REQUIRES: poseidon

; exp(-0.5 x^2) cos(4 x) on x in [-2, 2]. At tau 1e-6 single precision is
; accurate enough for both library calls; at 1e-12 nothing is.

; FLOAT: wave.c:4:3: Poseidon site tester: target 1e-06
; FLOAT: wave.c:4:41: Poseidon profile of exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0): x[i] in [-2, 2]; 1024 executions
; FLOAT: remark: wave.c:{{[0-9]+}}:{{[0-9]+}}: Poseidon runs {{.*}} in float: {{.*}}; modelled error {{.*}} -> {{.*}}, cost {{[0-9]+}} -> {{[0-9]+}} {{.*}}[PT0_{{[0-9]+}}]

; JSON: "kind": "precision",
; JSON: "status": "applied",
; JSON: "inputs": [
; JSON: "name": "x[i]",
; JSON-NOT: "var"
; JSON: "ops": [
; JSON: "opcode": "*",
; JSON: "opcode": "exp",
; JSON: "original": {
; JSON-NEXT: "c": "exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0)"
; JSON: "tau": {{[0-9.e+-]+}},

; KEPT: wave.c:4:41: Poseidon left exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0) as written: every alternative's modelled error is above the target 1e-12 (best: PT0_{{[0-9]+}}, {{.*}})
; KEPT: Poseidon rejected the precision change PT0_{{[0-9]+}} to {{.*}}: modelled error {{.*}} is above the target 1e-12

target triple = "x86_64-unknown-linux-gnu"

declare double @exp(double)
declare double @cos(double)

define void @tester(ptr %x, ptr %y, i32 %n) #0 !dbg !5 {
entry:
    #dbg_value(ptr %x, !11, !DIExpression(), !15)
  %cmp = icmp sgt i32 %n, 0, !dbg !16
  br i1 %cmp, label %ph, label %exit, !dbg !16

ph:
  %count = zext i32 %n to i64, !dbg !16
  br label %loop, !dbg !16

loop:
  %iv = phi i64 [ 0, %ph ], [ %iv.next, %loop ]
    #dbg_value(i64 %iv, !13, !DIExpression(), !15)
  %px = getelementptr inbounds double, ptr %x, i64 %iv, !dbg !17
  %v = load double, ptr %px, align 8, !dbg !17
  %h = fmul double %v, -5.000000e-01, !dbg !18
  %q = fmul double %v, %h, !dbg !19
  %e = tail call double @exp(double %q), !dbg !20
  %f = fmul double %v, 4.000000e+00, !dbg !21
  %c = tail call double @cos(double %f), !dbg !22
  %r = fmul double %e, %c, !dbg !23
  %py = getelementptr inbounds double, ptr %y, i64 %iv, !dbg !24
  store double %r, ptr %py, align 8, !dbg !24
  %iv.next = add nuw nsw i64 %iv, 1, !dbg !16
  %done = icmp eq i64 %iv.next, %count, !dbg !16
  br i1 %done, label %exit, label %loop, !dbg !16

exit:
  ret void, !dbg !16
}

define void @test_opt(ptr %x, ptr %dx, ptr %y, ptr %dy, i32 %n) !dbg !25 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %y, ptr %dy, i32 %n), !dbg !26
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "wave.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "tester", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !9}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !13}
!11 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 2, type: !8)
!13 = !DILocalVariable(name: "i", scope: !5, file: !1, line: 3, type: !9)
!14 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!15 = !DILocation(line: 0, scope: !5)
!16 = !DILocation(line: 3, column: 3, scope: !5)
!17 = !DILocation(line: 4, column: 28, scope: !5)
!18 = !DILocation(line: 4, column: 26, scope: !5)
!19 = !DILocation(line: 4, column: 33, scope: !5)
!20 = !DILocation(line: 4, column: 12, scope: !5)
!21 = !DILocation(line: 4, column: 56, scope: !5)
!22 = !DILocation(line: 4, column: 43, scope: !5)
!23 = !DILocation(line: 4, column: 41, scope: !5)
!24 = !DILocation(line: 4, column: 5, scope: !5)
!25 = distinct !DISubprogram(name: "test_opt", scope: !1, file: !1, line: 7, type: !6, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!26 = !DILocation(line: 4, column: 3, scope: !25)
