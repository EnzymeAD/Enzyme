; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/explain/cancellation -poseidon-herbie-binary=%S/Inputs/explain/fake-herbie.sh -poseidon-herbie-num-threads=1 -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/cache -poseidon-tau=1e-6 -pass-remarks=poseidon -pass-remarks-missed=poseidon -pass-remarks-analysis=poseidon -poseidon-explain=%t/explain -disable-output 2>&1 | FileCheck --check-prefix=REMARK %s
; RUN: FileCheck --check-prefix=JSON %s < %t/explain/preprocess_tester.explain.json
; RUN: rm -rf %t/cache
; RUN: %clang -O0 -x ir -c %s -o /dev/null %clangLoadPoseidon -mllvm -poseidon-profile-use=%S/Inputs/explain/cancellation -mllvm -poseidon-herbie-binary=%S/Inputs/explain/fake-herbie.sh -mllvm -poseidon-herbie-num-threads=1 -mllvm -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-tau=1e-6 -Rpass=poseidon 2>&1 | FileCheck --check-prefix=RPASS %s
; REQUIRES: poseidon

; (1 - cos x) / x^2 cancels for small x; the profiled range is [1e-8, 1e-4]
; and Herbie (replayed by fake-herbie.sh) answers with its series.

; REMARK: cancellation.c:9:3: Poseidon site tester: target 1e-06; modelled error as written {{.*}}
; REMARK: cancellation.c:4:37: Poseidon profile of (1.0 - cos(x[i])) / (x[i] * x[i]): x[i] in [1e-08, 0.0001]; 1024 executions
; REMARK: cancellation.c:4:37: Poseidon rewrote (1.0 - cos(x[i])) / (x[i] * x[i]) as 0.5 (series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]), valid on the profiled range x[i] in [1e-08, 0.0001]); modelled error {{.*}} -> {{.*}}, cost {{[0-9]+}} -> 0 (-100%) [R0_1]
; REMARK: cancellation.c:4:37: Poseidon {{did not select|rejected}} the rewrite R0_0 to fma(-0.041666666666666664, x[i] * x[i], 0.5)

; RPASS-NOT: Poseidon profile
; RPASS: cancellation.c:4:37: Poseidon rewrote (1.0 - cos(x[i])) / (x[i] * x[i]) as 0.5 {{.*}}[-Rpass=poseidon]
; RPASS-NOT: Poseidon rejected

; JSON: "function": {
; JSON: "name": "tester"
; JSON: "symbol": "tester"
; JSON: "pareto": [
; JSON: "result": {
; JSON-NEXT: "applied": [
; JSON-NEXT: "R0_1"
; JSON: "cost_after": 0,
; JSON: "schema": "poseidon-explain/1",
; JSON: "subgraphs": [
; JSON: "id": "R0_0",
; JSON: "approx": true,
; JSON-NEXT: "approx_note": "series approximation of (1.0 - cos(x[i])) / (x[i] * x[i])",
; JSON-NEXT: "c": "0.5",
; JSON: "fpcore": "#s(approx (/ (- 1 (cos v0)) (* v0 v0)) #s(literal 1/2 binary64))",
; JSON: "id": "R0_1",
; JSON: "status": "applied",
; JSON: "c": "(double)0.5f",
; JSON: "id": "R1_0",
; JSON: "precision": "binary32",
; JSON: "kind": "precision",
; JSON: "id": "S0",
; JSON: "inputs": [
; JSON: "name": "x[i]",
; JSON: "var": "v0"
; JSON: "original": {
; JSON-NEXT: "c": "(1.0 - cos(x[i])) / (x[i] * x[i])",
; JSON-NEXT: "fpcore": "(/ (- 1 (cos v0)) (* v0 v0))"
; JSON: "target": {
; JSON: "tau": {{[0-9.e+-]+}},
; JSON-NEXT: "tau_from": "flag"

target triple = "x86_64-unknown-linux-gnu"

declare double @cos(double)

define void @tester(ptr %x, ptr %out, i32 %n) #0 !dbg !5 {
entry:
    #dbg_value(ptr %x, !11, !DIExpression(), !15)
    #dbg_value(ptr %out, !12, !DIExpression(), !15)
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
  %c = tail call double @cos(double %v), !dbg !18
  %sub = fsub double 1.000000e+00, %c, !dbg !19
  %mul = fmul double %v, %v, !dbg !20
  %div = fdiv double %sub, %mul, !dbg !21
  %po = getelementptr inbounds double, ptr %out, i64 %iv, !dbg !22
  store double %div, ptr %po, align 8, !dbg !22
  %iv.next = add nuw nsw i64 %iv, 1, !dbg !16
  %done = icmp eq i64 %iv.next, %count, !dbg !16
  br i1 %done, label %exit, label %loop, !dbg !16

exit:
  ret void, !dbg !23
}

define void @test_opt(ptr %x, ptr %dx, ptr %out, ptr %dout, i32 %n) !dbg !24 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %out, ptr %dout, i32 %n), !dbg !25
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "cancellation.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "tester", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !9}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12, !13}
!11 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 2, type: !8)
!12 = !DILocalVariable(name: "out", arg: 2, scope: !5, file: !1, line: 2, type: !8)
!13 = !DILocalVariable(name: "i", scope: !5, file: !1, line: 3, type: !9)
!14 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!15 = !DILocation(line: 0, scope: !5)
!16 = !DILocation(line: 3, column: 3, scope: !5)
!17 = !DILocation(line: 4, column: 30, scope: !5)
!18 = !DILocation(line: 4, column: 21, scope: !5)
!19 = !DILocation(line: 4, column: 19, scope: !5)
!20 = !DILocation(line: 4, column: 45, scope: !5)
!21 = !DILocation(line: 4, column: 37, scope: !5)
!22 = !DILocation(line: 4, column: 12, scope: !5)
!23 = !DILocation(line: 5, column: 1, scope: !5)
!24 = distinct !DISubprogram(name: "test_opt", scope: !1, file: !1, line: 8, type: !6, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!25 = !DILocation(line: 9, column: 3, scope: !24)
