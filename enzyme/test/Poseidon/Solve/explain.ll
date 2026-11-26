; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -disable-output
; RUN: rm -rf %t && split-file %s %t
; RUN: %{opt} < %t/precision.ll -poseidon-profile-use=%S/../Inputs/explain/precision -poseidon-early-prune -poseidon-cache=%t/a -poseidon-tau=1e-6 -pass-remarks=poseidon -pass-remarks-missed=poseidon -pass-remarks-analysis=poseidon -poseidon-explain=%t/a -poseidon-print 2>&1 | FileCheck --check-prefix=FLOAT %s
; RUN: FileCheck --check-prefix=PJSON %s < %t/a/tester.explain.json
; RUN: %{opt} < %t/precision.ll -poseidon-profile-use=%S/../Inputs/explain/precision -poseidon-early-prune -poseidon-cache=%t/b -poseidon-tau=1e-12 -pass-remarks-missed=poseidon -poseidon-print 2>&1 | FileCheck --check-prefix=KEPT %s
; RUN: %python -c "print('tester' + 'x' * 250)" > %t/long.name
; RUN: sed "s/@tester\b/@$(cat %t/long.name)/" %t/precision.ll > %t/long.ll
; RUN: mkdir -p %t/long.prof && cp %S/../Inputs/explain/precision/tester.fpprofile %t/long.prof/$(%python -c "print('tester' + 'x' * 154 + '~851579e7fddbfcd5')").fpprofile
; RUN: %{opt} < %t/long.ll -poseidon-profile-use=%t/long.prof -poseidon-early-prune -poseidon-cache=%t/c -poseidon-tau=1e-6 -poseidon-explain=%t/long -poseidon-print 2>&1 | FileCheck --check-prefix=LONG --implicit-check-not="Error writing" %s
; RUN: ls %t/long | FileCheck --check-prefix=LONGFILES %s
; RUN: cat %t/long/*.explain.json | FileCheck --check-prefix=LONGJSON -DNAME=$(cat %t/long.name) %s

; precision.ll: exp(-0.5 x^2) cos(4 x) on [-2, 2], float at tau 1e-6, nothing
; at 1e-12. A name too long for a file keeps 160 characters and a hash.

; FLOAT: remark: wave.c:4:3: Poseidon site tester: target 1e-06; modelled error as written 7.1e-18, after 1.2e-08; 1 rewrite, cost 1 -> 0.3125 (-69%); 6 Pareto points{{$}}
; FLOAT-NEXT: remark: wave.c:4:41: Poseidon profile of exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0): x[i] in [-2, 2]; 1024 executions, sensitivity 3.9e+02{{$}}
; FLOAT-NEXT: remark: wave.c:4:26: Poseidon runs all 6 operations in float: (double)(expf((float)x[i] * ((float)x[i] * -0.5f)) * cosf((float)x[i] * 4.0f)); modelled error 7.1e-18 -> 1.2e-08, cost 1 -> 0.3125 (-69%) [P0_0]{{$}}
; FLOAT: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; FLOAT: [poseidon] pin: tester:P0=All FP64(0%) + FP32(100%){{$}}

; PJSON: "pareto": [
; PJSON-NEXT: {
; PJSON-NEXT: "applied": [
; PJSON-NEXT: "P0_0"
; PJSON-NEXT: ],
; PJSON-NEXT: "budget": -0.6875,
; PJSON-NEXT: "cost": 0.3125,
; PJSON-NEXT: "modelled_rel_error": 1.24440{{[0-9]*}}e-08,
; PJSON-NEXT: "selected": true
; PJSON: "result": {
; PJSON-NEXT: "applied": [
; PJSON-NEXT: "P0_0"
; PJSON-NEXT: ],
; PJSON-NEXT: "cost_after": 0.3125,
; PJSON-NEXT: "cost_before": 1,
; PJSON: "alternatives": [
; PJSON-NEXT: {
; PJSON-NEXT: "c": "(double)(expf((float)x[i] * ((float)x[i] * -0.5f)) * cosf((float)x[i] * 4.0f))",
; PJSON: "cost_delta": -0.6875,
; PJSON-NEXT: "id": "P0_0",
; PJSON-NEXT: "kind": "precision",
; PJSON-NEXT: "modelled_error": 1.24440{{[0-9]*}}e-08,
; PJSON: "reason": "part of the cheapest solution that meets the target",
; PJSON-NEXT: "status": "applied",
; PJSON-NEXT: "summary": "All FP64(0%) + FP32(100%): '*' (4:26), '*' (4:33), exp (4:12), '*' (4:41), cos (4:43) and '*' (4:56) in float (6 of 6 operations)"
; PJSON: "inputs": [
; PJSON: "name": "x[i]",
; PJSON-NOT: "var"
; PJSON: "ops": [
; PJSON: "opcode": "*",
; PJSON: "opcode": "exp",
; PJSON: "opcode": "cos",
; PJSON: "original": {
; PJSON-NEXT: "c": "exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0)"
; PJSON: "tau": 9.9999999999999995e-07,
; PJSON-NEXT: "tau_from": "flag"

; KEPT: remark: wave.c:4:41: Poseidon left exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0) as written: every alternative's modelled error is above the target 1e-12 (best: P0_5, 1.1e-09){{$}}
; KEPT-NEXT: remark: wave.c:4:26: Poseidon rejected the precision change P0_5 to exp(x[i] * (double)((float)x[i] * -0.5f)) * cos(x[i] * 4.0): modelled error 1.1e-09 is above the target 1e-12 (cost change -0.03125){{$}}
; KEPT: [poseidon] no rewrite applied for preprocess_tester; calling original body {{[^ ]+}} (bit-transparent no-op){{$}}

; LONG: [poseidon] explanation written to {{.*}}/long/tester{{x{154}~}}851579e7fddbfcd5.explain.json{{$}}

; LONGFILES: {{^}}tester{{x{154}~}}851579e7fddbfcd5.explain.json{{$}}
; LONGFILES-NOT: {{.}}

; LONGJSON: "ir": "preprocess_[[NAME]]",
; LONGJSON: "name": "[[NAME]]",

; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/explain %opt < %t/remarks.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/explain/cancellation -poseidon-early-prune -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-threads=1 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/cache -poseidon-tau=1e-6 -pass-remarks=poseidon -pass-remarks-missed=poseidon -pass-remarks-analysis=poseidon -poseidon-explain=%t/explain -disable-output 2>&1 | FileCheck --check-prefix=REMARK %s
; RUN: FileCheck --check-prefix=RJSON %s < %t/explain/tester.explain.json
; RUN: rm -rf %t/cache
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/explain %clang -O0 -x ir -c %t/remarks.ll -o /dev/null %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/explain/cancellation -mllvm -poseidon-early-prune -mllvm -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -mllvm -poseidon-herbie-threads=1 -mllvm -poseidon-cost-model=%cost_models/cm_cpu_native.csv -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-tau=1e-6 -Rpass=poseidon 2>&1 | FileCheck --check-prefix=RPASS %s

; remarks.ll: (1 - cos x) / x^2 on [1e-8, 1e-4], replayed Herbie's series.

; REMARK: cancellation.c:9:3: Poseidon site tester: target 1e-06; modelled error as written {{.*}}
; REMARK: cancellation.c:4:37: Poseidon profile of (1.0 - cos(x[i])) / (x[i] * x[i]): x[i] in [1e-08, 0.0001]; 1024 executions
; REMARK: cancellation.c:4:37: Poseidon rewrote (1.0 - cos(x[i])) / (x[i] * x[i]) as 0.5 (series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]), valid on the profiled range x[i] in [1e-08, 0.0001]); modelled error {{.*}} -> {{.*}}, cost {{[0-9]+}} -> 0 (-100%) [R0_1]
; REMARK: cancellation.c:4:37: Poseidon {{did not select|rejected}} the rewrite R0_0 to fma(-0.041666666666666664, x[i] * x[i], 0.5)

; RPASS-NOT: Poseidon profile
; RPASS: cancellation.c:4:37: Poseidon rewrote (1.0 - cos(x[i])) / (x[i] * x[i]) as 0.5 {{.*}}[-Rpass=poseidon]
; RPASS-NOT: Poseidon rejected

; RJSON: "function": {
; RJSON: "name": "tester"
; RJSON: "symbol": "tester"
; RJSON: "pareto": [
; RJSON: "result": {
; RJSON-NEXT: "applied": [
; RJSON-NEXT: "R0_1"
; RJSON: "cost_after": 0,
; RJSON: "schema": "poseidon-explain/1",
; RJSON: "subgraphs": [
; RJSON: "id": "R0_0",
; RJSON: "approx": true,
; RJSON-NEXT: "approx_note": "series approximation of (1.0 - cos(x[i])) / (x[i] * x[i])",
; RJSON-NEXT: "c": "0.5",
; RJSON: "fpcore": "#s(approx (/ (- 1 (cos v0)) (* v0 v0)) #s(literal 1/2 binary64))",
; RJSON: "id": "R0_1",
; RJSON: "status": "applied",
; RJSON: "c": "(double)0.5f",
; RJSON: "id": "R1_0",
; RJSON: "precision": "binary32",
; RJSON: "kind": "precision",
; RJSON: "id": "S0",
; RJSON: "inputs": [
; RJSON: "name": "x[i]",
; RJSON: "var": "v0"
; RJSON: "original": {
; RJSON-NEXT: "c": "(1.0 - cos(x[i])) / (x[i] * x[i])",
; RJSON-NEXT: "fpcore": "(/ (- 1 (cos v0)) (* v0 v0))"
; RJSON: "target": {
; RJSON: "tau": {{[0-9.e+-]+}},
; RJSON-NEXT: "tau_from": "flag"

; RUN: %opt < %t/left_alone.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-explain=%t/ex -S -o /dev/null 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: FileCheck --check-prefix=AJSON %s < %t/ex/counter.explain.json
; RUN: %poseidon_explain %t/ex | FileCheck --check-prefix=REPORT %s

; left_alone.ll: a site left as written before any solve says why.

; WARN: warning: {{.*}}counter holds no floating-point operation the pass rewrites; leaving the site unchanged

; AJSON: "function": {
; AJSON-NEXT: "ir": "preprocess_counter",
; AJSON-NEXT: "name": "counter",
; AJSON-NEXT: "symbol": "counter"
; AJSON-NEXT: },
; AJSON-NEXT: "result": {
; AJSON-NEXT: "reason": "the site holds no floating-point operation the pass rewrites",
; AJSON-NEXT: "status": "no_fp_operation"
; AJSON-NEXT: },
; AJSON-NEXT: "schema": "poseidon-explain/1",
; AJSON-NEXT: "target": {
; AJSON-NEXT: "tau": 0.5,
; AJSON-NEXT: "tau_from": "site"
; AJSON-NEXT: }

; REPORT: Poseidon site counter  [counter]
; REPORT-NEXT: accuracy target tau = 0.5 (the site's own); no fp operation: the site holds no floating-point operation the pass rewrites

;--- precision.ll
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

;--- remarks.ll
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

;--- left_alone.ll
define i64 @counter(i64 %n) {
entry:
  %r = add i64 %n, 1
  ret i64 %r
}

define i64 @site(i64 %n) {
entry:
  %0 = tail call i64 (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @counter, i64 %n, metadata !"poseidon_tau", double 0.5)
  ret i64 %0
}

declare i64 @__poseidon_fp_optimize(ptr, ...)
