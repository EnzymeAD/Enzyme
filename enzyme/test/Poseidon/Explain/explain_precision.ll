; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -disable-output
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} < %s -poseidon-profile-use=%S/../Inputs/explain/precision -poseidon-cache=%t/a -poseidon-tau=1e-6 -pass-remarks=poseidon -pass-remarks-missed=poseidon -pass-remarks-analysis=poseidon -poseidon-explain=%t/a 2>&1 \
; RUN:   | FileCheck --check-prefix=FLOAT %s
; RUN: FileCheck --check-prefix=JSON %s < %t/a/preprocess_tester.explain.json
; RUN: %{opt} < %s -poseidon-profile-use=%S/../Inputs/explain/precision -poseidon-cache=%t/b -poseidon-tau=1e-12 -pass-remarks-missed=poseidon 2>&1 \
; RUN:   | FileCheck --check-prefix=KEPT %s
; RUN: %python -c "print('tester' + 'x' * 250)" > %t/long.name
; RUN: sed "s/@tester\b/@$(cat %t/long.name)/" %s > %t/long.ll
; RUN: mkdir -p %t/long.prof && cp %S/../Inputs/explain/precision/preprocess_tester.fpprofile %t/long.prof/$(%python -c "print('preprocess_tester' + 'x' * 143 + '~940caf41237f8c7f')").fpprofile
; RUN: %{opt} < %t/long.ll -poseidon-profile-use=%t/long.prof -poseidon-cache=%t/c -poseidon-tau=1e-6 -poseidon-explain=%t/long -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=LONG --implicit-check-not="Error writing" %s
; RUN: ls %t/long | FileCheck --check-prefix=LONGFILES %s
; RUN: cat %t/long/*.explain.json | FileCheck --check-prefix=LONGJSON -DNAME=$(cat %t/long.name) %s

; exp(-0.5 x^2) cos(4 x) on x in [-2, 2]. At tau 1e-6 single precision is
; accurate enough for both library calls; at 1e-12 nothing is.
; The site's files are named as its profile is: a name longer than a file
; name may be keeps its first 160 characters and a hash of the whole, and the
; files hold the whole name.

; FLOAT: remark: wave.c:4:3: Poseidon site tester: target 1e-06; modelled error as written 7.1e-18, after 1.2e-08; 1 rewrite, cost 1 -> 0.3125 (-69%); 6 Pareto points{{$}}
; FLOAT-NEXT: remark: wave.c:4:41: Poseidon profile of exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0): x[i] in [-2, 2]; 1024 executions, sensitivity 3.9e+02{{$}}
; FLOAT-NEXT: remark: wave.c:4:26: Poseidon runs all 6 operations in float: (double)(expf((float)x[i] * ((float)x[i] * -0.5f)) * cosf((float)x[i] * 4.0f)); modelled error 7.1e-18 -> 1.2e-08, cost 1 -> 0.3125 (-69%) [PT0_0]{{$}}
; FLOAT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; FLOAT: [poseidon] pin: tester:PT0=All FP64(0%) + FP32(100%){{$}}

; JSON: "pareto": [
; JSON-NEXT: {
; JSON-NEXT: "applied": [
; JSON-NEXT: "PT0_0"
; JSON-NEXT: ],
; JSON-NEXT: "budget": -0.6875,
; JSON-NEXT: "cost": 0.3125,
; JSON-NEXT: "modelled_rel_error": 1.24440{{[0-9]*}}e-08,
; JSON-NEXT: "selected": true
; JSON: "result": {
; JSON-NEXT: "applied": [
; JSON-NEXT: "PT0_0"
; JSON-NEXT: ],
; JSON-NEXT: "cost_after": 0.3125,
; JSON-NEXT: "cost_before": 1,
; JSON: "alternatives": [
; JSON-NEXT: {
; JSON-NEXT: "c": "(double)(expf((float)x[i] * ((float)x[i] * -0.5f)) * cosf((float)x[i] * 4.0f))",
; JSON: "cost_delta": -0.6875,
; JSON-NEXT: "id": "PT0_0",
; JSON-NEXT: "kind": "precision",
; JSON-NEXT: "modelled_error": 1.24440{{[0-9]*}}e-08,
; JSON: "reason": "part of the cheapest solution that meets the target",
; JSON-NEXT: "status": "applied",
; JSON-NEXT: "summary": "All FP64(0%) + FP32(100%): '*' (4:26), '*' (4:33), exp (4:12), '*' (4:41), cos (4:43) and '*' (4:56) in float (6 of 6 operations)"
; JSON: "inputs": [
; JSON: "name": "x[i]",
; JSON-NOT: "var"
; JSON: "ops": [
; JSON: "opcode": "*",
; JSON: "opcode": "exp",
; JSON: "opcode": "cos",
; JSON: "original": {
; JSON-NEXT: "c": "exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0)"
; JSON: "tau": 9.9999999999999995e-07,
; JSON-NEXT: "tau_from": "flag"

; KEPT: remark: wave.c:4:41: Poseidon left exp(x[i] * (x[i] * -0.5)) * cos(x[i] * 4.0) as written: every alternative's modelled error is above the target 1e-12 (best: PT0_5, 1.1e-09){{$}}
; KEPT-NEXT: remark: wave.c:4:26: Poseidon rejected the precision change PT0_5 to exp(x[i] * (double)((float)x[i] * -0.5f)) * cos(x[i] * 4.0): modelled error 1.1e-09 is above the target 1e-12 (cost change -0.03125){{$}}
; KEPT: warning: {{.*}}Poseidon returned false (no change) for preprocess_tester{{$}}

; LONG: Poseidon explanation written to {{.*}}/long/preprocess_tester{{x{143}~}}940caf41237f8c7f.explain.json{{$}}

; LONGFILES: {{^}}preprocess_tester{{x{143}~}}940caf41237f8c7f.explain.json{{$}}
; LONGFILES-NOT: {{.}}

; LONGJSON: "ir": "preprocess_[[NAME]]",
; LONGJSON: "name": "[[NAME]]",

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
