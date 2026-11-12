; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/exact_dp/reprice/preprocess_matmul_body.fpprofile %t/prof/
; RUN: cp %S/../Inputs/exact_dp/reprice/preprocess_matmul_body.fpprofile %t/prof/preprocess_matmul_body_b.fpprofile
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-joint-dp -poseidon-cost-budget=-0.61 -poseidon-print -poseidon-explain=%t/ex_prune -poseidon-cache=%t/prune -S -o %t/prune.ll > %t/prune.log 2>&1
; RUN: FileCheck --check-prefix=PRUNE %s < %t/prune.log
; RUN: FileCheck --check-prefix=PRUNE-BUDGETS %s < %t/prune/budgets.txt
; RUN: cat %t/ex_prune/preprocess_matmul_body.explain.json | FileCheck --check-prefix=PRUNE-EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-joint-dp -poseidon-early-prune=false -poseidon-cost-budget=-0.952 -poseidon-print -poseidon-explain=%t/ex_exact -poseidon-cache=%t/exact -S -o %t/exact.ll > %t/exact.log 2>&1
; RUN: FileCheck --check-prefix=EXACT %s < %t/exact.log
; RUN: FileCheck --check-prefix=EXACT-BUDGETS %s < %t/exact/budgets.txt
; RUN: cat %t/ex_exact/preprocess_matmul_body_b.explain.json | FileCheck --check-prefix=EXACT-EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-joint-dp -poseidon-cache=%t/pins -poseidon-explain=%t/ex_pins -poseidon-apply-rewrites='matmul_body:R0_1,matmul_body_b:M0_2,matmul_body:M0_0,matmul_body_b:PT0_3' -S -o %t/pins.ll > %t/pins.log 2>&1
; RUN: FileCheck --check-prefix=PINS %s < %t/pins.log

; A joint solve over two sites that each hold algebraic, precision and product
; units: the body of exact_dp_reprice_nvptx.ll twice, each with its profile.
; The pruned solve takes every site's algebraic units, then every site's
; products, then every site's precision units; the exact solve takes each
; site's precision units after the algebraic units they are priced against,
; and every product last. A budget at the cheapest point of each frontier
; selects a raise in both sites; pins can name units of both sites and every
; kind. The poseidon pass runs alone, as in dp_cache_replay_all_kinds_nvptx.ll.

; PRUNE: Poseidon JOINT: collecting candidates from 2 marked function(s)
; PRUNE: JOINT DP table contains 34 entries; cost range [-6.112571e-01, 1.454545e-01]
; PRUNE: JOINT: minimum accuracy cost within budget: 1.118681e+10; computation cost used: -6.112571e-01
; PRUNE: Applying solution for (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)->
; PRUNE-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; PRUNE: [poseidon] pin: matmul_body:R0=
; PRUNE: Applying solution for (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)->
; PRUNE-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; PRUNE: [poseidon] pin: matmul_body_b:R0=
; PRUNE-BUDGETS: {{^}}-0.61125710227272723,-0.60898437500000002,-0.60671164772727271,-0.6044389204545455,-0.60216619318181819,-0.5233842329545455,-0.51544744318181823,-0.50978338068181817,-0.41834161931818181,-0.41040482954545454,-0.40474076704545453,-0.36979758522727274,-0.36573153409090908,-0.35779474431818181,-0.3521306818181818,-0.34985795454545454,-0.34419389204545453,-0.33852982954545452,-0.30790127840909093,-0.27295809659090908,-0.23801491477272727,-0.20307173295454545,-0.16926491477272726,-0.10786576704545454,-0.10220170454545455,-0.09653764204545455,-0.065909090909090903,-0.030965909090909093,0.0039772727272727269,0.038920454545454543,0.072727272727272724,0.13412642045454545,0.13979048295454546,0.14545454545454545{{$}}
; PRUNE-EXPLAIN: "result": {
; PRUNE-EXPLAIN-NEXT: "applied": [
; PRUNE-EXPLAIN-NEXT: "R0_1",
; PRUNE-EXPLAIN-NEXT: "M0_0"
; PRUNE-EXPLAIN-NEXT: ]

; EXACT: Exact DP: 69 frontier points; peak 146 entries in 4 signatures; 234 precision changes priced;
; EXACT: JOINT DP table contains 69 entries; cost range [-9.537642e-01, 1.454545e-01]
; EXACT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; EXACT-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; EXACT: [poseidon] pin: matmul_body:PT0=All FP64(0%) + FP32(100%),matmul_body:M0=wmma m16n16k16 f16/f32
; EXACT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; EXACT-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; EXACT: [poseidon] pin: matmul_body_b:PT0=All FP64(0%) + FP32(100%),matmul_body_b:M0=wmma m16n16k16 f16/f32
; EXACT-BUDGETS: {{^}}-0.95376420454545452,-0.95149147727272732,-0.94921875,-0.94694602272727268,-0.94467329545454548,-0.83963068181818179,-0.82872869318181819,-0.82645596590909087,-0.8185191761363636,-0.81285511363636365,-0.72141335227272729,-0.71347656250000002,-0.70823863636363638,-0.70030184659090911,-0.69463778409090904,-0.69236505681818183,-0.68670099431818177,-0.68103693181818181,-0.60319602272727268,-0.59525923295454541,-0.58959517045454546,-0.58732244318181814,-0.58165838068181819,-0.57599431818181823,-0.54105113636363633,-0.53698508522727273,-0.52904829545454546,-0.5233842329545455,-0.52111150568181819,-0.51544744318181823,-0.50978338068181817,-0.49815340909090911,-0.49021661931818183,-0.48455255681818182,-0.48227982954545456,-0.47661576704545455,-0.47095170454545454,-0.43600852272727275,-0.43194247159090909,-0.42400568181818182,-0.41834161931818181,-0.41606889204545455,-0.41040482954545454,-0.40474076704545453,-0.37411221590909088,-0.36573153409090908,-0.35779474431818181,-0.3521306818181818,-0.34985795454545454,-0.34419389204545453,-0.33852982954545452,-0.30790127840909093,-0.27727272727272728,-0.24232954545454546,-0.20738636363636365,-0.17244318181818183,-0.13863636363636364,-0.13750000000000001,-0.10786576704545454,-0.10220170454545455,-0.09653764204545455,-0.065909090909090903,-0.030965909090909093,0.0039772727272727269,0.038920454545454543,0.072727272727272724,0.13412642045454545,0.13979048295454546,0.14545454545454545{{$}}
; EXACT-EXPLAIN: "result": {
; EXACT-EXPLAIN-NEXT: "applied": [
; EXACT-EXPLAIN-NEXT: "PT0_0",
; EXACT-EXPLAIN-NEXT: "M0_0"
; EXACT-EXPLAIN-NEXT: ]

; PINS: [poseidon] Selecting matmul_body:R0_1:
; PINS: [poseidon] Selecting matmul_body:M0_0: matmul[0] -> wmma m16n16k16 f16/f32
; PINS: [poseidon] Selecting matmul_body_b:M0_2: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; PINS-NEXT: [poseidon] Selecting matmul_body_b:PT0_3: All FP64(30%) + FP32(70%)
; PINS: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; PINS: [poseidon] pin: matmul_body:R0=
; PINS: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; PINS-NEXT: Applying solution for CS: All FP64(30%) + FP32(70%) (#3)
; PINS: [poseidon] pin: matmul_body_b:M0=tcec n=3 wmma m16n16k16 f16/f32,matmul_body_b:PT0=All FP64(30%) + FP32(70%)



target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %D, ptr noundef readonly captures(none) %U, ptr noundef readonly captures(none) %W, ptr noundef writeonly captures(none) %Y, double noundef %alpha) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  %cell = add nuw nsw i32 %row, %tx
  %celli = zext nneg i32 %cell to i64
  br label %loop

exit:
  %d = fmul double %alpha, %acc.next
  %dp = getelementptr inbounds nuw [8 x i8], ptr %D, i64 %celli
  store double %d, ptr %dp, align 8
  %up = getelementptr inbounds nuw [8 x i8], ptr %U, i64 %celli
  %u = load double, ptr %up, align 8
  %wp = getelementptr inbounds nuw [8 x i8], ptr %W, i64 %celli
  %w = load double, ptr %wp, align 8
  %u1 = fadd double %u, 1.000000e+00
  %su1 = tail call double @llvm.sqrt.f64(double %u1)
  %su = tail call double @llvm.sqrt.f64(double %u)
  %p = fsub double %su1, %su
  %sh = tail call double @__nv_sinh(double %w)
  %ash = tail call double @__nv_asinh(double %w)
  %q1 = fadd double %sh, %ash
  %tn = tail call double @__nv_tan(double %w)
  %q2 = fadd double %q1, %tn
  %as = tail call double @__nv_asin(double %w)
  %q = fadd double %q2, %as
  %y = fadd double %p, %q
  %yp = getelementptr inbounds nuw [8 x i8], ptr %Y, i64 %celli
  store double %y, ptr %yp, align 8
  ret void

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %ai = add nuw nsw i32 %k, %row
  %aii = zext nneg i32 %ai to i64
  %ap = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %aii
  %a = load double, ptr %ap, align 8
  %bk = shl nuw nsw i32 %k, 4
  %bi = add nuw nsw i32 %bk, %tx
  %bii = zext nneg i32 %bi to i64
  %bp = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %bii
  %b = load double, ptr %bp, align 8
  %acc.next = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop, !llvm.loop !0
}

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_S0_S0_S1_S1_d(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4, ptr noundef %5, ptr noundef %6, ptr noundef %7, double noundef %8) #1 {
  %10 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %11 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iS1_iS1_iS2_S2_idEET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %10, ptr noundef %0, i32 noundef %10, ptr noundef %1, i32 noundef %11, ptr noundef %2, ptr noundef %3, i32 noundef %10, ptr noundef %4, i32 noundef %10, ptr noundef %5, i32 noundef %11, ptr noundef %6, ptr noundef %7, i32 noundef %10, double noundef %8) #2
  ret void
}

define dso_local void @matmul_body_b(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %D, ptr noundef readonly captures(none) %U, ptr noundef readonly captures(none) %W, ptr noundef writeonly captures(none) %Y, double noundef %alpha) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  %cell = add nuw nsw i32 %row, %tx
  %celli = zext nneg i32 %cell to i64
  br label %loop

exit:
  %d = fmul double %alpha, %acc.next
  %dp = getelementptr inbounds nuw [8 x i8], ptr %D, i64 %celli
  store double %d, ptr %dp, align 8
  %up = getelementptr inbounds nuw [8 x i8], ptr %U, i64 %celli
  %u = load double, ptr %up, align 8
  %wp = getelementptr inbounds nuw [8 x i8], ptr %W, i64 %celli
  %w = load double, ptr %wp, align 8
  %u1 = fadd double %u, 1.000000e+00
  %su1 = tail call double @llvm.sqrt.f64(double %u1)
  %su = tail call double @llvm.sqrt.f64(double %u)
  %p = fsub double %su1, %su
  %sh = tail call double @__nv_sinh(double %w)
  %ash = tail call double @__nv_asinh(double %w)
  %q1 = fadd double %sh, %ash
  %tn = tail call double @__nv_tan(double %w)
  %q2 = fadd double %q1, %tn
  %as = tail call double @__nv_asin(double %w)
  %q = fadd double %q2, %as
  %y = fadd double %p, %q
  %yp = getelementptr inbounds nuw [8 x i8], ptr %Y, i64 %celli
  store double %y, ptr %yp, align 8
  ret void

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %ai = add nuw nsw i32 %k, %row
  %aii = zext nneg i32 %ai to i64
  %ap = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %aii
  %a = load double, ptr %ap, align 8
  %bk = shl nuw nsw i32 %k, 4
  %bi = add nuw nsw i32 %bk, %tx
  %bii = zext nneg i32 %bi to i64
  %bp = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %bii
  %b = load double, ptr %bp, align 8
  %acc.next = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop, !llvm.loop !0
}

define dso_local ptx_kernel void @_Z15matmul_kernel_bPKdS0_PdS1_S0_S0_S1_S1_d(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4, ptr noundef %5, ptr noundef %6, ptr noundef %7, double noundef %8) #1 {
  %10 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %11 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iS1_iS1_iS2_S2_idEET_PvDpT0_(ptr noundef nonnull @matmul_body_b, i32 noundef %10, ptr noundef %0, i32 noundef %10, ptr noundef %1, i32 noundef %11, ptr noundef %2, ptr noundef %3, i32 noundef %10, ptr noundef %4, i32 noundef %10, ptr noundef %5, i32 noundef %11, ptr noundef %6, ptr noundef %7, i32 noundef %10, double noundef %8) #2
  ret void
}

declare void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iS1_iS1_iS2_S2_idEET_PvDpT0_(ptr, i32, ptr, i32, ptr, i32, ptr, ptr, i32, ptr, i32, ptr, i32, ptr, ptr, i32, double)
declare double @llvm.fmuladd.f64(double, double, double)
declare double @llvm.sqrt.f64(double)
declare double @__nv_sinh(double)
declare double @__nv_asinh(double)
declare double @__nv_tan(double)
declare double @__nv_asin(double)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y()

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "uniform-work-group-size" }

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.unroll.disable"}


