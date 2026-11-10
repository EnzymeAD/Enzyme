; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/Inputs/exact_dp/reprice/preprocess_matmul_body.fpprofile %t/prof/
; RUN: cp %S/Inputs/exact_dp/reprice/preprocess_matmul_body.fpprofile %t/prof/preprocess_matmul_body_b.fpprofile
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-joint-dp -poseidon-cost-budget=-0.622 -poseidon-print -poseidon-explain=%t/ex_prune -poseidon-cache=%t/prune -S -o %t/prune.ll > %t/prune.log 2>&1
; RUN: FileCheck --check-prefix=PRUNE %s < %t/prune.log
; RUN: FileCheck --check-prefix=PRUNE-BUDGETS %s < %t/prune/budgets.txt
; RUN: cat %t/ex_prune/preprocess_matmul_body.explain.json | FileCheck --check-prefix=PRUNE-EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-joint-dp -poseidon-early-prune=false -poseidon-cost-budget=-0.95 -poseidon-print -poseidon-explain=%t/ex_exact -poseidon-cache=%t/exact -S -o %t/exact.ll > %t/exact.log 2>&1
; RUN: FileCheck --check-prefix=EXACT %s < %t/exact.log
; RUN: FileCheck --check-prefix=EXACT-BUDGETS %s < %t/exact/budgets.txt
; RUN: cat %t/ex_exact/preprocess_matmul_body_b.explain.json | FileCheck --check-prefix=EXACT-EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%t/prof -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-joint-dp -poseidon-cache=%t/pins -poseidon-explain=%t/ex_pins -poseidon-apply-rewrites='matmul_body:R0_1,matmul_body_b:PT1_0,matmul_body_b:M0_2,matmul_body:M0_0,matmul_body_b:PT0_3' -S -o %t/pins.ll > %t/pins.log 2>&1
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
; PRUNE: JOINT DP table contains 34 entries; cost range [-6.229150e-01, 1.324742e-01]
; PRUNE: JOINT: minimum accuracy cost within budget: 1.118681e+10; computation cost used: -6.229150e-01
; PRUNE: Applying solution for (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)->
; PRUNE-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; PRUNE: [poseidon] pin: matmul_body:R0=
; PRUNE: Applying solution for (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)->
; PRUNE-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; PRUNE: [poseidon] pin: matmul_body_b:R0=
; PRUNE-BUDGETS: {{^}}-0.62291502318119141,-0.62066720595562619,-0.61852030536151992,-0.61841938873006086,-0.61627248813595459,-0.61412558754184832,-0.53064553878385146,-0.52265698042721653,-0.51662676340239844,-0.42493239694342561,-0.41694383858679074,-0.41091362156197259,-0.38103034251851825,-0.37304178416188338,-0.36701156713706523,-0.36505322580524852,-0.35902300878043036,-0.35299279175561221,-0.3218911113486656,-0.2785005424621721,-0.2363507939481598,-0.20416787773666467,-0.17649639587780611,-0.12231974926324135,-0.1162895322384232,-0.11025931521360506,-0.079157634806658467,-0.035767065920164973,0.0063826825938473261,0.038565598805342466,0.066237080664201045,0.12041372727876579,0.12644394430358394,0.13247416132840209{{$}}
; PRUNE-EXPLAIN: "result": {
; PRUNE-EXPLAIN-NEXT: "applied": [
; PRUNE-EXPLAIN-NEXT: "R0_1",
; PRUNE-EXPLAIN-NEXT: "M0_0"
; PRUNE-EXPLAIN-NEXT: ]

; EXACT: Exact DP: 66 frontier points; peak 144 entries in 4 signatures; 234 precision changes priced;
; EXACT: JOINT DP table contains 66 entries; cost range [-9.501830e-01, 1.324742e-01]
; EXACT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; EXACT-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; EXACT: [poseidon] pin: matmul_body:PT0=All FP64(0%) + FP32(100%),matmul_body:M0=wmma m16n16k16 f16/f32
; EXACT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; EXACT-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; EXACT: [poseidon] pin: matmul_body_b:PT0=All FP64(0%) + FP32(100%),matmul_body_b:M0=wmma m16n16k16 f16/f32
; EXACT-BUDGETS: {{^}}-0.95018296647476386,-0.94793514924919853,-0.94578824865509226,-0.94568733202363331,-0.94354043142952704,-0.94139353083542066,-0.83568038899499486,-0.82699280891786198,-0.82484590832375571,-0.81685734996712078,-0.81082713294230269,-0.7191327664833298,-0.71114420812669499,-0.70829828581209064,-0.70030972745545572,-0.69427951043063763,-0.69232116909882091,-0.6862909520740027,-0.68026073504918461,-0.60258514397166485,-0.59459658561502993,-0.58856636859021183,-0.586608027258395,-0.58057781023357691,-0.57454759320875881,-0.54466431416530448,-0.53667575580866955,-0.53064553878385146,-0.52868719745203463,-0.52265698042721653,-0.51662676340239844,-0.496872002131239,-0.48888344377460413,-0.48285322674978598,-0.48089488541796926,-0.47486466839315111,-0.46883445136833296,-0.43895117232487862,-0.43096261396824376,-0.42493239694342561,-0.42297405561160889,-0.41694383858679074,-0.41091362156197259,-0.38103034251851825,-0.37304178416188338,-0.36701156713706523,-0.36505322580524852,-0.35902300878043036,-0.35299279175561221,-0.3218911113486656,-0.29078943094171905,-0.24739886205522554,-0.20524911354121322,-0.17306619732971809,-0.14539471547085953,-0.12231974926324135,-0.1162895322384232,-0.11025931521360506,-0.079157634806658467,-0.035767065920164973,0.0063826825938473261,0.038565598805342466,0.066237080664201045,0.12041372727876579,0.12644394430358394,0.13247416132840209{{$}}
; EXACT-EXPLAIN: "result": {
; EXACT-EXPLAIN-NEXT: "applied": [
; EXACT-EXPLAIN-NEXT: "PT0_0",
; EXACT-EXPLAIN-NEXT: "M0_0"
; EXACT-EXPLAIN-NEXT: ]

; PINS: [poseidon] Selecting matmul_body:R0_1:
; PINS: [poseidon] Selecting matmul_body:M0_0: matmul[0] -> wmma m16n16k16 f16/f32
; PINS: [poseidon] Selecting matmul_body_b:PT1_0: All FP64(0%) + FP32(100%)
; PINS-NEXT: [poseidon] Selecting matmul_body_b:M0_2: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; PINS-NEXT: [poseidon] Selecting matmul_body_b:PT0_3: All FP64(30%) + FP32(70%)
; PINS: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; PINS: [poseidon] pin: matmul_body:R0=
; PINS: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; PINS-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; PINS-NEXT: Applying solution for CS: All FP64(30%) + FP32(70%) (#3)
; PINS: [poseidon] pin: matmul_body_b:PT1=All FP64(0%) + FP32(100%),matmul_body_b:M0=tcec n=3 wmma m16n16k16 f16/f32,matmul_body_b:PT0=All FP64(30%) + FP32(70%)



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


