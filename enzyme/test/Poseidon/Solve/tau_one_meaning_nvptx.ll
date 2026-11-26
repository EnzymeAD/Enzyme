; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/tau_one_meaning -poseidon-early-prune -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-enable-precision=true -poseidon-enable-matmul -poseidon-matmul-library-calls=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -S -poseidon-print
; DEFINE: %{err} = %python -c "import json, sys; d = json.load(open(sys.argv[1])); m = [a for u in d['subgraphs'] for a in u['alternatives'] if a['kind'] == 'matmul' and a['status'] == 'applied']; print('site %.6e product %.6e' % (d['result']['modelled_rel_error'], m[0]['modelled_domain_error']))"
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/ptr noundef %dd) #5/ptr noundef %dd, metadata !"poseidon_tau", double 1.000000e-04) #5/' %s > %t/site.ll
; RUN: sed 's/ptr noundef %dd) #5/ptr noundef %dd, metadata !"poseidon_confidence", double 5.000000e-01) #5/' %s > %t/key.ll
; RUN: %{opt} %t/site.ll -poseidon-cache=%t/c1 -poseidon-explain=%t/x1 -o %t/site.out.ll 2> %t/site.err
; RUN: %{opt} %s -poseidon-cache=%t/c2 -poseidon-explain=%t/x2 -poseidon-tau=1e-4 -o %t/flag.out.ll 2> %t/flag.err
; RUN: %{opt} %t/site.ll -poseidon-cache=%t/c3 -poseidon-tau=1e-6 -o /dev/null 2> %t/both.err
; RUN: FileCheck --check-prefix=SOLVE %s < %t/site.err
; RUN: FileCheck --check-prefix=SOLVE --check-prefix=FLAG %s < %t/flag.err
; RUN: FileCheck --check-prefix=BOTH %s < %t/both.err
; RUN: sed -e 's/, metadata !"poseidon_tau", double 1.000000e-04//' -e '/^; ModuleID/d' -e '/^source_filename/d' %t/site.out.ll > %t/site.cmp.ll
; RUN: sed -e '/^; ModuleID/d' -e '/^source_filename/d' %t/flag.out.ll > %t/flag.cmp.ll
; RUN: diff %t/site.cmp.ll %t/flag.cmp.ll
; RUN: cat %t/x1/*.explain.json | FileCheck --check-prefix=FROMSITE %s
; RUN: cat %t/x2/*.explain.json | FileCheck --check-prefix=FROMFLAG %s
; RUN: not %opt %t/key.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/tau_one_meaning -poseidon-early-prune -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/c4 -S -o /dev/null 2>&1 | FileCheck --check-prefix=KEY %s
; RUN: %{opt} %s -o /dev/null -poseidon-tau=1e-4 -poseidon-matmul-sample-log-bits=0 -poseidon-cache=%t/c5 -poseidon-explain=%t/x5 2>&1 | FileCheck --check-prefix=SUM %s
; RUN: %{opt} %s -o /dev/null -poseidon-tau=1e-4 -poseidon-matmul-sample-log-bits=0 -poseidon-solve-jointly -poseidon-cache=%t/c6 -poseidon-explain=%t/x6 2>&1 | FileCheck --check-prefix=SUM %s
; RUN: %{err} %t/x5/matmul_body.explain.json | FileCheck --check-prefix=SUM-EXPLAIN %s
; RUN: %{err} %t/x6/matmul_body.explain.json | FileCheck --check-prefix=SUM-EXPLAIN %s
; RUN: %{opt} %s -o /dev/null -poseidon-tau=2.909958e-4 -poseidon-cache=%t/c7 -poseidon-explain=%t/x7 2>&1 | FileCheck --check-prefix=SHARE %s
; RUN: %{err} %t/x7/matmul_body.explain.json | FileCheck --check-prefix=SHARE-EXPLAIN %s

; A site with a matrix product and elementwise work: the accuracy target at the
; site (poseidon_tau, site.ll) and on the command line select and emit the
; same; only a target the site wrote is reported as the site's own.

; FLAG-NOT: overrides -poseidon-tau
; SOLVE: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32
; SOLVE: Applying solution for precision change All FP64(0%) + FP32(100%)
; BOTH: the site's own accuracy target 1.000000e-04 overrides -poseidon-tau=1.000000e-06
; FROMSITE: "tau_from": "site"
; FROMFLAG: "tau_from": "flag"
; KEY: error: {{.*}}Poseidon: unsupported Poseidon annotation argument poseidon_confidence

; The site's modelled error sums its units; alone or jointly the same pair is
; selected. Just above the FP16 raise's own error the TCEC raise is taken.
; SUM: [poseidon] preprocess_matmul_body: tau=1.000000e-04 S=5.101078e+04 A0=2.628106e-14 selected cost=-1.051897e-01 accCost=4.434190e+00 rel=8.692653e-05{{$}}
; SUM: [poseidon] pin: matmul_body:M0=wmma m16n16k16 f16/f32,matmul_body:P0=All FP64(0%) + FP32(100%){{$}}
; SUM-EXPLAIN: site 8.692653e-05 product 8.692588e-05
; SHARE-NOT: M0=wmma
; SHARE: [poseidon] preprocess_matmul_body: tau=2.909958e-04 {{.*}} rel=1.583396e-07{{$}}
; SHARE: [poseidon] pin: matmul_body:M0=tcec n=2 wmma m16n16k16 f16/f32,matmul_body:P0=All FP64(0%) + FP32(100%){{$}}
; SHARE-EXPLAIN: site 1.583396e-07 product 1.576873e-07

source_filename = "tcec_matmul.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2, ptr noundef writeonly captures(none) %d) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  br label %11

7:                                                ; preds = %11
  %8 = add nuw nsw i32 %6, %4
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %23, ptr %10, align 8, !tbaa !9
  %ea = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %ex = load double, ptr %ea, align 8, !tbaa !9
  %e1 = fmul double %ex, %ex
  %e2 = fadd double %e1, %ex
  %e3 = fmul double %e2, %ex
  %ed = getelementptr inbounds nuw [8 x i8], ptr %d, i64 %9
  store double %e3, ptr %ed, align 8, !tbaa !9
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = add nuw nsw i32 %12, %6
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %15
  %17 = load double, ptr %16, align 8, !tbaa !9
  %18 = shl nuw nsw i32 %12, 4
  %19 = add nuw nsw i32 %18, %4
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %20
  %22 = load double, ptr %21, align 8, !tbaa !9
  %23 = tail call double @llvm.fmuladd.f64(double %17, double %22, double %13)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %7, label %11, !llvm.loop !11
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %d, ptr noundef %dd) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !5
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !5
  tail call void (ptr, ...) @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3, i32 noundef %6, ptr noundef %d, ptr noundef %dd) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, ...) local_unnamed_addr #3
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4
attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !7, i64 0}
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.unroll.disable"}
