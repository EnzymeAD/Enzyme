; DEFINE: %{use} = -poseidon-profile-use=%t/prof \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/expansion_two_components/fms_body.fpprofile %t/prof/fms_kernel.fpprofile
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" %{use} -poseidon-cache=%t/attr -poseidon-cost-budget=-1e-9 -poseidon-print -S > %t/attr.out 2>&1
; RUN: FileCheck %s < %t/attr.out
; RUN: FileCheck --check-prefix=CLEAN %s < %t/attr.out
; RUN: sed -n '/^; ModuleID/,$p' %t/attr.out > %t/attr.ll
; RUN: FileCheck --check-prefixes=MODULE,BODY %s < %t/attr.ll
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-kernels=fms.* %{use} -poseidon-cache=%t/regex -poseidon-cost-budget=-1e-9 -poseidon-print -S 2>&1 | FileCheck --check-prefix=REGEX %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -S 2> %t/noprofile.err | FileCheck --check-prefixes=NOPROFILE,FP64 %s
; RUN: FileCheck --check-prefix=NOPROFILE-WARN %s < %t/noprofile.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" %{use} -poseidon-cache=%t/noop -poseidon-cost-budget=0 -S 2> %t/noop.err | FileCheck --check-prefixes=NOOP,FP64 %s
; RUN: count 0 < %t/noop.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" %{use} -poseidon-cache=%t/inlined -poseidon-cost-budget=-1e-9 -S 2> %t/inlined.err | FileCheck --check-prefix=INLINED %s
; RUN: count 0 < %t/inlined.err

; A kernel carrying POSEIDON_OPTIMIZE is a site whose annotated computation is
; its whole body: same code and same profile as expansion_two_components_nvptx.ll, with no
; marker call and no shadow arguments. The second run reaches the same site by
; name with -poseidon-kernels instead of the attribute.

; ModuleID = 'annotated_kernel.cu'
source_filename = "annotated_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [20 x i8] c"annotated_kernel.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @fms_kernel, ptr @.str, ptr @.str.1, i32 2, ptr null }], section "llvm.metadata"

define dso_local ptx_kernel void @fms_kernel(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %16, double %17)
  %21 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %20, ptr %21, align 8
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) #1

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}

; CHECK: Candidates:
; CHECK: All FP64(0%) + FP32(100%)
; CHECK: All FP64(0%) + Expansion2(100%)
; CHECK: Applying solution for precision change All FP64(0%) + Expansion2(100%)
; CHECK: [poseidon] pin: fms_kernel:P0=All FP64(0%) + Expansion2(100%)

; Without inlining the kernel calls the rewritten body with its own arguments.
; MODULE-LABEL: define {{.*}} @fms_kernel(
; MODULE-SAME: ptr {{.*}}%[[KA:[0-9]+]], ptr {{.*}}%[[KB:[0-9]+]], ptr {{.*}}%[[KOUT:[0-9]+]])
; MODULE-NEXT: entry:
; MODULE-NEXT: call void @preprocess_fms_kernel_poseidon_body(ptr %[[KA]], ptr %[[KB]], ptr %[[KOUT]])
; MODULE-NEXT: ret void

; The body is the double-single expansion of the chain (expansion_two_components_nvptx.ll
; checks its instructions), joined back to FP64 at the store.
; BODY-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; BODY-SAME: ptr {{.*}}%[[A:[0-9]+]], ptr {{.*}}%[[B:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; BODY: %[[I:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; BODY: fptrunc double %{{[0-9]+}} to float
; BODY: %[[J:pair.f64[0-9]*]] = fadd double %{{.+}}, %{{.+}}, !poseidon.pair.join
; BODY-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; BODY-NEXT: store double %[[J]], ptr %[[OP]]
; BODY-NEXT: ret void

; CLEAN-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; CLEAN-NOT: call double @llvm.fmuladd.f64
; CLEAN-NOT: fmul double
; CLEAN-NOT: fneg double
; CLEAN: ret void

; REGEX: Applying solution for precision change All FP64(0%) + Expansion2(100%)
; REGEX: [poseidon] pin: fms_kernel:P0=All FP64(0%) + Expansion2(100%)
; REGEX-LABEL: define {{.*}} @fms_kernel(
; REGEX: call void @preprocess_fms_kernel_poseidon_body(

; The annotation alone must not change the kernel: with no profile the site is
; left as written (no outline, no call), and with a profile whose solve applies
; nothing the outline is folded back. With a rewrite applied the materialized
; body is folded back the same way. In every case the kernel is one entry
; function with the arithmetic inline and no _poseidon_body symbol survives.
; NOPROFILE-WARN: warning: <unknown>:0:0: in function fms_kernel void (ptr, ptr, ptr): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it
; NOPROFILE-WARN-NOT: warning
; NOPROFILE-LABEL: define {{.*}} @fms_kernel(
; NOOP-LABEL: define {{.*}} @fms_kernel(
; FP64-SAME: ptr {{.*}}%[[A:[0-9]+]], ptr {{.*}}%[[B:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; NOPROFILE-NOT: call void @
; NOOP-NOT: call void @
; FP64: %[[I:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; FP64-NEXT: %[[XP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[A]], i64 %[[I]]
; FP64-NEXT: %[[X:[0-9]+]] = load double, ptr %[[XP]]
; FP64-NEXT: %[[YP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[B]], i64 %[[I]]
; FP64-NEXT: %[[Y:[0-9]+]] = load double, ptr %[[YP]]
; FP64-NEXT: %[[T0:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[X]], double %[[Y]], double %[[X]])
; FP64-NEXT: %[[NY:[0-9]+]] = fneg double %[[Y]]
; FP64-NEXT: %[[T1:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T0]], double %[[T0]], double %[[NY]])
; FP64-NEXT: %[[T2:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T1]], double %[[Y]], double %[[T0]])
; FP64-NEXT: %[[NX:[0-9]+]] = fneg double %[[X]]
; FP64-NEXT: %[[T3:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T2]], double %[[T2]], double %[[NX]])
; FP64-NEXT: %[[T4:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T3]], double %[[T1]], double %[[T2]])
; FP64-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; FP64-NEXT: store double %[[T4]], ptr %[[OP]]
; FP64-NEXT: ret void
; NOPROFILE-NOT: _poseidon_body
; NOOP-NOT: _poseidon_body

; INLINED-LABEL: define {{.*}} @fms_kernel(
; INLINED-SAME: ptr {{.*}}%[[A:[0-9]+]], ptr {{.*}}%[[B:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; INLINED-NOT: call void @
; INLINED: %[[I:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; INLINED: fptrunc double %{{[0-9]+}} to float
; INLINED-NOT: fmul double
; INLINED: [[J:%[^ ]+]] = fadd double %{{.+}}, %{{.+}}, !poseidon.pair.join !{{[0-9]+}}
; INLINED-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; INLINED-NEXT: store double [[J]], ptr %[[OP]]
; INLINED-NEXT: ret void
; INLINED-NOT: _poseidon_body
