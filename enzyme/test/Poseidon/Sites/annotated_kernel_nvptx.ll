; DEFINE: %{use} = -poseidon-profile-use=%S/../Inputs/annotated_kernel \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; DEFINE:   -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 \
; DEFINE:   -poseidon-enable-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv
; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" %{use} -poseidon-cache=%t/attr -poseidon-cost-budget=-1e-9 -poseidon-print -S > %t/attr.out 2>&1
; RUN: FileCheck --check-prefixes=CHECK,BODY %s < %t/attr.out
; RUN: FileCheck --check-prefix=CLEAN %s < %t/attr.out
; RUN: sed -n '/^; ModuleID/,$p' %t/attr.out > %t/attr.ll
; RUN: FileCheck --check-prefixes=MODULE,BODY %s < %t/attr.ll
; RUN: grep -B3 -A4 '%ds.hi64[^ ]* = fpext' %t/attr.ll | FileCheck --check-prefix=JOIN %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-kernels=fms.* %{use} -poseidon-cache=%t/regex -poseidon-cost-budget=-1e-9 -poseidon-print -S 2>&1 | tee %t/regex.out | FileCheck --check-prefix=REGEX %s
; RUN: sed -n '/^; ModuleID/,$p' %t/regex.out > %t/regex.ll
; RUN: FileCheck --check-prefixes=MODULE,BODY %s < %t/regex.ll
; RUN: grep -B3 -A4 '%ds.hi64[^ ]* = fpext' %t/regex.ll | FileCheck --check-prefix=JOIN %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -S 2> %t/noprofile.err | FileCheck --check-prefixes=NOPROFILE,FP64 %s
; RUN: FileCheck --check-prefix=NOPROFILE-WARN %s < %t/noprofile.err
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" %{use} -poseidon-cache=%t/noop -poseidon-cost-budget=0 -S | FileCheck --check-prefixes=NOOP,FP64 %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" %{use} -poseidon-cache=%t/inlined -poseidon-cost-budget=-1e-9 -S | tee %t/inlined.ll | FileCheck --check-prefix=INLINED %s
; RUN: grep -B3 -A4 '%ds.hi64[^ ]* = fpext' %t/inlined.ll | FileCheck --check-prefix=JOIN %s

; A kernel carrying POSEIDON_OPTIMIZE is a site whose annotated computation is
; its whole body: same code and same profile as ds_expansion_nvptx.ll, with no
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
; CHECK: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; CHECK: [poseidon] pin: fms_kernel:PT0=All FP64(0%) + Expansion2(100%)

; Without inlining the kernel calls the rewritten body with its own arguments.
; MODULE-LABEL: define {{.*}} @fms_kernel(
; MODULE-SAME: ptr {{.*}}%[[KA:[0-9]+]], ptr {{.*}}%[[KB:[0-9]+]], ptr {{.*}}%[[KOUT:[0-9]+]])
; MODULE-NEXT: entry:
; MODULE-NEXT: call void @preprocess_fms_kernel_poseidon_body(ptr %[[KA]], ptr %[[KB]], ptr %[[KOUT]])
; MODULE-NEXT: ret void

; BODY-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; BODY-SAME: ptr {{.*}}%[[A:[0-9]+]], ptr {{.*}}%[[B:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; BODY: %[[I:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; BODY-NEXT: %[[XP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[A]], i64 %[[I]]
; BODY-NEXT: %[[X:[0-9]+]] = load double, ptr %[[XP]]
; BODY-NEXT: %[[YP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[B]], i64 %[[I]]
; BODY-NEXT: %[[Y:[0-9]+]] = load double, ptr %[[YP]]
; BODY-NEXT: %[[XHI:ds.hi[0-9]*]] = fptrunc double %[[X]] to float
; BODY-NEXT: %[[XHIB:ds.hib[0-9]*]] = fpext float %[[XHI]] to double
; BODY-NEXT: %[[XLOS:ds.los[0-9]*]] = fsub double %[[X]], %[[XHIB]]
; BODY-NEXT: %[[XLO:ds.lo[0-9]*]] = fptrunc double %[[XLOS]] to float
; BODY-NEXT: %[[YHI:ds.hi[0-9]*]] = fptrunc double %[[Y]] to float
; BODY-NEXT: %[[YHIB:ds.hib[0-9]*]] = fpext float %[[YHI]] to double
; BODY-NEXT: %[[YLOS:ds.los[0-9]*]] = fsub double %[[Y]], %[[YHIB]]
; BODY-NEXT: %[[YLO:ds.lo[0-9]*]] = fptrunc double %[[YLOS]] to float
; BODY-NEXT: %[[P:dsf.p[0-9]*]] = fmul float %[[XHI]], %[[YHI]]
; BODY-NEXT: %[[NP:dsf.np[0-9]*]] = fneg float %[[P]]
; BODY-NEXT: %[[E:dsf.e[0-9]*]] = {{.*}}call float @llvm.fma.f32(float %[[XHI]], float %[[YHI]], float %[[NP]])
; BODY-NEXT: %[[E1:dsf.e1[0-9]*]] = {{.*}}call float @llvm.fma.f32(float %[[XHI]], float %[[YLO]], float %[[E]])
; BODY-NEXT: %[[E2:dsf.e2[0-9]*]] = {{.*}}call float @llvm.fma.f32(float %[[XLO]], float %[[YHI]], float %[[E1]])
; BODY-NEXT: %[[S:ts.s[0-9]*]] = fadd float %[[P]], %[[XHI]]
; BODY-NEXT: %[[AP:ts.ap[0-9]*]] = fsub float %[[S]], %[[XHI]]
; BODY-NEXT: %[[BP:ts.bp[0-9]*]] = fsub float %[[S]], %[[AP]]
; BODY-NEXT: %[[DA:ts.da[0-9]*]] = fsub float %[[P]], %[[AP]]
; BODY-NEXT: %[[DB:ts.db[0-9]*]] = fsub float %[[XHI]], %[[BP]]
; BODY-NEXT: %[[TE:ts.e[0-9]*]] = fadd float %[[DA]], %[[DB]]
; BODY-NEXT: %[[T0:dsf.t0[0-9]*]] = fadd float %[[E2]], %[[XLO]]
; BODY-NEXT: %[[T1:dsf.t1[0-9]*]] = fadd float %[[T0]], %[[TE]]
; BODY-NEXT: %[[FS:fts.s[0-9]*]] = fadd float %[[S]], %[[T1]]
; BODY-NEXT: %[[FBP:fts.bp[0-9]*]] = fsub float %[[FS]], %[[S]]
; BODY-NEXT: %[[FE:fts.e[0-9]*]] = fsub float %[[T1]], %[[FBP]]
; BODY-NOT: double
; BODY: %[[HI64:ds.hi64[0-9]*]] = fpext float %{{.+}} to double
; BODY-NEXT: %[[LO64:ds.lo64[0-9]*]] = fpext float %{{.+}} to double
; BODY-NEXT: %[[J:ds.f64[0-9]*]] = fadd double %[[HI64]], %[[LO64]], !poseidon.ds.join
; BODY-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; BODY-NEXT: store double %[[J]], ptr %[[OP]]
; BODY-NEXT: ret void

; The joined pair is the last fast two-sum of the expansion.
; JOIN: [[S:%[^ ]+]] = fadd float [[X:%[^ ,]+]], [[Y:%[^ ,]+]]{{$}}
; JOIN-NEXT: [[BP:%[^ ]+]] = fsub float [[S]], [[X]]{{$}}
; JOIN-NEXT: [[E:%[^ ]+]] = fsub float [[Y]], [[BP]]{{$}}
; JOIN-NEXT: [[HI:%[^ ]+]] = fpext float [[S]] to double
; JOIN-NEXT: [[LO:%[^ ]+]] = fpext float [[E]] to double
; JOIN-NEXT: [[J:%[^ ]+]] = fadd double [[HI]], [[LO]], !poseidon.ds.join !{{[0-9]+}}
; JOIN-NEXT: [[OP:%[0-9]+]] = getelementptr inbounds {{.*}}, i64 %{{[0-9]+}}
; JOIN-NEXT: store double [[J]], ptr [[OP]]

; CLEAN-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; CLEAN-NOT: call double @llvm.fmuladd.f64
; CLEAN-NOT: fmul double
; CLEAN-NOT: fneg double
; CLEAN: ret void

; REGEX: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; REGEX: [poseidon] pin: fms_kernel:PT0=All FP64(0%) + Expansion2(100%)
; REGEX-LABEL: define {{.*}} @fms_kernel(
; REGEX: call void @preprocess_fms_kernel_poseidon_body(

; The annotation alone must not change the kernel: with no profile the site is
; left as written (no outline, no call), and with a profile whose solve applies
; nothing the outline is folded back. With a rewrite applied the materialized
; body is folded back the same way. In every case the kernel is one entry
; function with the arithmetic inline and no _poseidon_body symbol survives.
; NOPROFILE-WARN: warning: <unknown>:0:0: in function fms_kernel void (ptr, ptr, ptr): Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>. Emitting the original computation.
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
; INLINED-NEXT: %[[XP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[A]], i64 %[[I]]
; INLINED-NEXT: %[[X:[0-9]+]] = load double, ptr %[[XP]]
; INLINED-NEXT: %[[YP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[B]], i64 %[[I]]
; INLINED-NEXT: %[[Y:[0-9]+]] = load double, ptr %[[YP]]
; INLINED-NEXT: [[XHI:%[^ ]+]] = fptrunc double %[[X]] to float
; INLINED-NEXT: [[XHIB:%[^ ]+]] = fpext float [[XHI]] to double
; INLINED-NEXT: [[XLOS:%[^ ]+]] = fsub double %[[X]], [[XHIB]]
; INLINED-NEXT: [[XLO:%[^ ]+]] = fptrunc double [[XLOS]] to float
; INLINED-NEXT: [[YHI:%[^ ]+]] = fptrunc double %[[Y]] to float
; INLINED-NEXT: [[YHIB:%[^ ]+]] = fpext float [[YHI]] to double
; INLINED-NEXT: [[YLOS:%[^ ]+]] = fsub double %[[Y]], [[YHIB]]
; INLINED-NEXT: [[YLO:%[^ ]+]] = fptrunc double [[YLOS]] to float
; INLINED-NEXT: [[P:%[^ ]+]] = fmul float [[XHI]], [[YHI]]
; INLINED-NEXT: [[NP:%[^ ]+]] = fneg float [[P]]
; INLINED-NEXT: [[E:%[^ ]+]] = {{.*}}call float @llvm.fma.f32(float [[XHI]], float [[YHI]], float [[NP]])
; INLINED-NEXT: [[E1:%[^ ]+]] = {{.*}}call float @llvm.fma.f32(float [[XHI]], float [[YLO]], float [[E]])
; INLINED-NEXT: {{.*}}call float @llvm.fma.f32(float [[XLO]], float [[YHI]], float [[E1]])
; INLINED-NOT: double
; INLINED: [[HI64:%[^ ]+]] = fpext float %{{[^ ]+}} to double
; INLINED-NEXT: [[LO64:%[^ ]+]] = fpext float %{{[^ ]+}} to double
; INLINED-NEXT: [[J:%[^ ]+]] = fadd double [[HI64]], [[LO64]], !poseidon.ds.join !{{[0-9]+}}
; INLINED-NEXT: %[[OP:[0-9]+]] = getelementptr inbounds {{.*}}ptr %[[OUT]], i64 %[[I]]
; INLINED-NEXT: store double [[J]], ptr %[[OP]]
; INLINED-NEXT: ret void
; INLINED-NOT: _poseidon_body
