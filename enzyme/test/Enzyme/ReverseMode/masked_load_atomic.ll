; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli -

@loc = private unnamed_addr constant { i32, i32, i32, i32, ptr } { i32 0, i32 2, i32 0, i32 0, ptr null }, align 8

@g_x = internal global [4 x float] [float 2.000000e+00, float 1.000000e+02, float 4.000000e+00, float 2.000000e+02], align 16, !enzyme_shadow !{ptr @g_dx}
@g_dx = internal global [4 x float] zeroinitializer, align 16
@g_out = internal global [2 x float] zeroinitializer, align 8, !enzyme_shadow !{ptr @g_dout}
@g_dout = internal global [2 x float] zeroinitializer, align 8
@g_limit = internal global i32 3, align 4

declare void @__enzyme_autodiff(ptr, ...)
declare void @abort()
declare <4 x float> @llvm.masked.load.v4f32.p0(ptr, i32 immarg, <4 x i1>, <4 x float>)
declare <1 x double> @llvm.masked.load.v1f64.p0(ptr, i32 immarg, <1 x i1>, <1 x double>)

define void @__kmpc_fork_call(ptr %loc_arg, i32 %argc, ptr %microtask, ...) noinline {
entry:
  %tid = alloca i32, align 4
  store i32 0, ptr %tid, align 4
  call void %microtask(ptr %tid, ptr %tid)
  store i32 1, ptr %tid, align 4
  call void %microtask(ptr %tid, ptr %tid)
  ret void
}

define internal void @omp_global_body(ptr noalias readonly %gtid_ptr, ptr noalias readnone %btid_ptr) {
entry:
  %tid = load i32, ptr %gtid_ptr, align 4
  %lim = load i32, ptr @g_limit, align 4
  %lim_splat_ins = insertelement <4 x i32> poison, i32 %lim, i64 0
  %lim_vec = shufflevector <4 x i32> %lim_splat_ins, <4 x i32> poison, <4 x i32> zeroinitializer
  %base_mask = icmp slt <4 x i32> <i32 0, i32 1, i32 2, i32 3>, %lim_vec
  %even_mask = and <4 x i1> %base_mask, <i1 true, i1 false, i1 true, i1 true>
  %vec = call <4 x float> @llvm.masked.load.v4f32.p0(ptr @g_x, i32 4, <4 x i1> %even_mask, <4 x float> zeroinitializer)
  %e0 = extractelement <4 x float> %vec, i64 0
  %e2 = extractelement <4 x float> %vec, i64 2
  %sum = fadd float %e0, %e2
  %tid_plus_1 = add i32 %tid, 1
  %scale = sitofp i32 %tid_plus_1 to float
  %scaled = fmul float %sum, %scale
  %tid64 = sext i32 %tid to i64
  %out_slot = getelementptr inbounds [2 x float], ptr @g_out, i64 0, i64 %tid64
  store float %scaled, ptr %out_slot, align 4
  ret void
}

define void @run_omp_global() {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @loc, i32 0, ptr nonnull @omp_global_body)
  ret void
}

define internal void @omp_arg_body(ptr noalias readonly %gtid_ptr, ptr noalias readnone %btid_ptr, ptr noalias readonly %x, ptr noalias %out, ptr noalias readonly %mask_ptr) {
entry:
  %tid = load i32, ptr %gtid_ptr, align 4
  %mask = load <4 x i1>, ptr %mask_ptr, align 1
  %vec = call <4 x float> @llvm.masked.load.v4f32.p0(ptr %x, i32 4, <4 x i1> %mask, <4 x float> zeroinitializer)
  %e0 = extractelement <4 x float> %vec, i64 0
  %e1 = extractelement <4 x float> %vec, i64 1
  %e2 = extractelement <4 x float> %vec, i64 2
  %e3 = extractelement <4 x float> %vec, i64 3
  %s01 = fadd float %e0, %e1
  %s23 = fadd float %e2, %e3
  %sum = fadd float %s01, %s23
  %tid64 = sext i32 %tid to i64
  %out_slot = getelementptr inbounds float, ptr %out, i64 %tid64
  store float %sum, ptr %out_slot, align 4
  ret void
}

define void @omp_arg_kernel(ptr %x, ptr %out, ptr %mask_ptr) {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @loc, i32 3, ptr nonnull @omp_arg_body, ptr %x, ptr %out, ptr %mask_ptr)
  ret void
}

define void @diff_arg_kernel(ptr %x, ptr %dx, ptr %out, ptr %dout, ptr %mask_ptr) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @omp_arg_kernel, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %mask_ptr)
  ret void
}

define void @diff_arg_kernel_runtime(ptr %x, ptr %dx, ptr %out, ptr %dout, ptr %mask_ptr) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @omp_arg_kernel, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %mask_ptr)
  ret void
}

define internal void @omp_v1_body(ptr noalias readonly %gtid_ptr, ptr noalias readnone %btid_ptr, ptr noalias readonly %x, ptr noalias %out, ptr noalias readonly %mask_ptr) {
entry:
  %tid = load i32, ptr %gtid_ptr, align 4
  %mask = load <1 x i1>, ptr %mask_ptr, align 1
  %vec = call <1 x double> @llvm.masked.load.v1f64.p0(ptr %x, i32 8, <1 x i1> %mask, <1 x double> zeroinitializer)
  %val = extractelement <1 x double> %vec, i64 0
  %tid64 = sext i32 %tid to i64
  %out_slot = getelementptr inbounds double, ptr %out, i64 %tid64
  store double %val, ptr %out_slot, align 8
  ret void
}

define void @omp_v1_kernel(ptr %x, ptr %out, ptr %mask_ptr) {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @loc, i32 3, ptr nonnull @omp_v1_body, ptr %x, ptr %out, ptr %mask_ptr)
  ret void
}

define void @diff_v1_kernel(ptr %x, ptr %dx, ptr %out, ptr %dout, ptr %mask_ptr) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @omp_v1_kernel, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %mask_ptr)
  ret void
}

define i32 @main() {
entry:
  store float 0.000000e+00, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 0), align 16
  store float -7.700000e+01, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 1), align 4
  store float 0.000000e+00, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 2), align 8
  store float -9.900000e+01, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 3), align 4
  store float 3.000000e+00, ptr getelementptr inbounds ([2 x float], ptr @g_dout, i64 0, i64 0), align 8
  store float 5.000000e+00, ptr getelementptr inbounds ([2 x float], ptr @g_dout, i64 0, i64 1), align 4

  call void (ptr, ...) @__enzyme_autodiff(ptr @run_omp_global)
  store float 1.000000e+00, ptr getelementptr inbounds ([2 x float], ptr @g_dout, i64 0, i64 0), align 8
  store float 2.000000e+00, ptr getelementptr inbounds ([2 x float], ptr @g_dout, i64 0, i64 1), align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @run_omp_global, metadata !"enzyme_runtime_activity")

  %dx0 = load float, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 0), align 16
  %dx1 = load float, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 1), align 4
  %dx2 = load float, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 2), align 8
  %dx3 = load float, ptr getelementptr inbounds ([4 x float], ptr @g_dx, i64 0, i64 3), align 4

  %ok0 = fcmp oeq float %dx0, 1.800000e+01
  %ok1 = fcmp oeq float %dx1, -7.700000e+01
  %ok2 = fcmp oeq float %dx2, 1.800000e+01
  %ok3 = fcmp oeq float %dx3, -9.900000e+01
  %ok01 = and i1 %ok0, %ok1
  %ok23 = and i1 %ok2, %ok3
  %all_ok = and i1 %ok01, %ok23
  br i1 %all_ok, label %done, label %fail

fail:
  call void @abort()
  unreachable

done:
  ret i32 0
}

; CHECK-LABEL: define internal void @diffeomp_arg_body(
; CHECK: %[[M0:.+]] = extractelement <4 x i1> %mask, i64 0
; CHECK-NEXT: br i1 %[[M0]], label %invertentry_atomic, label %invertentry_merge
; CHECK: invertentry_atomic:
; CHECK-NEXT: %[[D0:.+]] = extractelement <4 x float> %[[DIF:.+]], i64 0
; CHECK-NEXT: %{{.+}} = atomicrmw fadd ptr %"x'", float %[[D0]] monotonic, align 4
; CHECK-NEXT: br label %invertentry_merge
; CHECK: invertentry_merge:
; CHECK-NEXT: %[[M1:.+]] = extractelement <4 x i1> %mask, i64 1
; CHECK-NEXT: br i1 %[[M1]], label %invertentry_atomic1, label %invertentry_merge2
; CHECK: invertentry_atomic1:
; CHECK-NEXT: %[[D1:.+]] = extractelement <4 x float> %[[DIF]], i64 1
; CHECK-NEXT: %[[P1:.+]] = getelementptr <4 x float>, ptr %"x'", i64 0, i32 1
; CHECK-NEXT: %{{.+}} = atomicrmw fadd ptr %[[P1]], float %[[D1]] monotonic, align 4
; CHECK-NEXT: br label %invertentry_merge2

; CHECK-LABEL: define internal void @diffeomp_arg_body.{{[0-9]+}}(
; CHECK: %[[ACT:.+]] = icmp ne ptr %x, %"x'"
; CHECK-NEXT: br i1 %[[ACT]], label %invertentry_amerge_active, label %invertentry_amerge_amerge
; CHECK: invertentry_amerge_active:
; CHECK-NEXT: %[[RM0:.+]] = extractelement <4 x i1> %mask, i64 0
; CHECK-NEXT: br i1 %[[RM0]], label %invertentry_amerge_active_atomic, label %invertentry_amerge_active_merge
; CHECK: invertentry_amerge_active_atomic:
; CHECK-NEXT: %[[RD0:.+]] = extractelement <4 x float> %{{.+}}, i64 0
; CHECK-NEXT: %{{.+}} = atomicrmw fadd ptr %"x'", float %[[RD0]] monotonic, align 4
; CHECK-NEXT: br label %invertentry_amerge_active_merge
; CHECK: invertentry_amerge_amerge:

; CHECK-LABEL: define internal void @diffeomp_v1_body(
; CHECK: %[[VM0:.+]] = extractelement <1 x i1> %mask, i64 0
; CHECK-NEXT: br i1 %[[VM0]], label %invertentry_atomic, label %invertentry_merge
; CHECK: invertentry_atomic:
; CHECK-NEXT: %{{.+}} = atomicrmw fadd ptr %"x'", double %{{.+}} monotonic, align 8
; CHECK-NEXT: br label %invertentry_merge
