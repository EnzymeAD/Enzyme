; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme-loose-types=0 -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-loose-types=0 -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@omp_str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@omp_ident = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @omp_str }, align 8

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)
declare void @__kmpc_for_static_fini(ptr, i32)
declare void @__kmpc_fork_call(ptr, i32, ptr, ...)
declare void @__enzyme_autodiff(ptr, ...)

define internal void @omp_zero_slice(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture readonly %bound_ptr, ptr nocapture readonly %out_slot) {
entry:
  %lower = alloca i32, align 4
  %upper = alloca i32, align 4
  %stride = alloca i32, align 4
  %last = alloca i32, align 4
  %bound = load i32, ptr %bound_ptr, align 4
  %has_work = icmp sgt i32 %bound, -1
  br i1 %has_work, label %omp.init, label %done

omp.init:
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %lower)
  store i32 0, ptr %lower, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %upper)
  store i32 %bound, ptr %upper, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %stride)
  store i32 1, ptr %stride, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %last)
  store i32 0, ptr %last, align 4
  %gtid = load i32, ptr %gtid_ptr, align 4
  call void @__kmpc_for_static_init_4(ptr nonnull @omp_ident, i32 %gtid, i32 34, ptr nonnull %last, ptr nonnull %lower, ptr nonnull %upper, ptr nonnull %stride, i32 1, i32 1)
  %raw_ub = load i32, ptr %upper, align 4
  %over = icmp sgt i32 %raw_ub, %bound
  %clamped_ub = select i1 %over, i32 %bound, i32 %raw_ub
  store i32 %clamped_ub, ptr %upper, align 4
  %lb = load i32, ptr %lower, align 4
  %skip = icmp slt i32 %clamped_ub, %lb
  br i1 %skip, label %omp.fini, label %omp.body

omp.body:
  %out_ptr = load ptr, ptr %out_slot, align 8
  %lb64 = sext i32 %lb to i64
  %byte_off = shl nsw i64 %lb64, 3
  %dst = getelementptr i8, ptr %out_ptr, i64 %byte_off
  %span = sub i32 %clamped_ub, %lb
  %span64 = zext i32 %span to i64
  %scaled = shl nuw nsw i64 %span64, 3
  %bytes = add nuw nsw i64 %scaled, 8
  call void @llvm.memset.p0.i64(ptr align 8 %dst, i8 0, i64 %bytes, i1 false)
  br label %omp.fini

omp.fini:
  call void @__kmpc_for_static_fini(ptr nonnull @omp_ident, i32 %gtid)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %last)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %stride)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %upper)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %lower)
  br label %done

done:
  ret void
}

define internal void @omp_copy_slice(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture readonly %dst_slot, ptr nocapture readonly %src_slot) {
entry:
  %lower = alloca i32, align 4
  %upper = alloca i32, align 4
  %stride = alloca i32, align 4
  %last = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %lower)
  store i32 0, ptr %lower, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %upper)
  store i32 1, ptr %upper, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %stride)
  store i32 1, ptr %stride, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %last)
  store i32 0, ptr %last, align 4
  %gtid = load i32, ptr %gtid_ptr, align 4
  call void @__kmpc_for_static_init_4(ptr nonnull @omp_ident, i32 %gtid, i32 34, ptr nonnull %last, ptr nonnull %lower, ptr nonnull %upper, ptr nonnull %stride, i32 1, i32 1)
  %raw_ub = load i32, ptr %upper, align 4
  %over = icmp sgt i32 %raw_ub, 1
  %clamped_ub = select i1 %over, i32 1, i32 %raw_ub
  store i32 %clamped_ub, ptr %upper, align 4
  %lb = load i32, ptr %lower, align 4
  %skip = icmp sgt i32 %lb, %clamped_ub
  br i1 %skip, label %exit, label %preheader

preheader:
  %src_ptr = load ptr, ptr %src_slot, align 8
  %dst_ptr = load ptr, ptr %dst_slot, align 8
  %lb64 = sext i32 %lb to i64
  %end32 = add nsw i32 %clamped_ub, 1
  br label %loop

loop:
  %iv = phi i64 [ %lb64, %preheader ], [ %iv.next, %loop ]
  %src_elem = getelementptr inbounds double, ptr %src_ptr, i64 %iv
  %val = load double, ptr %src_elem, align 8
  %scaled = fmul double %val, 2.000000e+00
  %dst_elem = getelementptr inbounds double, ptr %dst_ptr, i64 %iv
  store double %scaled, ptr %dst_elem, align 8
  %iv.next = add nsw i64 %iv, 1
  %iv32 = trunc i64 %iv.next to i32
  %done_loop = icmp eq i32 %iv32, %end32
  br i1 %done_loop, label %exit, label %loop

exit:
  call void @__kmpc_for_static_fini(ptr nonnull @omp_ident, i32 %gtid)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %last)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %stride)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %upper)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %lower)
  ret void
}

define void @omp_kernel(ptr nocapture readonly %in, ptr nocapture %out, i32 %flag) {
entry:
  %stack = alloca [64 x i8], align 16
  %bound_slot = alloca i32, align 4
  %slice_slot = alloca ptr, align 8
  %dst_slot = alloca ptr, align 8
  store i32 %flag, ptr %bound_slot, align 4
  %slice = getelementptr inbounds [64 x i8], ptr %stack, i64 0, i64 16
  store ptr %slice, ptr %slice_slot, align 8
  store ptr %out, ptr %dst_slot, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @omp_ident, i32 2, ptr @omp_zero_slice, ptr nonnull %bound_slot, ptr nonnull %slice_slot)
  %is_zero = icmp eq i32 %flag, 0
  br i1 %is_zero, label %write_second, label %copy_out

write_second:
  %in_val = load double, ptr %in, align 8
  %cur_slice = load ptr, ptr %slice_slot, align 8
  %second = getelementptr inbounds double, ptr %cur_slice, i64 1
  store double %in_val, ptr %second, align 8
  br label %copy_out

copy_out:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @omp_ident, i32 2, ptr @omp_copy_slice, ptr nonnull %dst_slot, ptr nonnull %slice_slot)
  ret void
}

define void @test_derivative(ptr %in, ptr %din, ptr %out, ptr %dout) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @omp_kernel, ptr %in, ptr %din, ptr %out, ptr %dout, i32 0)
  ret void
}

; CHECK-LABEL: define internal void @diffeomp_kernel(
; CHECK-DAG: call void (ptr, i32, ptr, ...) @__kmpc_fork_call({{.*}}@diffeomp_copy_slice
; CHECK-DAG: call void (ptr, i32, ptr, ...) @__kmpc_fork_call({{.*}}@diffeomp_zero_slice

; CHECK-LABEL: define internal void @diffeomp_zero_slice(
; CHECK: call void @llvm.memset.p0.i64(
