; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=analyze_omp_and_slice -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=analyze_omp_and_slice -disable-output | FileCheck %s

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@omp_str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@omp_ident = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 0, ptr @omp_str }, align 8

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)
declare void @__kmpc_for_static_fini(ptr, i32)
declare void @__kmpc_fork_call(ptr, i32, ptr, ...)

define internal void @omp_read_task(ptr noalias nocapture readonly %gtid_ptr, ptr noalias nocapture readnone %btid_ptr, ptr nocapture readonly %buf_slot, ptr nocapture writeonly %sink) {
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
  %lb = load i32, ptr %lower, align 4
  %ub = load i32, ptr %upper, align 4
  %cmp = icmp sle i32 %lb, %ub
  br i1 %cmp, label %loop.preheader, label %exit

loop.preheader:
  %buf = load ptr, ptr %buf_slot, align 8
  %lb64 = sext i32 %lb to i64
  %ub64 = sext i32 %ub to i64
  br label %loop

loop:
  %iv = phi i64 [ %lb64, %loop.preheader ], [ %iv.next, %loop ]
  %elem_ptr = getelementptr inbounds double, ptr %buf, i64 %iv
  %val = load double, ptr %elem_ptr, align 8
  %scaled = fmul double %val, 2.000000e+00
  store double %scaled, ptr %sink, align 8
  %iv.next = add nsw i64 %iv, 1
  %cont = icmp sle i64 %iv.next, %ub64
  br i1 %cont, label %loop, label %exit

exit:
  call void @__kmpc_for_static_fini(ptr nonnull @omp_ident, i32 %gtid)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %last)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %stride)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %upper)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %lower)
  ret void
}

define double @analyze_omp_and_slice(ptr %buf_slot, ptr %sink, ptr %slice_buf, ptr %two_buf) {
entry:
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull @omp_ident, i32 2, ptr @omp_read_task, ptr nonnull %buf_slot, ptr nonnull %sink)

  %second = getelementptr inbounds double, ptr %slice_buf, i64 1
  store double 2.000000e+00, ptr %second, align 8

  %zero_slot = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %zero_slot)
  store i32 0, ptr %zero_slot, align 4
  %idx = load i32, ptr %zero_slot, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %zero_slot)
  %idx64 = sext i32 %idx to i64
  %elem_gep = getelementptr double, ptr %slice_buf, i64 %idx64
  %byte_off = shl nsw i64 %idx64, 3
  %byte_gep = getelementptr i8, ptr %slice_buf, i64 %byte_off

  %two_stores = alloca i32, align 4
  store i32 0, ptr %two_stores, align 4
  store i32 2, ptr %two_stores, align 4
  %two_idx = load i32, ptr %two_stores, align 4
  %two_idx64 = sext i32 %two_idx to i64
  %two_gep = getelementptr inbounds double, ptr %two_buf, i64 %two_idx64
  %loaded = load double, ptr %two_gep, align 8
  ret double %loaded
}

; CHECK-LABEL: analyze_omp_and_slice - {[-1]:Float@double}
; CHECK-NEXT: ptr %buf_slot: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@double}
; CHECK-NEXT: ptr %sink: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: ptr %slice_buf: {[-1]:Pointer, [-1,8]:Float@double}
; CHECK-NEXT: ptr %two_buf: {[-1]:Pointer}
; CHECK:   %elem_gep = getelementptr double, ptr %slice_buf, i64 %idx64: {[-1]:Pointer, [-1,8]:Float@double}
; CHECK:   %byte_gep = getelementptr i8, ptr %slice_buf, i64 %byte_off: {[-1]:Pointer, [-1,8]:Float@double}
