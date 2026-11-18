; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg)" -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)

; 1. Constant non-unit stride (x[2 * j]) inside invariant outer loop k = 0..4
define void @const_stride(ptr noalias %x, ptr noalias %y, ptr noalias %z) {
entry:
  br label %loop1

loop1:
  %k = phi i64 [ 0, %entry ], [ %k1, %cleanup ]
  %yk_gep = getelementptr inbounds double, ptr %y, i64 %k
  %yk = load double, ptr %yk_gep, align 8
  br label %loop2

loop2:
  %j = phi i64 [ 0, %loop1 ], [ %j1, %loop2 ]
  %idx = mul nuw nsw i64 %j, 2
  %x_gep = getelementptr inbounds double, ptr %x, i64 %idx
  %xj = load double, ptr %x_gep, align 8
  %prod = fmul fast double %yk, %xj
  %out_idx = add nuw nsw i64 %j, %k
  %z_gep = getelementptr inbounds double, ptr %z, i64 %out_idx
  store double %prod, ptr %z_gep, align 8
  %j1 = add nuw nsw i64 %j, 1
  %exit2 = icmp eq i64 %j1, 4
  br i1 %exit2, label %cleanup, label %loop2

cleanup:
  %k1 = add nuw nsw i64 %k, 1
  %exit1 = icmp eq i64 %k1, 4
  br i1 %exit1, label %exit, label %loop1

exit:
  store double 0.000000e+00, ptr %x, align 8
  ret void
}

; 2. Runtime loop-invariant stride (x[stride * j]) inside invariant outer loop k = 0..4
define void @runtime_stride(ptr noalias %x, ptr noalias %y, ptr noalias %z, i64 %stride) {
entry:
  br label %loop1

loop1:
  %k = phi i64 [ 0, %entry ], [ %k1, %cleanup ]
  %yk_gep = getelementptr inbounds double, ptr %y, i64 %k
  %yk = load double, ptr %yk_gep, align 8
  br label %loop2

loop2:
  %j = phi i64 [ 0, %loop1 ], [ %j1, %loop2 ]
  %idx = mul nsw i64 %j, %stride
  %x_gep = getelementptr inbounds double, ptr %x, i64 %idx
  %xj = load double, ptr %x_gep, align 8
  %prod = fmul fast double %yk, %xj
  %out_idx = add nuw nsw i64 %j, %k
  %z_gep = getelementptr inbounds double, ptr %z, i64 %out_idx
  store double %prod, ptr %z_gep, align 8
  %j1 = add nuw nsw i64 %j, 1
  %exit2 = icmp eq i64 %j1, 4
  br i1 %exit2, label %cleanup, label %loop2

cleanup:
  %k1 = add nuw nsw i64 %k, 1
  %exit1 = icmp eq i64 %k1, 4
  br i1 %exit1, label %exit, label %loop1

exit:
  store double 0.000000e+00, ptr %x, align 8
  ret void
}

; 3. Negative stride (x[3 - j]) inside invariant outer loop k = 0..4
define void @neg_stride(ptr noalias %x, ptr noalias %y, ptr noalias %z) {
entry:
  br label %loop1

loop1:
  %k = phi i64 [ 0, %entry ], [ %k1, %cleanup ]
  %yk_gep = getelementptr inbounds double, ptr %y, i64 %k
  %yk = load double, ptr %yk_gep, align 8
  br label %loop2

loop2:
  %j = phi i64 [ 0, %loop1 ], [ %j1, %loop2 ]
  %idx = sub nsw i64 3, %j
  %x_gep = getelementptr inbounds double, ptr %x, i64 %idx
  %xj = load double, ptr %x_gep, align 8
  %prod = fmul fast double %yk, %xj
  %out_idx = add nuw nsw i64 %j, %k
  %z_gep = getelementptr inbounds double, ptr %z, i64 %out_idx
  store double %prod, ptr %z_gep, align 8
  %j1 = add nuw nsw i64 %j, 1
  %exit2 = icmp eq i64 %j1, 4
  br i1 %exit2, label %cleanup, label %loop2

cleanup:
  %k1 = add nuw nsw i64 %k, 1
  %exit1 = icmp eq i64 %k1, 4
  br i1 %exit1, label %exit, label %loop1

exit:
  store double 0.000000e+00, ptr %x, align 8
  ret void
}

; 4. Zero-stride loop-invariant scalar load (*scale) inside nested loops
define void @zero_stride(ptr noalias %scale, ptr noalias %y, ptr noalias %z) {
entry:
  br label %loop1

loop1:
  %k = phi i64 [ 0, %entry ], [ %k1, %cleanup ]
  br label %loop2

loop2:
  %j = phi i64 [ 0, %loop1 ], [ %j1, %loop2 ]
  %s = load double, ptr %scale, align 8
  %yj_gep = getelementptr inbounds double, ptr %y, i64 %j
  %yj = load double, ptr %yj_gep, align 8
  %prod = fmul fast double %s, %yj
  %out_idx = add nuw nsw i64 %j, %k
  %z_gep = getelementptr inbounds double, ptr %z, i64 %out_idx
  store double %prod, ptr %z_gep, align 8
  %j1 = add nuw nsw i64 %j, 1
  %exit2 = icmp eq i64 %j1, 4
  br i1 %exit2, label %cleanup, label %loop2

cleanup:
  %k1 = add nuw nsw i64 %k, 1
  %exit1 = icmp eq i64 %k1, 4
  br i1 %exit1, label %exit, label %loop1

exit:
  store double 0.000000e+00, ptr %scale, align 8
  ret void
}

define void @caller(ptr %x, ptr %dx, ptr %y, ptr %dy, ptr %z, ptr %dz, i64 %stride) {
entry:
  call void (...) @__enzyme_autodiff(ptr @const_stride, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_dup", ptr %z, ptr %dz)
  call void (...) @__enzyme_autodiff(ptr @runtime_stride, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_dup", ptr %z, ptr %dz, metadata !"enzyme_const", i64 %stride)
  call void (...) @__enzyme_autodiff(ptr @neg_stride, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_dup", ptr %z, ptr %dz)
  call void (...) @__enzyme_autodiff(ptr @zero_stride, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_dup", ptr %z, ptr %dz)
  ret void
}

; CHECK-LABEL: define internal void @diffeconst_stride(
; CHECK-NOT: @malloc(i64 128)
; CHECK: %[[c_cache:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) ptr @malloc(i64 32)
; CHECK: strided.cache.loop:
; CHECK:   %strided.iv = phi i64 [ 0, %entry ], [ %strided.iv.next, %strided.cache.loop ]
; CHECK:   %strided.byte.off = mul i64 %strided.iv, 16
; CHECK:   %strided.src.i8 = getelementptr i8, ptr %x, i64 %strided.byte.off
; CHECK:   %strided.ld = load double, ptr %strided.src.i8, align 8
; CHECK:   %strided.dst = getelementptr inbounds double, ptr %[[c_cache]], i64 %strided.iv
; CHECK:   store double %strided.ld, ptr %strided.dst, align 8
; CHECK: invertloop2:
; CHECK:   getelementptr inbounds double, ptr %[[c_cache]], i64 %"iv1'ac.0"

; CHECK-LABEL: define internal void @differuntime_stride(
; CHECK-NOT: @malloc(i64 128)
; CHECK: %[[r_cache:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) ptr @malloc(i64 32)
; CHECK: strided.cache.loop:
; CHECK:   %strided.iv = phi i64 [ 0, %entry ], [ %strided.iv.next, %strided.cache.loop ]
; CHECK:   %strided.byte.off = mul i64 %strided.iv, %{{.+}}
; CHECK:   %strided.src.i8 = getelementptr i8, ptr %x, i64 %strided.byte.off
; CHECK:   %strided.ld = load double, ptr %strided.src.i8, align 8
; CHECK:   %strided.dst = getelementptr inbounds double, ptr %[[r_cache]], i64 %strided.iv
; CHECK:   store double %strided.ld, ptr %strided.dst, align 8
; CHECK: invertloop2:
; CHECK:   getelementptr inbounds double, ptr %[[r_cache]], i64 %"iv1'ac.0"

; CHECK-LABEL: define internal void @diffeneg_stride(
; CHECK-NOT: @malloc(i64 128)
; CHECK: %[[n_cache:.+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) ptr @malloc(i64 32)
; CHECK: strided.cache.loop:
; CHECK:   %strided.iv = phi i64 [ 0, %entry ], [ %strided.iv.next, %strided.cache.loop ]
; CHECK:   %strided.byte.off = mul i64 %strided.iv, -8
; CHECK:   %strided.src.i8 = getelementptr i8, ptr %{{.+}}, i64 %strided.byte.off
; CHECK:   %strided.ld = load double, ptr %strided.src.i8, align 8
; CHECK:   %strided.dst = getelementptr inbounds double, ptr %[[n_cache]], i64 %strided.iv
; CHECK:   store double %strided.ld, ptr %strided.dst, align 8
; CHECK: invertloop2:
; CHECK:   getelementptr inbounds double, ptr %[[n_cache]], i64 %"iv1'ac.0"

; CHECK-LABEL: define internal void @diffezero_stride(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[s_val:.+]] = load double, ptr %scale, align 8
; CHECK-NOT: @malloc
; CHECK: invertentry:
; CHECK-NEXT:   ret void
; CHECK: invertloop2:
; CHECK:   fmul fast double %{{.+}}, %[[s_val]]
; CHECK: }
