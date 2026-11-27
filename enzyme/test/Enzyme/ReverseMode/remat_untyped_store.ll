; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 6.000000 10.000000

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare double @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr nocapture readonly, ...)

@.str = private unnamed_addr constant [7 x i8] c"%f %f\0A\00", align 1

define internal double @read_view(ptr nocapture readonly %view) noinline {
entry:
  %data_ptr = load ptr, ptr %view, align 8
  %value = load double, ptr %data_ptr, align 8
  %sq = fmul double %value, %value
  ret double %sq
}

; CHECK-LABEL: define internal void @diffecompute_sum(
define double @compute_sum(ptr nocapture readonly %input, i64 %count) {
entry:
  %view = alloca { ptr, i64 }, align 8
  br label %loop

loop:
  %index = phi i64 [ 0, %entry ], [ %next_index, %loop ]
  %accum = phi double [ 0.000000e+00, %entry ], [ %next_accum, %loop ]
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %view)
  %elem_ptr = getelementptr inbounds double, ptr %input, i64 %index
  store ptr %elem_ptr, ptr %view, align 8
  %unread_stride = getelementptr inbounds i8, ptr %view, i64 8
  store i64 0, ptr %unread_stride, align 8
; CHECK: call void @differead_view(ptr %view, ptr %"view'ipa", double %{{.+}})
  %term = call double @read_view(ptr nonnull %view)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %view)
  %next_accum = fadd double %accum, %term
  %next_index = add nuw nsw i64 %index, 1
  %done = icmp eq i64 %next_index, %count
  br i1 %done, label %exit, label %loop

; CHECK: remat_loop_loop:
; CHECK: store i64 0, ptr %"unread_stride'ipg_unwrap", align 8
exit:
  ret double %next_accum
}

define i32 @main() {
entry:
  %x = alloca [2 x double], align 16
  %dx = alloca [2 x double], align 16
  store double 3.000000e+00, ptr %x, align 16
  %x1 = getelementptr inbounds double, ptr %x, i64 1
  store double 5.000000e+00, ptr %x1, align 8
  store double 0.000000e+00, ptr %dx, align 16
  %dx1 = getelementptr inbounds double, ptr %dx, i64 1
  store double 0.000000e+00, ptr %dx1, align 8
  %call = call double (ptr, ...) @__enzyme_autodiff(ptr @compute_sum, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i64 2)
  %g0 = load double, ptr %dx, align 16
  %g1 = load double, ptr %dx1, align 8
  %printed = call i32 (ptr, ...) @printf(ptr @.str, double %g0, double %g1)
  ret i32 0
}
