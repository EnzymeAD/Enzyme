; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 12.000000

@.fmt = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.assume(i1 noundef)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare double @__enzyme_autodiff(...)

define internal double @read_view(ptr nocapture readonly %view) noinline {
entry:
  %data_ptr = load ptr, ptr %view, align 8
  %val = load double, ptr %data_ptr, align 8
  %len_gep = getelementptr inbounds i8, ptr %view, i64 8
  %len = load i64, ptr %len_gep, align 8
  %scale = sitofp i64 %len to double
  %sq = fmul double %val, %val
  %prod = fmul double %sq, %scale
  ret double %prod
}

define i32 @main() {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @eval_block_view, double 3.000000e+00, i64 2)
  %printed = call i32 (ptr, ...) @printf(ptr nonnull @.fmt, double %dx)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffeeval_block_view(
define double @eval_block_view(double %x, i64 %len) {
entry:
  %cell = alloca double, align 8
  %view = alloca [48 x i8], align 8
  store double %x, ptr %cell, align 8
  br label %body

body:
  call void @llvm.lifetime.start.p0(i64 48, ptr nonnull %view)
  %pos = icmp sge i64 %len, 0
  call void @llvm.assume(i1 %pos)
  store ptr %cell, ptr %view, align 8
  %len_slot = getelementptr inbounds i8, ptr %view, i64 8
  store i64 %len, ptr %len_slot, align 8
  %untyped_tail = getelementptr inbounds i8, ptr %view, i64 16
  call void @llvm.memset.p0.i64(ptr nonnull align 8 %untyped_tail, i8 0, i64 32, i1 false)
  ; CHECK: call void @differead_view(
  %res = call double @read_view(ptr nonnull %view)
  call void @llvm.lifetime.end.p0(i64 48, ptr nonnull %view)
  ; CHECK: %[[OUT:.+]] = load double, ptr %{{.+}}, align 8
  ; CHECK: %[[RET:.+]] = insertvalue { double } undef, double %[[OUT]], 0
  ; CHECK: ret { double } %[[RET]]
  ret double %res
}
