; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 6.000000 10.000000

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare double @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr nocapture readonly, ...)

@.str = private unnamed_addr constant [7 x i8] c"%f %f\0A\00", align 1

define internal double @eval_block(ptr nocapture readonly %block) noinline {
entry:
  %data = load ptr, ptr %block, align 8
  %count_ptr = getelementptr inbounds i8, ptr %block, i64 8
  %count = load i64, ptr %count_ptr, align 8
  %stride_ptr = getelementptr inbounds i8, ptr %block, i64 32
  %stride = load i64, ptr %stride_ptr, align 8
  %v0 = load double, ptr %data, align 8
  %p1 = getelementptr inbounds double, ptr %data, i64 %stride
  %v1 = load double, ptr %p1, align 8
  %s0 = fmul double %v0, %v0
  %s1 = fmul double %v1, %v1
  %has_more = icmp sgt i64 %count, 1
  %sum = fadd double %s0, %s1
  %res = select i1 %has_more, double %sum, double %s0
  ret double %res
}

; CHECK-LABEL: define internal void @diffewrap_block(
define double @wrap_block(ptr nocapture readonly %input, i64 %count, i1 %cond) {
entry:
  %zero_scratch = alloca [16 x i8], align 8
  %block = alloca [40 x i8], align 8
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %block)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %zero_scratch)
  br i1 %cond, label %left, label %right

left:
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %zero_scratch, i8 0, i64 16, i1 false)
  br label %join

right:
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %zero_scratch, i8 0, i64 16, i1 false)
  br label %join

join:
  store ptr %input, ptr %block, align 8
  %count_slot = getelementptr inbounds i8, ptr %block, i64 8
  store i64 %count, ptr %count_slot, align 8
  %unread_offsets = getelementptr inbounds i8, ptr %block, i64 16
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.+}}%"unread_offsets'ipg", ptr {{.+}}%zero_scratch, i64 16, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %unread_offsets, ptr noundef nonnull align 8 dereferenceable(16) %zero_scratch, i64 16, i1 false)
  %stride_slot = getelementptr inbounds i8, ptr %block, i64 32
  store i64 1, ptr %stride_slot, align 8
; CHECK: call void @diffeeval_block(ptr %block, ptr %"block'ipa", double %differeturn)
  %result = call double @eval_block(ptr nonnull %block)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %zero_scratch)
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %block)
  ret double %result
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
  %call = call double (ptr, ...) @__enzyme_autodiff(ptr @wrap_block, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i64 2, metadata !"enzyme_const", i1 true)
  %g0 = load double, ptr %dx, align 16
  %g1 = load double, ptr %dx1, align 8
  %printed = call i32 (ptr, ...) @printf(ptr @.str, double %g0, double %g1)
  ret i32 0
}
