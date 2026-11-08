; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme-loose-types=0 -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-loose-types=0 -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-loose-types=0 -passes="enzyme,default<O2>" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: din=5.000000,3.000000

@.fmt = private unnamed_addr constant [11 x i8] c"din=%f,%f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare double @__enzyme_autodiff(ptr, ...)

define double @slice_memset_product(ptr nocapture readonly %in) {
entry:
  %buf = alloca [2 x double], align 8
  %idx_slot = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %idx_slot)
  store i32 0, ptr %idx_slot, align 4
  %idx = load i32, ptr %idx_slot, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %idx_slot)

  %idx64 = sext i32 %idx to i64
  %byte_off = shl nsw i64 %idx64, 3
  %dst = getelementptr i8, ptr %buf, i64 %byte_off
  call void @llvm.memset.p0.i64(ptr align 8 %dst, i8 0, i64 16, i1 false)

  %a = load double, ptr %in, align 8
  %in1 = getelementptr inbounds double, ptr %in, i64 1
  %b = load double, ptr %in1, align 8
  %prod = fmul double %a, %b

  %buf1 = getelementptr inbounds double, ptr %buf, i64 1
  store double %prod, ptr %buf1, align 8

  %v1 = load double, ptr %buf1, align 8
  ret double %v1
}

define i32 @main() {
entry:
  %in = alloca [2 x double], align 8
  %din = alloca [2 x double], align 8
  store double 3.000000e+00, ptr %in, align 8
  %in1 = getelementptr inbounds double, ptr %in, i64 1
  store double 5.000000e+00, ptr %in1, align 8
  store double 0.000000e+00, ptr %din, align 8
  %din1 = getelementptr inbounds double, ptr %din, i64 1
  store double 0.000000e+00, ptr %din1, align 8

  %res = call double (ptr, ...) @__enzyme_autodiff(ptr @slice_memset_product, ptr %in, ptr %din)
  %g0 = load double, ptr %din, align 8
  %g1 = load double, ptr %din1, align 8
  %p = call i32 (ptr, ...) @printf(ptr @.fmt, double %g0, double %g1)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffeslice_memset_product(
; CHECK: call void @llvm.memset.p0.i64(
