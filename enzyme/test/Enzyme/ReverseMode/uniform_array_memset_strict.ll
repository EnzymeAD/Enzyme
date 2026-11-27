; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-loose-types=0 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-loose-types=0 -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-loose-types=0 -S | %lli - | FileCheck %s --check-prefix=EVAL

@lb = global i64 1, align 8
@ub = global i64 3, align 8
@row_idx = global i64 1, align 8
@col_write_limit = global i64 2, align 8
@col_read_limit = global i64 3, align 8

@work = global [4 x [4 x double]] zeroinitializer, align 16, !enzyme_shadow !0
@dwork = global [4 x [4 x double]] zeroinitializer, align 16

@.fmt = private unnamed_addr constant [42 x i8] c"local=%.1f global=%.1f copy=%.1f d0=%.1f\0A\00", align 1
@.fmt2 = private unnamed_addr constant [19 x i8] c"d12=%.1f d13=%.1f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare double @__enzyme_autodiff(ptr, ...)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define double @multi_phase_local(double %x) {
entry:
  %scratch = alloca [4 x [4 x double]], align 16
  call void @llvm.memset.p0.i64(ptr align 16 %scratch, i8 0, i64 128, i1 false)
  %start = load i64, ptr @lb, align 8
  %end = load i64, ptr @ub, align 8
  br label %phase1.write.cond

phase1.write.cond:
  %i = phi i64 [ %start, %entry ], [ %i.next, %phase1.write.body ]
  %cmp1 = icmp slt i64 %i, %end
  br i1 %cmp1, label %phase1.write.body, label %phase1.read.cond

phase1.write.body:
  %ip1 = add nsw i64 %i, 1
  %scale = sitofp i64 %ip1 to double
  %val1 = fmul double %x, %scale
  %wptr1 = getelementptr inbounds [4 x [4 x double]], ptr %scratch, i64 0, i64 %i, i64 %i
  store double %val1, ptr %wptr1, align 8
  %i.next = add nsw i64 %i, 1
  br label %phase1.write.cond

phase1.read.cond:
  %j = phi i64 [ %start, %phase1.write.cond ], [ %j.next, %phase1.read.body ]
  %sum1 = phi double [ 0.000000e+00, %phase1.write.cond ], [ %sum1.next, %phase1.read.body ]
  %cmp2 = icmp slt i64 %j, %end
  br i1 %cmp2, label %phase1.read.body, label %inter_phase_zero

phase1.read.body:
  %rptr1 = getelementptr inbounds [4 x [4 x double]], ptr %scratch, i64 0, i64 %j, i64 %j
  %rval1 = load double, ptr %rptr1, align 8
  %sum1.next = fadd double %sum1, %rval1
  %j.next = add nsw i64 %j, 1
  br label %phase1.read.cond

inter_phase_zero:
  %phase2_start = load i64, ptr @lb, align 8
  call void @llvm.memset.p0.i64(ptr align 16 %scratch, i8 0, i64 128, i1 false)
  %wptr2 = getelementptr inbounds [4 x [4 x double]], ptr %scratch, i64 0, i64 %phase2_start, i64 %phase2_start
  %val2 = fmul double %x, 7.000000e+00
  store double %val2, ptr %wptr2, align 8
  br label %phase2.read.cond

phase2.read.cond:
  %k = phi i64 [ %start, %inter_phase_zero ], [ %k.next, %phase2.read.body ]
  %sum2 = phi double [ 0.000000e+00, %inter_phase_zero ], [ %sum2.next, %phase2.read.body ]
  %cmp3 = icmp slt i64 %k, %end
  br i1 %cmp3, label %phase2.read.body, label %exit

phase2.read.body:
  %rptr2 = getelementptr inbounds [4 x [4 x double]], ptr %scratch, i64 0, i64 %k, i64 %k
  %rval2 = load double, ptr %rptr2, align 8
  %weighted2 = fmul double %rval2, 1.000000e+01
  %sum2.next = fadd double %sum2, %weighted2
  %k.next = add nsw i64 %k, 1
  br label %phase2.read.cond

exit:
  %total = fadd double %sum1, %sum2
  ret double %total
}

define internal void @scale_row_into_work(ptr %mat, i64 %row, i64 %limit, double %x) noinline {
entry:
  br label %loop.cond

loop.cond:
  %c = phi i64 [ 0, %entry ], [ %c.next, %loop.body ]
  %cmp = icmp slt i64 %c, %limit
  br i1 %cmp, label %loop.body, label %exit

loop.body:
  %cell = getelementptr inbounds [4 x [4 x double]], ptr %mat, i64 0, i64 %row, i64 %c
  %scaled = fmul double %x, 4.000000e+00
  store double %scaled, ptr %cell, align 8
  %c.next = add nsw i64 %c, 1
  br label %loop.cond

exit:
  ret void
}

define double @global_row_phase(double %x) {
entry:
  %row = load i64, ptr @row_idx, align 8
  %wlim = load i64, ptr @col_write_limit, align 8
  %rlim = load i64, ptr @col_read_limit, align 8
  %row_ptr = getelementptr inbounds [4 x [4 x double]], ptr @work, i64 0, i64 %row, i64 0
  call void @llvm.memset.p0.i64(ptr align 16 %row_ptr, i8 0, i64 32, i1 false)
  call void @scale_row_into_work(ptr @work, i64 %row, i64 %wlim, double %x)
  br label %read.cond

read.cond:
  %c = phi i64 [ 0, %entry ], [ %c.next, %read.body ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %read.body ]
  %cmp = icmp slt i64 %c, %rlim
  br i1 %cmp, label %read.body, label %exit

read.body:
  %cell = getelementptr inbounds [4 x [4 x double]], ptr @work, i64 0, i64 %row, i64 %c
  %val = load double, ptr %cell, align 8
  %twice = fmul double %val, 2.000000e+00
  %acc.next = fadd double %acc, %twice
  %c.next = add nsw i64 %c, 1
  br label %read.cond

exit:
  %w00 = load double, ptr @work, align 16
  %prod00 = fmul double %w00, %x
  %res = fadd double %acc, %prod00
  ret double %res
}

define double @copy_uniform_array(double %x) {
entry:
  %src = alloca [4 x double], align 16
  %dst = alloca [4 x double], align 16
  call void @llvm.memset.p0.i64(ptr align 16 %src, i8 0, i64 32, i1 false)
  %start = load i64, ptr @lb, align 8
  %end = load i64, ptr @ub, align 8
  br label %write.cond

write.cond:
  %i = phi i64 [ %start, %entry ], [ %i.next, %write.body ]
  %cmp1 = icmp slt i64 %i, %end
  br i1 %cmp1, label %write.body, label %do.copy

write.body:
  %sptr = getelementptr inbounds [4 x double], ptr %src, i64 0, i64 %i
  %val = fmul double %x, 5.000000e+00
  store double %val, ptr %sptr, align 8
  %i.next = add nsw i64 %i, 1
  br label %write.cond

do.copy:
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dst, ptr align 16 %src, i64 32, i1 false)
  br label %read.cond

read.cond:
  %k = phi i64 [ %start, %do.copy ], [ %k.next, %read.body ]
  %sum = phi double [ 0.000000e+00, %do.copy ], [ %sum.next, %read.body ]
  %cmp2 = icmp slt i64 %k, %end
  br i1 %cmp2, label %read.body, label %exit

read.body:
  %dptr = getelementptr inbounds [4 x double], ptr %dst, i64 0, i64 %k
  %dval = load double, ptr %dptr, align 8
  %sum.next = fadd double %sum, %dval
  %k.next = add nsw i64 %k, 1
  br label %read.cond

exit:
  ret double %sum
}

define i32 @main() {
entry:
  %grad_local = call double (ptr, ...) @__enzyme_autodiff(ptr @multi_phase_local, double 3.000000e+00)
  store double 3.000000e+00, ptr @work, align 16
  store double 5.000000e+00, ptr @dwork, align 16
  %d13_ptr = getelementptr inbounds [4 x [4 x double]], ptr @dwork, i64 0, i64 1, i64 3
  store double 9.900000e+01, ptr %d13_ptr, align 8
  %grad_global = call double (ptr, ...) @__enzyme_autodiff(ptr @global_row_phase, double 2.000000e+00)
  %grad_copy = call double (ptr, ...) @__enzyme_autodiff(ptr @copy_uniform_array, double 4.000000e+00)
  %d00 = load double, ptr @dwork, align 16
  %d12_ptr = getelementptr inbounds [4 x [4 x double]], ptr @dwork, i64 0, i64 1, i64 2
  %d12 = load double, ptr %d12_ptr, align 8
  %d13 = load double, ptr %d13_ptr, align 8
  %fmt_ptr = getelementptr inbounds [42 x i8], ptr @.fmt, i64 0, i64 0
  call i32 (ptr, ...) @printf(ptr %fmt_ptr, double %grad_local, double %grad_global, double %grad_copy, double %d00)
  %fmt2_ptr = getelementptr inbounds [19 x i8], ptr @.fmt2, i64 0, i64 0
  call i32 (ptr, ...) @printf(ptr %fmt2_ptr, double %d12, double %d13)
  ret i32 0
}

!0 = !{ptr @dwork}

; CHECK-LABEL: define internal { double } @diffemulti_phase_local(
; CHECK: invertinter_phase_zero:
; CHECK: call void @llvm.memset.p0.i64(ptr align 16 %"scratch'ipa", i8 0, i64 128, i1 false)

; CHECK-LABEL: define internal { double } @diffeglobal_row_phase(
; CHECK: invertentry:
; CHECK: call void @llvm.memset.p0.i64(ptr align 16 %"row_ptr'ipg{{.*}}", i8 0, i64 32, i1 false)

; CHECK-LABEL: define internal { double } @diffecopy_uniform_array(

; EVAL: local=75.0 global=19.0 copy=10.0 d0=7.0
; EVAL-NEXT: d12=0.0 d13=0.0
