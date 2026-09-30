; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; The accesses of a checkpointed step through the object its first argument
; after the index points to, which a scheme that copies the state itself gets
; from set_paths. The step reads the pointer fields u (offset 0) and param
; (offset 8), reads and writes t (offset 24), writes through u and reads
; through param; it never touches unused (offset 16).
;
; Encoding: length of the path, its offsets, the offset of the access (-1: the
; whole object), flags (1 read, 2 write).
; CHECK: @enzyme.ckpt.paths.step = private constant [17 x i64] [
; CHECK-SAME: i64 0, i64 0, i64 1,
; CHECK-SAME: i64 0, i64 8, i64 1,
; CHECK-SAME: i64 0, i64 24, i64 3,
; CHECK-SAME: i64 1, i64 0, i64 -1, i64 3,
; CHECK-SAME: i64 1, i64 8, i64 -1, i64 1]
; CHECK: call ptr @__enzyme_ckpt_fwd({{.*}}, ptr @enzyme.ckpt.paths.step, i64 17)


@enzyme_scheme = external global i32, align 4
@enzyme_const = external global i32, align 4

define void @step(i64 %i, ptr %m) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %k.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %k.0, 4
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = load ptr, ptr %m, align 8
  %idxprom = zext i32 %k.0 to i64
  %arrayidx = getelementptr inbounds double, ptr %0, i64 %idxprom
  %1 = load double, ptr %arrayidx, align 8
  %param = getelementptr inbounds i8, ptr %m, i64 8
  %2 = load ptr, ptr %param, align 8
  %idxprom1 = zext i32 %k.0 to i64
  %arrayidx2 = getelementptr inbounds double, ptr %2, i64 %idxprom1
  %3 = load double, ptr %arrayidx2, align 8
  %t = getelementptr inbounds i8, ptr %m, i64 24
  %4 = load double, ptr %t, align 8
  %5 = call double @llvm.fmuladd.f64(double %1, double %3, double %4)
  %6 = load ptr, ptr %m, align 8
  %idxprom4 = zext i32 %k.0 to i64
  %arrayidx5 = getelementptr inbounds double, ptr %6, i64 %idxprom4
  store double %5, ptr %arrayidx5, align 8
  %inc = add nsw i32 %k.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %t6 = getelementptr inbounds i8, ptr %m, i64 24
  %7 = load double, ptr %t6, align 8
  %add = fadd double %7, 5.000000e-01
  store double %add, ptr %t6, align 8
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double)

define double @f(ptr %m, i64 %n, ptr %s, ptr %c) {
entry:
  %0 = load i32, ptr @enzyme_scheme, align 4
  call void (ptr, i64, i64, ...) @__enzyme_checkpoint_for(ptr @step, i64 0, i64 %n, i32 %0, ptr %s, ptr %c, ptr %m)
  %1 = load ptr, ptr %m, align 8
  %2 = load double, ptr %1, align 8
  ret double %2
}

declare void @__enzyme_checkpoint_for(ptr, i64, i64, ...)

define void @df(ptr %m, ptr %dm, i64 %n, ptr %s, ptr %c) {
entry:
  %0 = load i32, ptr @enzyme_const, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %m, ptr %dm, i64 %n, i32 %0, ptr %s, i32 %0, ptr %c)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

