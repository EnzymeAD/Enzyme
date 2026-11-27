; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)
declare void @panic_bounds_check() cold noreturn

define internal void @checked_write(ptr %buf, i64 %idx, i64 %len, double %val) noinline {
entry:
  %oob = icmp uge i64 %idx, %len
  br i1 %oob, label %panic, label %ok

panic:
  call void @panic_bounds_check()
  unreachable

ok:
  %p = getelementptr inbounds double, ptr %buf, i64 %idx
  store double %val, ptr %p, align 8
  ret void
}

define internal double @sqdist(ptr %arr, double %m) noinline {
entry:
  %data = load ptr, ptr %arr, align 8
  %np = getelementptr inbounds i8, ptr %arr, i64 8
  %n = load i64, ptr %np, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds double, ptr %data, i64 %i
  %y = load double, ptr %p, align 8
  %d = fsub double %y, %m
  %sq = fmul double %d, %d
  %acc.next = fadd double %acc, %sq
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %acc.next
}

define void @f(ptr noalias nocapture %out, ptr noalias %arr, ptr noalias nocapture %buf, i64 %idx, i64 %len, double %m) {
entry:
  %r = call double @sqdist(ptr %arr, double %m)
  call void @checked_write(ptr %buf, i64 %idx, i64 %len, double 2.0)
  store double %r, ptr %out, align 8
  ret void
}

define void @test(ptr %out, ptr %dout, ptr %arr, ptr %buf, i64 %idx, i64 %len, double %m) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", ptr %arr, metadata !"enzyme_const", ptr %buf, i64 %idx, i64 %len, double %m)
  ret void
}

define double @g(ptr noalias readonly %w, ptr noalias %buf, i64 %n, i64 %idx, i64 %len, double %x) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %acc.next, %loop ]
  %wp = getelementptr inbounds double, ptr %w, i64 %i
  %wv = load double, ptr %wp, align 8
  %prod = fmul double %wv, %x
  %acc.next = fadd double %acc, %prod
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %after, label %loop

after:
  call void @checked_write(ptr %buf, i64 %idx, i64 %len, double 3.0)
  ret double %acc.next
}

define double @test_g(ptr %w, ptr %buf, i64 %n, i64 %idx, i64 %len, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @g, metadata !"enzyme_const", ptr %w, metadata !"enzyme_const", ptr %buf, i64 %n, i64 %idx, i64 %len, double %x)
  ret double %dx
}

; @checked_write is deduced as enzyme_ArgMemOnlyOrThrow, and its writable argument
; %buf is noalias with %arr and %w, so neither @sqdist nor @g needs a tape allocation.
; CHECK: define internal void @checked_write({{.*}}) #[[CW:[0-9]+]]
; CHECK-DAG: define internal { double } @diffef(
; CHECK-DAG: define internal { double } @diffeg(
; CHECK-DAG: define internal double @augmented_sqdist(ptr {{.*}}readonly{{.*}} %arr, double %m)
; CHECK-DAG: define internal { double } @diffesqdist(ptr {{.*}}readonly{{.*}} %arr, double %m, double %differeturn)
; CHECK-NOT: call {{.*}}@malloc
; CHECK: attributes #[[CW]] = { {{.*}}"enzyme_ArgMemOnlyOrThrow"{{.*}} }
