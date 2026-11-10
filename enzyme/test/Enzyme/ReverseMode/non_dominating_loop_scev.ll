; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-strict-aliasing=0 -passes="enzyme,verify" -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)

define void @f(ptr %p, i1 %c) {
entry:
  br i1 %c, label %outer, label %load.loop

load.loop:
  %i = phi i64 [ %i.next, %load.loop ], [ 0, %entry ]
  %i.next = add i64 %i, 1
  %i.gep = getelementptr double, ptr %p, i64 %i
  %i.val = load double, ptr %i.gep, align 8
  br i1 %c, label %outer, label %load.loop

outer:
  %j = phi i64 [ %j.next, %body ], [ 0, %load.loop ], [ 0, %entry ]
  %j.next = add i64 %j, 1
  br label %inner

inner:
  %k = phi i64 [ %k.next, %body ], [ %j, %outer ]
  %k.next = add i64 %k, 1
  %k.cond = icmp ult i64 %k, 100
  br i1 %k.cond, label %body, label %exit

body:
  %k.gep = getelementptr double, ptr %p, i64 %k
  store double 0.000000e+00, ptr %k.gep, align 8
  br i1 %c, label %outer, label %inner

exit:
  ret void
}

define double @f_active(ptr %p, i64 %n, i1 %c) {
entry:
  br i1 %c, label %outer, label %load.loop

load.loop:
  %i = phi i64 [ %i.next, %load.loop ], [ 0, %entry ]
  %sum = phi double [ %sum.next, %load.loop ], [ 0.000000e+00, %entry ]
  %i.next = add nuw nsw i64 %i, 1
  %i.gep = getelementptr double, ptr %p, i64 %i
  %i.val = load double, ptr %i.gep, align 8
  %sq = fmul double %i.val, %i.val
  %sum.next = fadd double %sum, %sq
  %ld.done = icmp eq i64 %i.next, %n
  br i1 %ld.done, label %outer, label %load.loop

outer:
  %res = phi double [ %res, %body ], [ %sum.next, %load.loop ], [ 0.000000e+00, %entry ]
  %j = phi i64 [ %j.next, %body ], [ 0, %load.loop ], [ 0, %entry ]
  %j.next = add i64 %j, 1
  br label %inner

inner:
  %k = phi i64 [ %k.next, %body ], [ %j, %outer ]
  %k.next = add i64 %k, 1
  %k.cond = icmp ult i64 %k, 100
  br i1 %k.cond, label %body, label %exit

body:
  %k.gep = getelementptr double, ptr %p, i64 %k
  store double 0.000000e+00, ptr %k.gep, align 8
  br i1 %c, label %outer, label %inner

exit:
  ret double %res
}

define double @caller(ptr %p, ptr %dp, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, ptr %p, ptr %dp, i1 false)
  %d = call double (...) @__enzyme_autodiff(ptr @f_active, ptr %p, ptr %dp, i64 %n, i1 false)
  ret double %d
}

; CHECK-LABEL: define internal void @diffef(
; CHECK-SAME: ptr {{.*}}%p, ptr {{.*}}%"p'", i1 %c)
; CHECK: invertbody:
; CHECK:   store double 0.000000e+00, ptr %"k.gep'ipg_unwrap", align 8

; CHECK-LABEL: define internal void @diffef_active(
; CHECK-SAME: ptr {{.*}}%p, ptr {{.*}}%"p'", i64 %n, i1 %c, double %differeturn)
; CHECK: invertload.loop:
; CHECK:   fmul fast double
