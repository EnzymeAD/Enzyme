; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; %a is filled in by an opaque call (e.g. a Fortran READ into a local), which
; cannot be replayed in the reverse pass, and then loaded. The loaded value
; only reaches the reverse pass through a store into the heap allocation %b,
; so %a must not be rematerialized: the load would read memory the reverse
; pass never wrote. Enzyme previously marked %a as rematerializable and then
; had to keep a heap copy of it per iteration for the reverse pass. Now %a
; stays a stack slot and only the loaded value is cached.

declare void @fill(ptr nocapture) nofree "enzyme_inactive"
declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare void @__enzyme_autodiff(...)

define double @f(double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %accn, %loop ]
  %a = alloca double, align 8
  %b = call noalias ptr @malloc(i64 8)
  call void @fill(ptr %a)
  %v = load double, ptr %a, align 8
  store double %v, ptr %b, align 8
  %w = load double, ptr %b, align 8
  call void @free(ptr %b)
  %m = fmul double %x, %w
  %m2 = fmul double %m, %m
  %accn = fadd double %acc, %m2
  %inext = add i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %accn
}

define void @df(double %x, i64 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, double %x, i64 %n)
  ret void
}

; CHECK: define internal { double } @diffef(double %x, i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[a:.+]] = alloca double, i64 1, align 8
; CHECK-NOT: enzyme_fromstack
; CHECK: loop:
; CHECK:   call void @fill(ptr %[[a]])
; CHECK-NEXT:   %v = load double, ptr %[[a]], align 8
; CHECK-NOT: store ptr %[[a]]
; CHECK: invertloop:
; CHECK-NOT: call void @free(
; CHECK: }
