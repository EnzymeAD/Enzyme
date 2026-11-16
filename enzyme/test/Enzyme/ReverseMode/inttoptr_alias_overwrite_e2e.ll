; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce),verify" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce),verify" -enzyme-preopt=false -S | %lli -; fi

; Regression test for areBaseObjectsGuaranteedNoAlias when one base object is a
; non-pointer integer (e.g., from ptrtoint -> or -> inttoptr) and the other is a
; noalias allocation. Previously the `&&` check instead of `||` fell through and
; falsely proved %p2 = inttoptr(or(ptrtoint %p, 8)) noalias with %p8 = gep %p, 8,
; causing overwritesToMemoryReadBy to miss the loop overwrite of %v and compute
; the wrong reverse-mode derivative (32.0 instead of 20.0).

declare ptr @malloc(i64)
declare void @free(ptr)
declare double @__enzyme_autodiff(...)

define double @f(double %x, i64 %n) {
entry:
  %p = call ptr @malloc(i64 16)
  %p8 = getelementptr inbounds i8, ptr %p, i64 8
  %pi = ptrtoint ptr %p to i64
  %pi8 = or i64 %pi, 8
  %p2 = inttoptr i64 %pi8 to ptr
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %acc.next, %loop ]
  %xi = sitofp i64 %iv to double
  %val = fadd double %x, %xi
  store double %val, ptr %p8, align 8
  %v = load double, ptr %p2, align 8
  %prod = fmul double %v, %v
  %acc.next = fadd double %acc, %prod
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  call void @free(ptr %p)
  ret double %acc.next
}

define double @df(double %x, i64 %n) {
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x, i64 %n)
  ret double %r
}

define i32 @main() {
  %g = call double @df(double 1.0, i64 4)
  ; sum_{i=0}^3 (x+i)^2 -> d/dx = 2 * (1 + 2 + 3 + 4) = 20.0
  %ok = fcmp oeq double %g, 20.0
  %res = select i1 %ok, i32 0, i32 1
  ret i32 %res
}

; CHECK: define internal { double } @diffef(double %x, i64 %n, double %differeturn)
; CHECK: %[[CACHE:[a-zA-Z0-9_.]+]] = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK: loop:
; CHECK: store double %val, ptr %p8, align 8
; CHECK: %v = load double, ptr %p2, align 8
; CHECK: store double %v, ptr
; CHECK: invertloop:
; CHECK-NOT: load double, ptr %p2
; CHECK: load double, ptr %{{[0-9]+}}, align 8
