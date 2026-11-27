; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,default<O2>,verify" -S | %lli -

; Regression test for nested dynamic loop caches grown via Enzyme's exponential
; allocator: when differentiating a function that already contains a nested
; dynamic loop cache (e.g. second-order reverse mode), the inner cache pointer
; is loaded from an outer heap array (or passed as an argument if un-inlined),
; so OldAllocationSize cannot trace its prior size statically. Tagging the
; exponential realloc call lets ReplaceReallocs lower it directly using
; (ptr == null ? 0 : req >> 1).

declare ptr @realloc(ptr, i64)
declare double @__enzyme_autodiff(ptr, ...)

define double @grow_arg(ptr %p, i64 %next) noinline {
entry:
  %q = call ptr @realloc(ptr %p, i64 %next) #0
  %v = load double, ptr %q, align 8
  %sq = fmul double %v, %v
  ret double %sq
}

define void @dgrow_arg(ptr %p, ptr %dp, i64 %next) {
entry:
  call double (ptr, ...) @__enzyme_autodiff(ptr @grow_arg, ptr %p, ptr %dp, i64 %next)
  ret void
}

define double @nested(double %x) noinline {
entry:
  %factor = fadd double %x, 5.000000e-01
  br label %outer

outer:
  %k = phi i64 [ 0, %entry ], [ %k.next, %outer.latch ]
  %total = phi double [ 0.000000e+00, %entry ], [ %total.next, %outer.latch ]
  %kf = sitofp i64 %k to double
  %koff = fmul double %kf, 2.500000e-01
  %s0 = fadd double %x, %koff
  br label %inner

inner:
  %s = phi double [ %s0, %outer ], [ %s.next, %inner ]
  %s.next = fmul double %s, %factor
  %cont = fcmp olt double %s.next, 6.000000e+00
  br i1 %cont, label %inner, label %outer.latch

outer.latch:
  %total.next = fadd double %total, %s.next
  %k.next = add nuw nsw i64 %k, 1
  %outer.done = icmp eq i64 %k.next, 3
  br i1 %outer.done, label %exit, label %outer

exit:
  ret double %total.next
}

define double @dnested(double %x) noinline {
entry:
  %dx = call double (ptr, ...) @__enzyme_autodiff(ptr @nested, double %x)
  ret double %dx
}

define double @d2nested(double %x) {
entry:
  %d2x = call double (ptr, ...) @__enzyme_autodiff(ptr @dnested, double %x)
  ret double %d2x
}

define i32 @main() {
entry:
  ; At x = 1.0:
  ;   k=0: 5 inner iters (realloc at size=1, 3, 5): x*(x+0.5)^5
  ;   k=1: 4 inner iters (realloc at size=1, 3):    (x+0.25)*(x+0.5)^4
  ;   k=2: 4 inner iters (realloc at size=1, 3):    (x+0.5)^5
  ;   df(1.0)  = 80.15625
  ;   d2f(1.0) = 246.375
  %d1 = call double @dnested(double 1.000000e+00)
  %err1 = fsub double %d1, 8.015625e+01
  %err1_sq = fmul double %err1, %err1
  %bad1 = fcmp ugt double %err1_sq, 1.000000e-20
  br i1 %bad1, label %fail, label %check2

check2:
  %d2 = call double @d2nested(double 1.000000e+00)
  %err2 = fsub double %d2, 2.463750e+02
  %err2_sq = fmul double %err2, %err2
  %bad2 = fcmp ugt double %err2_sq, 1.000000e-20
  br i1 %bad2, label %fail, label %ok

fail:
  ret i32 1

ok:
  ret i32 0
}

attributes #0 = { "enzyme_exponential_realloc" }

; CHECK-LABEL: define internal void @diffegrow_arg(
; CHECK-NOT: @realloc(
; CHECK: call {{.*}}ptr @malloc(i64 %next)
; CHECK: %[[ISNULL:.+]] = icmp eq ptr %p, null
; CHECK: br i1 %[[ISNULL]], label %{{.+}}, label %expcopy
; CHECK: expcopy{{.*}}:
; CHECK: %[[HALF:.+]] = lshr i64 %next, 1
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr %{{.+}}, ptr %p, i64 %[[HALF]], i1 false)
; CHECK: ret void

; CHECK-LABEL: define internal { double } @diffediffenested(
; CHECK: expcopy{{.*}}:
; CHECK: lshr i64 %{{.+}}, 1
; CHECK: call void @llvm.memcpy.p0.p0.i64(
; CHECK: invertexpcopy{{.*}}:
; CHECK: }
