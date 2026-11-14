; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -enzyme-global-activity=1 -S | FileCheck %s; fi

; Under global activity, a time query in a callee of the differentiated
; function must not be treated as a call that could return an allocation.

%struct.timeval = type { i64, i64 }
declare i32 @gettimeofday(ptr, ptr)

define void @cloc(ptr %curtim) noinline {
entry:
  %tv = alloca %struct.timeval, align 8
  %c = call i32 @gettimeofday(ptr %tv, ptr null)
  %s = load i64, ptr %tv, align 8
  %u.p = getelementptr inbounds %struct.timeval, ptr %tv, i64 0, i32 1
  %u = load i64, ptr %u.p, align 8
  %sf = sitofp i64 %s to double
  %uf = sitofp i64 %u to double
  %t = fadd double %uf, %sf
  store double %t, ptr %curtim, align 8
  ret void
}

define double @f(double %x) {
entry:
  %t = alloca double, align 8
  call void @cloc(ptr %t)
  %v = load double, ptr %t
  %m = fmul double %x, %v
  ret double %m
}

declare double @__enzyme_autodiff(...)
define double @test(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

; CHECK: define internal void @augmented_cloc(ptr {{.*}}%curtim, ptr {{.*}}%"curtim'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tv = alloca %struct.timeval, i64 1, align 8
; CHECK-NEXT:   %c = call i32 @gettimeofday(ptr %tv, ptr null)
; CHECK:   store double %t, ptr %curtim, align 8
; CHECK-NEXT:   ret void

; CHECK: define internal void @diffecloc(ptr {{.*}}%curtim, ptr {{.*}}%"curtim'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertentry
; CHECK: invertentry:
; CHECK-NEXT:   store double 0.000000e+00, ptr %"curtim'", align 8
; CHECK-NEXT:   ret void
