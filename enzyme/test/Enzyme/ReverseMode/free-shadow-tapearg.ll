; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; The shadow of %p is the only value on the tape of @helper, so in
; @diffehelper it is the tape argument itself rather than an instruction.

declare void @__enzyme_autodiff(...)

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare void @keep(ptr) nofree "enzyme_inactive"

define internal void @helper(ptr %x) noinline {
entry:
  %p = call noalias ptr @malloc(i64 8)
  call void @keep(ptr %p)
  %v = load double, ptr %x, align 8
  %m = fmul double %v, 2.000000e+00
  store double %m, ptr %p, align 8
  %res = load double, ptr %p, align 8
  store double %res, ptr %x, align 8
  call void @free(ptr %p)
  ret void
}

define void @caller(ptr %x) {
entry:
  call void @helper(ptr %x)
  %v = load double, ptr %x, align 8
  %m = fmul double %v, %v
  store double %m, ptr %x, align 8
  ret void
}

define void @test_derivative(ptr %x, ptr %dx) {
entry:
  call void (...) @__enzyme_autodiff(ptr @caller, metadata !"enzyme_dup", ptr %x, ptr %dx)
  ret void
}

; CHECK: define internal ptr @augmented_helper(ptr {{.*}}%x, ptr {{.*}}%"x'")
; CHECK:   %"p'mi" = call noalias nonnull {{.*}}ptr @malloc(i64 8)
; CHECK:   ret ptr %"p'mi"

; CHECK: define internal void @diffehelper(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr %"p'mi")
; CHECK:   call void @free(ptr nonnull %"p'mi")
; CHECK-NEXT:   ret void
