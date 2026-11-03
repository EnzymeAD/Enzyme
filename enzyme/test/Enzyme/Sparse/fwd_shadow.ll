; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; The shadow of a todense pointer is built from the shadows of its active
; arguments; inactive arguments (here the index %i) are passed unchanged.

declare ptr @__enzyme_todense(...)

declare double @load(i64, i64, ptr)

declare void @store(double, i64, i64, ptr)

declare double @__enzyme_fwddiff(...)

define double @f(ptr %x, i64 %i) {
entry:
  %p = call ptr (...) @__enzyme_todense(ptr @load, ptr @store, i64 %i, ptr %x)
  %v = load double, ptr %p, align 8
  ret double %v
}

define double @test(ptr %x, ptr %dx, i64 %i) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i64 %i)
  ret double %r
}

; CHECK: define internal double @fwddiffef(ptr {{.*}}%x, ptr {{.*}}%"x'", i64 %i)
; CHECK:   call double @load(i64 0, i64 %i, ptr %"x'")
