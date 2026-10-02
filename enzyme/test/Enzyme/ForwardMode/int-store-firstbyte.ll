; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Forward mode through a constant store of an i32 whose destination is known
; to hold an integer only at its first byte (as a parameter type may say).
; The store is an integer throughout: its shadow gets the primal value, and
; none of its bytes are zeroed as if they could be floats.

define void @f(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Integer}" %p, ptr %x, ptr %y, ptr %q, i64 %i) {
entry:
  %gep = getelementptr i32, ptr %q, i64 %i
  %v = load i32, ptr %gep, align 4
  store i32 %v, ptr %p, align 4
  %d = load double, ptr %x, align 8
  %m = fmul double %d, %d
  store double %m, ptr %y, align 8
  ret void
}

declare void @__enzyme_fwddiff(...)

define void @caller(ptr %p, ptr %dp, ptr %x, ptr %dx, ptr %y, ptr %dy, ptr %q, i64 %i) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_const", ptr %q, metadata !"enzyme_const", i64 %i)
  ret void
}

; CHECK: define internal void @fwddiffef(ptr {{.*}}%p, ptr {{.*}}%"p'", ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%y, ptr {{.*}}%"y'", ptr {{.*}}%q, i64 %i)
; CHECK-NOT: store i8 0
; CHECK:   store i32 %v, ptr %"p'"
; CHECK-NEXT:   store i32 %v, ptr %p
; CHECK:   store double %{{.*}}, ptr %"y'"
