; RUN: if [ %llvmver -ge 16 ]; then printf "sq forward+c2\n" > %t.exports; %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-export-list=%t.exports -enzyme-global-activity -enzyme-globals-default-inactive -S | FileCheck %s; fi

; Separate compilation: an argument constant at a call through another
; module's derivative gets no shadow, as at a call within the module, rather
; than the primal standing in for its shadow (which the callee's derivative
; would write tangents into). So is a local object holding only constants,
; such as the descriptor flang builds to pass on a section of an inactive
; array, whose shadow would hold the primal pointers. The variant called is
; named after the arguments constant at the call (_c<hex mask>); the
; defining module exports it on request ("+c<hex mask>" in the export list).

@table = global [4 x double] zeroinitializer

define void @sq(ptr %x, ptr %t) {
  %v = load double, ptr %x
  %w = load double, ptr %t
  %m = fmul double %v, %w
  store double %m, ptr %x
  ret void
}

declare void @ext(ptr, ptr)
declare void @mpi_abort_(ptr, ptr, ptr)

define void @user(ptr %x, ptr %c, ptr %e) {
  call void @ext(ptr %x, ptr @table)
  %box = alloca { ptr, i64 }
  store ptr @table, ptr %box
  %len = getelementptr inbounds { ptr, i64 }, ptr %box, i32 0, i32 1
  store i64 4, ptr %len
  call void @ext(ptr %x, ptr %box)
  call void @mpi_abort_(ptr %c, ptr %e, ptr %e)
  ret void
}

declare void @__enzyme_fwddiff(...)

define void @caller(ptr %x, ptr %dx, ptr %c, ptr %e) {
  call void (...) @__enzyme_fwddiff(ptr @user, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %c, metadata !"enzyme_const", ptr %e)
  ret void
}

; CHECK-DAG: @__enzyme_sep_fwd_w1_c2_ext = external constant ptr
; CHECK-DAG: @__enzyme_sep_fwd_w1_c2_sq = constant ptr @fwddiffesq
; CHECK-NOT: @__enzyme_sep_{{.*}}mpi_abort_

; CHECK: define internal void @fwddiffeuser(ptr %x, ptr %"x'", ptr %c, ptr %e)
; CHECK: call void %{{[0-9]+}}(ptr %x, ptr %"x'", ptr @table)
; CHECK: call void %{{[0-9]+}}(ptr %x, ptr %"x'", ptr %box)
; CHECK: call void @mpi_abort_(ptr %c, ptr %e, ptr %e)

; CHECK: define internal void @fwddiffesq(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%t)
