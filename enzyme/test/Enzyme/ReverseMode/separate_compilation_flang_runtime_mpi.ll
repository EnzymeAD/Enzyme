; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-global-activity -S | FileCheck %s; fi

; With -enzyme-global-activity, an inactive call to a function without a body
; still needs its derivative unless the function is known to leave no
; allocation the reverse pass needs. That holds for the flang runtime's
; character results (TRIM) and for MPI_Abort; and MPI routines, in any
; calling convention, are not called through another module's derivative
; under separate compilation.

declare void @_FortranATrim(ptr nocapture, ptr nocapture, ptr, i32)
declare void @mpi_abort_(ptr nocapture, ptr nocapture, ptr nocapture)

define void @f(ptr %x, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" %res, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer}" %s, ptr %c, ptr %e) {
  %v = load double, ptr %x
  %m = fmul double %v, %v
  store double %m, ptr %x
  call void @_FortranATrim(ptr %res, ptr %s, ptr null, i32 0)
  call void @mpi_abort_(ptr %c, ptr %e, ptr %e)
  ret void
}

declare void @__enzyme_autodiff(...)

define void @caller(ptr %x, ptr %dx, ptr %res, ptr %s, ptr %c, ptr %e) {
  call void (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %res, metadata !"enzyme_const", ptr %s, metadata !"enzyme_const", ptr %c, metadata !"enzyme_const", ptr %e)
  ret void
}

; CHECK-NOT: @__enzyme_sep_{{.*}}mpi_abort_
; CHECK: define internal void @diffef(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%res, ptr {{.*}}%s, ptr {{.*}}%c, ptr {{.*}}%e)
; CHECK: call void @_FortranATrim(ptr %res, ptr %s, ptr null, i32 0)
; CHECK: call void @mpi_abort_(ptr %c, ptr %e, ptr %e)
