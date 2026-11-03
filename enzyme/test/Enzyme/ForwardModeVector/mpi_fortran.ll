; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s; fi

; In vector forward mode, the derivative of an MPI reduction (of a sum) or
; broadcast is the same call on the shadow buffers of every lane. Here with
; the Fortran ABI (arguments by reference, trailing `ierr`).

@one = constant i32 1
@dt = constant i32 1275070475
@sum = constant i32 1476395011
@zero = constant i32 0
@comm = constant i32 1140850688

define void @allreduce(ptr %x, ptr %y) {
entry:
  %ierr = alloca i32, align 4
  call void @mpi_allreduce_(ptr %x, ptr %y, ptr @one, ptr @dt, ptr @sum, ptr @comm, ptr %ierr)
  ret void
}

define void @bcast(ptr %x) {
entry:
  %ierr = alloca i32, align 4
  call void @mpi_bcast_(ptr %x, ptr @one, ptr @dt, ptr @zero, ptr @comm, ptr %ierr)
  ret void
}

declare void @mpi_allreduce_(ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_bcast_(ptr, ptr, ptr, ptr, ptr, ptr)

define void @caller(ptr %x, ptr %dx1, ptr %dx2, ptr %y, ptr %dy1, ptr %dy2) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @allreduce, metadata !"enzyme_width", i64 2, ptr %x, ptr %dx1, ptr %dx2, ptr %y, ptr %dy1, ptr %dy2)
  call void (...) @__enzyme_fwddiff(ptr @bcast, metadata !"enzyme_width", i64 2, ptr %x, ptr %dx1, ptr %dx2)
  ret void
}

declare void @__enzyme_fwddiff(...)

; CHECK-LABEL: define internal void @fwddiffe2allreduce(ptr %x, [2 x ptr] %"x'", ptr %y, [2 x ptr] %"y'")
; CHECK: call void @mpi_allreduce_(ptr %x, ptr %y, ptr @one, ptr @dt, ptr @sum, ptr @comm, ptr %ierr)
; CHECK-DAG: %[[x0:.+]] = extractvalue [2 x ptr] %"x'", 0
; CHECK-DAG: %[[y0:.+]] = extractvalue [2 x ptr] %"y'", 0
; CHECK-DAG: %[[x1:.+]] = extractvalue [2 x ptr] %"x'", 1
; CHECK-DAG: %[[y1:.+]] = extractvalue [2 x ptr] %"y'", 1
; CHECK-DAG: call void @mpi_allreduce_(ptr %[[x0]], ptr %[[y0]], ptr @one, ptr @dt, ptr @sum, ptr @comm, ptr %ierr)
; CHECK-DAG: call void @mpi_allreduce_(ptr %[[x1]], ptr %[[y1]], ptr @one, ptr @dt, ptr @sum, ptr @comm, ptr %ierr)

; CHECK-LABEL: define internal void @fwddiffe2bcast(ptr %x, [2 x ptr] %"x'")
; CHECK: call void @mpi_bcast_(ptr %x, ptr @one, ptr @dt, ptr @zero, ptr @comm, ptr %ierr)
; CHECK-DAG: %[[b0:.+]] = extractvalue [2 x ptr] %"x'", 0
; CHECK-DAG: %[[b1:.+]] = extractvalue [2 x ptr] %"x'", 1
; CHECK-DAG: call void @mpi_bcast_(ptr %[[b0]], ptr @one, ptr @dt, ptr @zero, ptr @comm, ptr %ierr)
; CHECK-DAG: call void @mpi_bcast_(ptr %[[b1]], ptr @one, ptr @dt, ptr @zero, ptr @comm, ptr %ierr)
