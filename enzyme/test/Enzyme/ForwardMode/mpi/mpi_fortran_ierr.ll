; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s; fi

; The Fortran MPI ABI passes all arguments by reference and appends an `ierr`
; argument; the shadow calls take the same arguments as the original call,
; with the shadow in place of the buffer.

@one = constant i32 1
@zero = constant i32 0
@dt = constant i32 1275070475
@comm = constant i32 1140850688

define double @bcast(double %b) {
entry:
  %ierr = alloca i32, align 4
  %b.addr = alloca double, align 8
  store double %b, ptr %b.addr, align 8
  call void @mpi_bcast_(ptr %b.addr, ptr @one, ptr @dt, ptr @zero, ptr @comm, ptr %ierr)
  %r = load double, ptr %b.addr, align 8
  ret double %r
}

define double @send(double %b) {
entry:
  %ierr = alloca i32, align 4
  %b.addr = alloca double, align 8
  store double %b, ptr %b.addr, align 8
  call void @mpi_send_(ptr %b.addr, ptr @one, ptr @dt, ptr @zero, ptr @zero, ptr @comm, ptr %ierr)
  %r = load double, ptr %b.addr, align 8
  ret double %r
}

define double @recv(double %b) {
entry:
  %ierr = alloca i32, align 4
  %status = alloca [6 x i32], align 4
  %b.addr = alloca double, align 8
  store double %b, ptr %b.addr, align 8
  call void @mpi_recv_(ptr %b.addr, ptr @one, ptr @dt, ptr @zero, ptr @zero, ptr @comm, ptr %status, ptr %ierr)
  %r = load double, ptr %b.addr, align 8
  ret double %r
}

declare void @mpi_bcast_(ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_send_(ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_recv_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

define void @caller(double %x) {
entry:
  %0 = call double (...) @__enzyme_fwddiff(ptr @bcast, double %x, double 1.0)
  %1 = call double (...) @__enzyme_fwddiff(ptr @send, double %x, double 1.0)
  %2 = call double (...) @__enzyme_fwddiff(ptr @recv, double %x, double 1.0)
  ret void
}

declare double @__enzyme_fwddiff(...)

; CHECK-LABEL: define internal double @fwddiffebcast(
; CHECK: call void @mpi_bcast_(ptr %b.addr, ptr @one, ptr @dt, ptr @zero, ptr @comm, ptr %ierr)
; CHECK: call void @mpi_bcast_(ptr %"b.addr'ipa", ptr @one, ptr @dt, ptr @zero, ptr @comm, ptr %ierr)

; CHECK-LABEL: define internal double @fwddiffesend(
; CHECK: call void @mpi_send_(ptr %b.addr, ptr @one, ptr @dt, ptr @zero, ptr @zero, ptr @comm, ptr %ierr)
; CHECK: call void @mpi_send_(ptr %"b.addr'ipa", ptr @one, ptr @dt, ptr @zero, ptr @zero, ptr @comm, ptr %ierr)

; CHECK-LABEL: define internal double @fwddifferecv(
; CHECK: call void @mpi_recv_(ptr %b.addr, ptr @one, ptr @dt, ptr @zero, ptr @zero, ptr @comm, ptr %status, ptr %ierr)
; CHECK: call void @mpi_recv_(ptr %"b.addr'ipa", ptr @one, ptr @dt, ptr @zero, ptr @zero, ptr @comm, ptr %status, ptr %ierr)
