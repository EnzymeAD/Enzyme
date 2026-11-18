; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s; fi

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque

@ompi_mpi_double = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@one = constant i32 1
@f_dt = constant i32 1275070475
@f_comm = constant i32 1140850688

define void @mpi_alltoall_fwd(ptr %send_buf, ptr %recv_buf, ptr %sizes, ptr %subsizes, ptr %starts) {
entry:
  %subdt = alloca ptr, align 8
  %r0 = call i32 @MPI_Type_create_subarray(i32 2, ptr %sizes, ptr %subsizes, ptr %starts, i32 56, ptr @ompi_mpi_double, ptr %subdt)
  %r1 = call i32 @MPI_Type_commit(ptr %subdt)
  %dt = load ptr, ptr %subdt, align 8
  %r2 = call i32 @MPI_Alltoall(ptr %send_buf, i32 2, ptr %dt, ptr %recv_buf, i32 2, ptr %dt, ptr @ompi_mpi_comm_world)
  %r3 = call i32 @MPI_Type_free(ptr %subdt)
  ret void
}

define void @mpi_alltoallv_fwd(ptr %send_buf, ptr %sc, ptr %sd, ptr %recv_buf, ptr %rc, ptr %rd) {
entry:
  %call = call i32 @MPI_Alltoallv(ptr %send_buf, ptr %sc, ptr %sd, ptr @ompi_mpi_double, ptr %recv_buf, ptr %rc, ptr %rd, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
  ret void
}

define void @mpi_alltoallw_fwd(ptr %send_buf, ptr %sc, ptr %sd, ptr %st, ptr %recv_buf, ptr %rc, ptr %rd, ptr %rt) {
entry:
  %call = call i32 @MPI_Alltoallw(ptr %send_buf, ptr %sc, ptr %sd, ptr %st, ptr %recv_buf, ptr %rc, ptr %rd, ptr %rt, ptr @ompi_mpi_comm_world)
  ret void
}

define void @mpi_alltoall_fortran_fwd(ptr %send_buf, ptr %recv_buf) {
entry:
  %ierr = alloca i32, align 4
  call void @mpi_alltoall_(ptr %send_buf, ptr @one, ptr @f_dt, ptr %recv_buf, ptr @one, ptr @f_dt, ptr @f_comm, ptr %ierr)
  ret void
}

define void @mpi_alltoallv_fortran_fwd(ptr %send_buf, ptr %sc, ptr %sd, ptr %recv_buf, ptr %rc, ptr %rd) {
entry:
  %ierr = alloca i32, align 4
  call void @mpi_alltoallv_(ptr %send_buf, ptr %sc, ptr %sd, ptr @f_dt, ptr %recv_buf, ptr %rc, ptr %rd, ptr @f_dt, ptr @f_comm, ptr %ierr)
  ret void
}

define void @mpi_alltoallw_fortran_fwd(ptr %send_buf, ptr %sc, ptr %sd, ptr %st, ptr %recv_buf, ptr %rc, ptr %rd, ptr %rt) {
entry:
  %ierr = alloca i32, align 4
  call void @mpi_alltoallw_(ptr %send_buf, ptr %sc, ptr %sd, ptr %st, ptr %recv_buf, ptr %rc, ptr %rd, ptr %rt, ptr @f_comm, ptr %ierr)
  ret void
}

declare i32 @MPI_Type_create_subarray(i32, ptr, ptr, ptr, i32, ptr, ptr)
declare i32 @MPI_Type_commit(ptr)
declare i32 @MPI_Type_free(ptr)
declare i32 @MPI_Alltoall(ptr, i32, ptr, ptr, i32, ptr, ptr)
declare i32 @MPI_Alltoallv(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare i32 @MPI_Alltoallw(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_alltoall_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_alltoallv_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_alltoallw_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

define void @caller(ptr %s, ptr %ds, ptr %sc, ptr %sd, ptr %st, ptr %r, ptr %dr, ptr %rc, ptr %rd, ptr %rt, ptr %sz, ptr %ssz, ptr %sta) {
entry:
  call void (ptr, ...) @__enzyme_fwddiff(ptr @mpi_alltoall_fwd, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %sz, metadata !"enzyme_const", ptr %ssz, metadata !"enzyme_const", ptr %sta)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @mpi_alltoallv_fwd, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @mpi_alltoallw_fwd, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_const", ptr %st, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd, metadata !"enzyme_const", ptr %rt)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @mpi_alltoall_fortran_fwd, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dup", ptr %r, ptr %dr)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @mpi_alltoallv_fortran_fwd, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd)
  call void (ptr, ...) @__enzyme_fwddiff(ptr @mpi_alltoallw_fortran_fwd, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_const", ptr %st, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd, metadata !"enzyme_const", ptr %rt)
  ret void
}

declare void @__enzyme_fwddiff(ptr, ...)

; CHECK-LABEL: define internal void @fwddiffempi_alltoall_fwd(
; CHECK:   call i32 @MPI_Type_create_subarray
; CHECK:   call i32 @MPI_Type_commit
; CHECK:   call i32 @MPI_Alltoall(ptr %send_buf, i32 2, ptr %dt, ptr %recv_buf, i32 2, ptr %dt, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Alltoall(ptr %"send_buf'", i32 2, ptr %dt, ptr %"recv_buf'", i32 2, ptr %dt, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Type_free(ptr %subdt)

; CHECK-LABEL: define internal void @fwddiffempi_alltoallv_fwd(
; CHECK:   call i32 @MPI_Alltoallv(ptr %send_buf, ptr %sc, ptr %sd, ptr @ompi_mpi_double, ptr %recv_buf, ptr %rc, ptr %rd, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Alltoallv(ptr %"send_buf'", ptr %sc, ptr %sd, ptr @ompi_mpi_double, ptr %"recv_buf'", ptr %rc, ptr %rd, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)

; CHECK-LABEL: define internal void @fwddiffempi_alltoallw_fwd(
; CHECK:   call i32 @MPI_Alltoallw(ptr %send_buf, ptr %sc, ptr %sd, ptr %st, ptr %recv_buf, ptr %rc, ptr %rd, ptr %rt, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Alltoallw(ptr %"send_buf'", ptr %sc, ptr %sd, ptr %st, ptr %"recv_buf'", ptr %rc, ptr %rd, ptr %rt, ptr @ompi_mpi_comm_world)

; CHECK-LABEL: define internal void @fwddiffempi_alltoall_fortran_fwd(
; CHECK:   call void @mpi_alltoall_(ptr %send_buf, ptr @one, ptr @f_dt, ptr %recv_buf, ptr @one, ptr @f_dt, ptr @f_comm, ptr %ierr)
; CHECK:   call void @mpi_alltoall_(ptr %"send_buf'", ptr @one, ptr @f_dt, ptr %"recv_buf'", ptr @one, ptr @f_dt, ptr @f_comm, ptr %ierr)

; CHECK-LABEL: define internal void @fwddiffempi_alltoallv_fortran_fwd(
; CHECK:   call void @mpi_alltoallv_(ptr %send_buf, ptr %sc, ptr %sd, ptr @f_dt, ptr %recv_buf, ptr %rc, ptr %rd, ptr @f_dt, ptr @f_comm, ptr %ierr)
; CHECK:   call void @mpi_alltoallv_(ptr %"send_buf'", ptr %sc, ptr %sd, ptr @f_dt, ptr %"recv_buf'", ptr %rc, ptr %rd, ptr @f_dt, ptr @f_comm, ptr %ierr)

; CHECK-LABEL: define internal void @fwddiffempi_alltoallw_fortran_fwd(
; CHECK:   call void @mpi_alltoallw_(ptr %send_buf, ptr %sc, ptr %sd, ptr %st, ptr %recv_buf, ptr %rc, ptr %rd, ptr %rt, ptr @f_comm, ptr %ierr)
; CHECK:   call void @mpi_alltoallw_(ptr %"send_buf'", ptr %sc, ptr %sd, ptr %st, ptr %"recv_buf'", ptr %rc, ptr %rd, ptr %rt, ptr @f_comm, ptr %ierr)
