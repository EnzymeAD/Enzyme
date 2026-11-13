; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque

@ompi_mpi_double = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@f_dt = constant i32 1275070475
@f_comm = constant i32 1140850688

define void @mpi_alltoallv_test(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %recv_buf, ptr %recvcounts, ptr %rdispls) {
entry:
  %call = call i32 @MPI_Alltoallv(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr @ompi_mpi_double, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
  ret void
}

define void @mpi_alltoallv_fortran_test(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %recv_buf, ptr %recvcounts, ptr %rdispls) {
entry:
  %ierr = alloca i32, align 4
  %v = load double, ptr %send_buf, align 8
  %v2 = fadd double %v, 1.0
  store double %v2, ptr %send_buf, align 8
  call void @mpi_alltoallv_(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr @f_dt, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr @f_dt, ptr @f_comm, ptr %ierr)
  ret void
}

define void @mpi_alltoallw_test(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %sendtypes, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr %recvtypes) {
entry:
  %v = load double, ptr %send_buf, align 8
  %v2 = fadd double %v, 1.0
  store double %v2, ptr %send_buf, align 8
  %call = call i32 @MPI_Alltoallw(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %sendtypes, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr %recvtypes, ptr @ompi_mpi_comm_world)
  ret void
}

define void @mpi_alltoallw_fortran_test(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %sendtypes, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr %recvtypes) {
entry:
  %ierr = alloca i32, align 4
  %v = load double, ptr %send_buf, align 8
  %v2 = fadd double %v, 1.0
  store double %v2, ptr %send_buf, align 8
  call void @mpi_alltoallw_(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %sendtypes, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr %recvtypes, ptr @f_comm, ptr %ierr)
  ret void
}

declare i32 @MPI_Alltoallv(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_alltoallv_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare i32 @MPI_Alltoallw(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_alltoallw_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

define void @caller(ptr %s, ptr %ds, ptr %sc, ptr %sd, ptr %st, ptr %r, ptr %dr, ptr %rc, ptr %rd, ptr %rt) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @mpi_alltoallv_test, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd)
  call void (ptr, ...) @__enzyme_autodiff(ptr @mpi_alltoallv_fortran_test, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd)
  call void (ptr, ...) @__enzyme_autodiff(ptr @mpi_alltoallw_test, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_const", ptr %st, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd, metadata !"enzyme_const", ptr %rt)
  call void (ptr, ...) @__enzyme_autodiff(ptr @mpi_alltoallw_fortran_test, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", ptr %sc, metadata !"enzyme_const", ptr %sd, metadata !"enzyme_const", ptr %st, metadata !"enzyme_dup", ptr %r, ptr %dr, metadata !"enzyme_const", ptr %rc, metadata !"enzyme_const", ptr %rd, metadata !"enzyme_const", ptr %rt)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; CHECK-LABEL: define internal void @diffempi_alltoallv_test(
; CHECK:   call i32 @MPI_Alltoallv(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr @ompi_mpi_double, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Comm_size(ptr @ompi_mpi_comm_world, ptr %{{.*}})
; CHECK:   %[[VDISPLS:.*]] = tail call noalias nonnull ptr @malloc(i64 %{{.*}})
; CHECK:   br label %invertentry_count_hdr
; CHECK: invertentry_count_exit:
; CHECK:   %[[VBUF:.*]] = tail call noalias nonnull ptr @malloc(i64 %{{.*}})
; CHECK:   call i32 @MPI_Alltoallv(ptr %"recv_buf'", ptr %recvcounts, ptr %rdispls, ptr @ompi_mpi_double, ptr %[[VBUF]], ptr %sendcounts, ptr %[[VDISPLS]], ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
; CHECK:   br label %invertentry_zero_hdr
; CHECK: invertentry_zero_body:
; CHECK:   call void @llvm.memset.p0.i64(ptr %{{.*}}, i8 0, i64 %{{.*}}, i1 false)
; CHECK: invertentry_acc_body:
; CHECK:   fadd fast double
; CHECK: invertentry_acc_exit:
; CHECK:   tail call void @free(ptr nonnull %[[VBUF]])
; CHECK:   tail call void @free(ptr nonnull %[[VDISPLS]])

; CHECK-LABEL: define internal void @diffempi_alltoallv_fortran_test(
; CHECK:   call void @mpi_alltoallv_(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr @f_dt, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr @f_dt, ptr @f_comm, ptr %ierr)
; CHECK:   call void @mpi_comm_size_(ptr @f_comm, ptr %{{.*}}, ptr %{{.*}})
; CHECK:   call void @mpi_type_size_(ptr @f_dt, ptr %{{.*}}, ptr %{{.*}})
; CHECK:   call void @mpi_alltoallv_(ptr %"recv_buf'", ptr %recvcounts, ptr %rdispls, ptr @f_dt, ptr %{{.*}}, ptr %sendcounts, ptr %{{.*}}, ptr @f_dt, ptr @f_comm, ptr %enzyme_mpi_ierr)

; CHECK-LABEL: define internal void @diffempi_alltoallw_test(
; CHECK:   call i32 @MPI_Alltoallw(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %sendtypes, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr %recvtypes, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Comm_size(ptr @ompi_mpi_comm_world, ptr %{{.*}})
; CHECK:   %[[WDISPLS:.*]] = tail call noalias nonnull ptr @malloc(i64 %{{.*}})
; CHECK: invertentry_count_body:
; CHECK:   call i32 @MPI_Type_size(ptr %{{.*}}, ptr %{{.*}})
; CHECK: invertentry_count_exit:
; CHECK:   %[[WBUF:.*]] = tail call noalias nonnull ptr @malloc(i64 %{{.*}})
; CHECK:   call i32 @MPI_Alltoallw(ptr %"recv_buf'", ptr %recvcounts, ptr %rdispls, ptr %recvtypes, ptr %[[WBUF]], ptr %sendcounts, ptr %[[WDISPLS]], ptr %sendtypes, ptr @ompi_mpi_comm_world)
; CHECK: invertentry_zero_body:
; CHECK:   call i32 @MPI_Type_size(ptr %{{.*}}, ptr %{{.*}})
; CHECK:   call void @llvm.memset.p0.i64(ptr %{{.*}}, i8 0, i64 %{{.*}}, i1 false)
; CHECK: invertentry_acc_body:
; CHECK:   call i32 @MPI_Type_size(ptr %{{.*}}, ptr %{{.*}})
; CHECK: invertentry_acc_exit:
; CHECK:   tail call void @free(ptr nonnull %[[WBUF]])
; CHECK:   tail call void @free(ptr nonnull %[[WDISPLS]])

; CHECK-LABEL: define internal void @diffempi_alltoallw_fortran_test(
; CHECK:   call void @mpi_alltoallw_(ptr %send_buf, ptr %sendcounts, ptr %sdispls, ptr %sendtypes, ptr %recv_buf, ptr %recvcounts, ptr %rdispls, ptr %recvtypes, ptr @f_comm, ptr %ierr)
; CHECK:   call void @mpi_comm_size_(ptr @f_comm, ptr %{{.*}}, ptr %{{.*}})
; CHECK:   call void @mpi_type_size_(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
; CHECK:   call void @mpi_alltoallw_(ptr %"recv_buf'", ptr %recvcounts, ptr %rdispls, ptr %recvtypes, ptr %{{.*}}, ptr %sendcounts, ptr %{{.*}}, ptr %sendtypes, ptr @f_comm, ptr %enzyme_mpi_ierr)
