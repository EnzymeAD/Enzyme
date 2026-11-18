; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque

@ompi_mpi_double = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@one = constant i32 1
@f_dt = constant i32 1275070475
@f_comm = constant i32 1140850688

define void @mpi_alltoall_test(ptr %send_buf, ptr %recv_buf) {
entry:
  %call = call i32 @MPI_Alltoall(ptr %send_buf, i32 2, ptr @ompi_mpi_double, ptr %recv_buf, i32 2, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
  ret void
}

define void @pmpi_alltoall_test(ptr %send_buf, ptr %recv_buf) {
entry:
  %call = call i32 @PMPI_Alltoall(ptr %send_buf, i32 2, ptr @ompi_mpi_double, ptr %recv_buf, i32 2, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
  ret void
}

define void @mpi_alltoall_fortran_test(ptr %send_buf, ptr %recv_buf) {
entry:
  %ierr = alloca i32, align 4
  %v = load double, ptr %send_buf, align 8
  %v2 = fadd double %v, 1.0
  store double %v2, ptr %send_buf, align 8
  call void @mpi_alltoall_(ptr %send_buf, ptr @one, ptr @f_dt, ptr %recv_buf, ptr @one, ptr @f_dt, ptr @f_comm, ptr %ierr)
  ret void
}

declare i32 @MPI_Alltoall(ptr, i32, ptr, ptr, i32, ptr, ptr)
declare i32 @PMPI_Alltoall(ptr, i32, ptr, ptr, i32, ptr, ptr)
declare void @mpi_alltoall_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

define void @caller(ptr %s, ptr %ds, ptr %r, ptr %dr) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @mpi_alltoall_test, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dup", ptr %r, ptr %dr)
  call void (ptr, ...) @__enzyme_autodiff(ptr @pmpi_alltoall_test, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dup", ptr %r, ptr %dr)
  call void (ptr, ...) @__enzyme_autodiff(ptr @mpi_alltoall_fortran_test, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dup", ptr %r, ptr %dr)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; CHECK-LABEL: define internal void @diffempi_alltoall_test(ptr %send_buf, ptr %"send_buf'", ptr %recv_buf, ptr %"recv_buf'")
; CHECK:   call i32 @MPI_Alltoall(ptr %send_buf, i32 2, ptr @ompi_mpi_double, ptr %recv_buf, i32 2, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Comm_size(ptr @ompi_mpi_comm_world, ptr %{{.*}})
; CHECK:   %[[BUF:.*]] = tail call noalias nonnull ptr @malloc(i64 %[[SENDLEN:.*]])
; CHECK:   call i32 @MPI_Alltoall(ptr %"recv_buf'", i32 2, ptr @ompi_mpi_double, ptr %[[BUF]], i32 2, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
; CHECK:   call void @llvm.memset.p0.i64(ptr nonnull %"recv_buf'", i8 0, i64 %{{.*}}, i1 false)
; CHECK:   fadd fast double
; CHECK:   tail call void @free(ptr nonnull %[[BUF]])

; CHECK-LABEL: define internal void @diffepmpi_alltoall_test(ptr %send_buf, ptr %"send_buf'", ptr %recv_buf, ptr %"recv_buf'")
; CHECK:   call i32 @PMPI_Alltoall(ptr %send_buf, i32 2, ptr @ompi_mpi_double, ptr %recv_buf, i32 2, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @PMPI_Comm_size(ptr @ompi_mpi_comm_world, ptr %{{.*}})
; CHECK:   %[[PBUF:.*]] = tail call noalias nonnull ptr @malloc(i64 %{{.*}})
; CHECK:   call i32 @PMPI_Alltoall(ptr %"recv_buf'", i32 2, ptr @ompi_mpi_double, ptr %[[PBUF]], i32 2, ptr @ompi_mpi_double, ptr @ompi_mpi_comm_world)
; CHECK:   call void @llvm.memset.p0.i64(ptr nonnull %"recv_buf'", i8 0, i64 %{{.*}}, i1 false)
; CHECK:   fadd fast double
; CHECK:   tail call void @free(ptr nonnull %[[PBUF]])

; CHECK-LABEL: define internal void @diffempi_alltoall_fortran_test(ptr %send_buf, ptr %"send_buf'", ptr %recv_buf, ptr %"recv_buf'")
; CHECK:   call void @mpi_alltoall_(ptr %send_buf, ptr @one, ptr @f_dt, ptr %recv_buf, ptr @one, ptr @f_dt, ptr @f_comm, ptr %ierr)
; CHECK:   call void @mpi_type_size_(ptr @f_dt, ptr %{{.*}}, ptr %{{.*}})
; CHECK:   call void @mpi_comm_size_(ptr @f_comm, ptr %{{.*}}, ptr %{{.*}})
; CHECK:   %[[FBUF:.*]] = tail call noalias nonnull ptr @malloc(i64 %{{.*}})
; CHECK:   call void @mpi_alltoall_(ptr %"recv_buf'", ptr @one, ptr @f_dt, ptr %[[FBUF]], ptr @one, ptr @f_dt, ptr @f_comm, ptr %enzyme_mpi_ierr)
; CHECK:   call void @llvm.memset.p0.i64(ptr nonnull %"recv_buf'", i8 0, i64 %{{.*}}, i1 false)
; CHECK:   tail call void @free(ptr nonnull %[[FBUF]])
