; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque

@ompi_mpi_double = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@key_str = private constant [4 x i8] c"key\00"
@val_str = private constant [4 x i8] c"val\00"
@ndims_c = constant i32 2
@order_c = constant i32 56
@f_dt = constant i32 1275070475
@f_comm = constant i32 1140850688
@one = constant i32 1

define double @inactive_mpi_c(double %x, ptr %ranks, ptr %sizes, ptr %subsizes, ptr %starts, ptr %sendbuf, ptr %recvbuf) {
entry:
  %flag = alloca i32, align 4
  %grp = alloca ptr, align 8
  %subgrp = alloca ptr, align 8
  %info = alloca ptr, align 8
  %subdt = alloca ptr, align 8
  %r0 = call i32 @MPI_Initialized(ptr %flag)
  %t0 = call double @MPI_Wtick()
  %r1 = call i32 @MPI_Comm_group(ptr @ompi_mpi_comm_world, ptr %grp)
  %g = load ptr, ptr %grp, align 8
  %r2 = call i32 @MPI_Group_incl(ptr %g, i32 1, ptr %ranks, ptr %subgrp)
  %r3 = call i32 @MPI_Info_create(ptr %info)
  %iv = load ptr, ptr %info, align 8
  %r4 = call i32 @MPI_Info_set(ptr %iv, ptr @key_str, ptr @val_str)
  %r5 = call i32 @MPI_Type_create_subarray(i32 2, ptr %sizes, ptr %subsizes, ptr %starts, i32 56, ptr @ompi_mpi_double, ptr %subdt)
  %r6 = call i32 @MPI_Type_commit(ptr %subdt)
  %dt = load ptr, ptr %subdt, align 8
  %v = load double, ptr %sendbuf, align 8
  %v2 = fadd double %v, 1.0
  store double %v2, ptr %sendbuf, align 8
  %r7 = call i32 @MPI_Alltoall(ptr %sendbuf, i32 1, ptr %dt, ptr %recvbuf, i32 1, ptr %dt, ptr @ompi_mpi_comm_world)
  %r8 = call i32 @MPI_Type_free(ptr %subdt)
  %mul = fmul double %x, %x
  %scaled = fmul double %x, %t0
  %add = fadd double %mul, %scaled
  ret double %add
}

define double @inactive_mpi_fortran(double %x, ptr %ranks, ptr %sizes, ptr %subsizes, ptr %starts, ptr %sendbuf, ptr %recvbuf) {
entry:
  %ierr = alloca i32, align 4
  %flag = alloca i32, align 4
  %grp = alloca i32, align 4
  %subgrp = alloca i32, align 4
  %info = alloca i32, align 4
  %subdt = alloca i32, align 4
  call void @mpi_initialized_(ptr %flag, ptr %ierr)
  %t0 = call double @mpi_wtick_()
  call void @mpi_comm_group_(ptr @f_comm, ptr %grp, ptr %ierr)
  call void @mpi_group_incl_(ptr %grp, ptr @one, ptr %ranks, ptr %subgrp, ptr %ierr)
  call void @mpi_info_create_(ptr %info, ptr %ierr)
  call void @mpi_info_set_(ptr %info, ptr @key_str, ptr @val_str, ptr %ierr, i64 3, i64 3)
  call void @mpi_type_create_subarray_(ptr @ndims_c, ptr %sizes, ptr %subsizes, ptr %starts, ptr @order_c, ptr @f_dt, ptr %subdt, ptr %ierr)
  call void @mpi_type_commit_(ptr %subdt, ptr %ierr)
  %v = load double, ptr %sendbuf, align 8
  %v2 = fadd double %v, 1.0
  store double %v2, ptr %sendbuf, align 8
  call void @mpi_alltoall_(ptr %sendbuf, ptr @one, ptr %subdt, ptr %recvbuf, ptr @one, ptr %subdt, ptr @f_comm, ptr %ierr)
  call void @mpi_type_free_(ptr %subdt, ptr %ierr)
  %mul = fmul double %x, %x
  %scaled = fmul double %x, %t0
  %add = fadd double %mul, %scaled
  ret double %add
}

declare i32 @MPI_Initialized(ptr)
declare double @MPI_Wtick()
declare i32 @MPI_Comm_group(ptr, ptr)
declare i32 @MPI_Group_incl(ptr, i32, ptr, ptr)
declare i32 @MPI_Info_create(ptr)
declare i32 @MPI_Info_set(ptr, ptr, ptr)
declare i32 @MPI_Type_create_subarray(i32, ptr, ptr, ptr, i32, ptr, ptr)
declare i32 @MPI_Type_commit(ptr)
declare i32 @MPI_Type_free(ptr)
declare i32 @MPI_Alltoall(ptr, i32, ptr, ptr, i32, ptr, ptr)

declare void @mpi_initialized_(ptr, ptr)
declare double @mpi_wtick_()
declare void @mpi_comm_group_(ptr, ptr, ptr)
declare void @mpi_group_incl_(ptr, ptr, ptr, ptr, ptr)
declare void @mpi_info_create_(ptr, ptr)
declare void @mpi_info_set_(ptr, ptr, ptr, ptr, i64, i64)
declare void @mpi_type_create_subarray_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)
declare void @mpi_type_commit_(ptr, ptr)
declare void @mpi_type_free_(ptr, ptr)
declare void @mpi_alltoall_(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

define void @caller(double %x, ptr %ranks, ptr %sizes, ptr %subsizes, ptr %starts, ptr %s, ptr %ds, ptr %r, ptr %dr) {
entry:
  %0 = call double (ptr, ...) @__enzyme_autodiff(ptr @inactive_mpi_c, double %x, metadata !"enzyme_const", ptr %ranks, metadata !"enzyme_const", ptr %sizes, metadata !"enzyme_const", ptr %subsizes, metadata !"enzyme_const", ptr %starts, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dup", ptr %r, ptr %dr)
  %1 = call double (ptr, ...) @__enzyme_autodiff(ptr @inactive_mpi_fortran, double %x, metadata !"enzyme_const", ptr %ranks, metadata !"enzyme_const", ptr %sizes, metadata !"enzyme_const", ptr %subsizes, metadata !"enzyme_const", ptr %starts, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_dup", ptr %r, ptr %dr)
  ret void
}

declare double @__enzyme_autodiff(ptr, ...)

; In ReverseMode, MPI_Type_free is removed from the forward pass so %subdt stays
; valid for the reverse MPI_Alltoall, and freed afterwards at the reverse of
; MPI_Type_create_subarray.

; CHECK-LABEL: define internal { double } @diffeinactive_mpi_c(
; CHECK:   call i32 @MPI_Initialized(ptr %flag)
; CHECK:   call double @MPI_Wtick()
; CHECK:   call i32 @MPI_Comm_group(ptr @ompi_mpi_comm_world, ptr %grp)
; CHECK:   call i32 @MPI_Group_incl(ptr %g, i32 1, ptr %ranks, ptr %subgrp)
; CHECK:   call i32 @MPI_Info_create(ptr %info)
; CHECK:   call i32 @MPI_Info_set(ptr %iv, ptr @key_str, ptr @val_str)
; CHECK:   call i32 @MPI_Type_create_subarray(i32 2, ptr %sizes, ptr %subsizes, ptr %starts, i32 56, ptr @ompi_mpi_double, ptr %subdt)
; CHECK:   call i32 @MPI_Type_commit(ptr %subdt)
; CHECK:   call i32 @MPI_Alltoall(ptr %sendbuf, i32 1, ptr %dt, ptr %recvbuf, i32 1, ptr %dt, ptr @ompi_mpi_comm_world)
; CHECK-NOT: call i32 @MPI_Type_free
; CHECK:   call i32 @MPI_Alltoall(ptr %"recvbuf'", i32 1, ptr %dt, ptr %{{.*}}, i32 1, ptr %dt, ptr @ompi_mpi_comm_world)
; CHECK:   call i32 @MPI_Type_free(ptr %subdt)

; CHECK-LABEL: define internal { double } @diffeinactive_mpi_fortran(
; CHECK:   call void @mpi_initialized_(ptr %flag, ptr %ierr)
; CHECK:   call double @mpi_wtick_()
; CHECK:   call void @mpi_comm_group_(ptr @f_comm, ptr %grp, ptr %ierr)
; CHECK:   call void @mpi_group_incl_(ptr %grp, ptr @one, ptr %ranks, ptr %subgrp, ptr %ierr)
; CHECK:   call void @mpi_info_create_(ptr %info, ptr %ierr)
; CHECK:   call void @mpi_info_set_(ptr %info, ptr @key_str, ptr @val_str, ptr %ierr, i64 3, i64 3)
; CHECK:   call void @mpi_type_create_subarray_(ptr @ndims_c, ptr %sizes, ptr %subsizes, ptr %starts, ptr @order_c, ptr @f_dt, ptr %subdt, ptr %ierr)
; CHECK:   call void @mpi_type_commit_(ptr %subdt, ptr %ierr)
; CHECK:   call void @mpi_alltoall_(ptr %sendbuf, ptr @one, ptr %subdt, ptr %recvbuf, ptr @one, ptr %subdt, ptr @f_comm, ptr %ierr)
; CHECK-NOT: call void @mpi_type_free_
; CHECK:   call void @mpi_alltoall_(ptr %"recvbuf'", ptr @one, ptr %subdt, ptr %{{.*}}, ptr @one, ptr %subdt, ptr @f_comm, ptr %enzyme_mpi_ierr)
; CHECK:   call void @mpi_type_free_(ptr %subdt, ptr %enzyme_mpi_ierr{{.*}})
