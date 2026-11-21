; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

@ompi_mpi_double = external global i8

declare i32 @MPI_Send(ptr addrspace(1), i32, ptr, i32, i32, ptr)
declare i32 @MPI_Recv(ptr addrspace(1), i32, ptr, i32, i32, ptr, ptr)
declare void @__enzyme_autodiff(...)

; Non-zero address space on MPI_Send (DifferentiableMemCopyFloats must use
; separate dst and src pointer types) and MPI_Recv (reverse memset must cast
; in the shadow pointer's address space, not addrspace(0)).
define void @f_send_as1(ptr addrspace(1) "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %buf, ptr %comm) {
  %r = call i32 @MPI_Send(ptr addrspace(1) %buf, i32 4, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm)
  ret void
}

define void @df_send_as1(ptr addrspace(1) %buf, ptr addrspace(1) %dbuf, ptr %comm) {
  call void (...) @__enzyme_autodiff(ptr @f_send_as1, metadata !"enzyme_dup", ptr addrspace(1) %buf, ptr addrspace(1) %dbuf, metadata !"enzyme_const", ptr %comm)
  ret void
}

define void @f_recv_as1(ptr addrspace(1) %buf, ptr %comm, ptr %st) {
  store double 1.0, ptr addrspace(1) %buf, align 8
  %c = call i32 @MPI_Recv(ptr addrspace(1) %buf, i32 1, ptr @ompi_mpi_double, i32 0, i32 0, ptr %comm, ptr %st)
  ret void
}

define void @df_recv_as1(ptr addrspace(1) %buf, ptr addrspace(1) %dbuf, ptr %comm, ptr %st) {
  call void (...) @__enzyme_autodiff(ptr @f_recv_as1, metadata !"enzyme_dup", ptr addrspace(1) %buf, ptr addrspace(1) %dbuf, metadata !"enzyme_const", ptr %comm, metadata !"enzyme_const", ptr %st)
  ret void
}

; CHECK-LABEL: define internal void @diffef_send_as1(
; CHECK: %[[ALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull dereferenceable(32) dereferenceable_or_null(32) ptr @malloc(i64 32)
; CHECK: call i32 @MPI_Recv(ptr %[[ALLOC]], i32 4, ptr @ompi_mpi_double
; CHECK: getelementptr inbounds double, ptr addrspace(1) %"buf'"
; CHECK: load double, ptr addrspace(1)
; CHECK: fadd fast double
; CHECK: store double {{.*}}, ptr addrspace(1)

; CHECK-LABEL: define internal void @diffef_recv_as1(
; CHECK: call i32 @MPI_Send(ptr addrspace(1) %"buf'", i32 1, ptr @ompi_mpi_double
; CHECK: call void @llvm.memset.p1.i64(ptr addrspace(1) nonnull %"buf'", i8 0, i64 8, i1 false)
