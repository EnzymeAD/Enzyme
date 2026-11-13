; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

@ompi_mpi_double = external global i8

declare i32 @MPI_Isend(ptr, i32, ptr, i32, i32, ptr, ptr)
declare i32 @MPI_Recv(ptr, i32, ptr, i32, i32, ptr, ptr)
declare i32 @MPI_Waitall(i32, ptr, ptr)
declare i32 @PMPI_Waitall(i32, i64, ptr)
declare i32 @MPI_Gather(ptr, i32, ptr, ptr, i32, ptr, i32, ptr)
declare i32 @ext_get_count(i32) nofree nounwind willreturn
declare void @__enzyme_autodiff(...)
declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)

; 1. MPI_Waitall in split reverse mode: DifferentialUseAnalysis must retain
; both count (arg 0) and array_of_requests (arg 1).
define void @f_waitall_split(ptr %buf, ptr %reqs, ptr %sts, ptr %comm) {
entry:
  %cnt = call i32 @ext_get_count(i32 0)
  store double 1.0, ptr %buf, align 8
  %c = call i32 @MPI_Isend(ptr %buf, i32 1, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm, ptr %reqs)
  %w = call i32 @MPI_Waitall(i32 %cnt, ptr %reqs, ptr %sts)
  ret void
}

define void @df_waitall_split(ptr %buf, ptr %dbuf, ptr %reqs, ptr %dreqs, ptr %sts, ptr %comm) {
entry:
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @f_waitall_split, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_dup", ptr %reqs, ptr %dreqs, metadata !"enzyme_const", ptr %sts, metadata !"enzyme_const", ptr %comm)
  call void (...) @__enzyme_reverse(ptr @f_waitall_split, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_dup", ptr %reqs, ptr %dreqs, metadata !"enzyme_const", ptr %sts, metadata !"enzyme_const", ptr %comm, ptr %tape)
  ret void
}

; 2. PMPI_Waitall with integer-typed request array must inttoptr in reverse pass.
define void @f_waitall_int(i32 %cnt, ptr %reqs, ptr %sts) {
entry:
  %reqi = ptrtoint ptr %reqs to i64
  %w = call i32 @PMPI_Waitall(i32 %cnt, i64 %reqi, ptr %sts)
  ret void
}

define void @df_waitall_int(i32 %cnt, ptr %reqs, ptr %dreqs, ptr %sts) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f_waitall_int, i32 %cnt, metadata !"enzyme_dup", ptr %reqs, ptr %dreqs, metadata !"enzyme_const", ptr %sts)
  ret void
}

; 3. MPI_Recv with jl_roots bundle must forward MemsetDefs to the reverse memset.
define void @f_recv_bundle(ptr %buf, ptr %comm, ptr %st, ptr %root) {
entry:
  store double 1.0, ptr %buf, align 8
  %c = call i32 @MPI_Recv(ptr %buf, i32 1, ptr @ompi_mpi_double, i32 0, i32 0, ptr %comm, ptr %st) [ "jl_roots"(ptr %root) ]
  ret void
}

define void @df_recv_bundle(ptr %buf, ptr %dbuf, ptr %comm, ptr %st, ptr %root, ptr %droot) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f_recv_bundle, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_const", ptr %comm, metadata !"enzyme_const", ptr %st, metadata !"enzyme_dup", ptr %root, ptr %droot)
  ret void
}

; 4. MPI_Gather in a loop with dynamic sendcount must look up the per-iteration
; cached sendcount in the reverse pass.
define void @f_gather_loop(ptr %sbuf, ptr %rbuf, ptr %comm, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %cnt = call i32 @ext_get_count(i32 %i)
  store double 1.0, ptr %sbuf, align 8
  store double 1.0, ptr %rbuf, align 8
  %c = call i32 @MPI_Gather(ptr %sbuf, i32 %cnt, ptr @ompi_mpi_double, ptr %rbuf, i32 1, ptr @ompi_mpi_double, i32 0, ptr %comm)
  %inc = add nuw nsw i32 %i, 1
  %cmp = icmp eq i32 %inc, %n
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}

define void @df_gather_loop(ptr %sbuf, ptr %dsbuf, ptr %rbuf, ptr %drbuf, ptr %comm, i32 %n) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f_gather_loop, metadata !"enzyme_dup", ptr %sbuf, ptr %dsbuf, metadata !"enzyme_dup", ptr %rbuf, ptr %drbuf, metadata !"enzyme_const", ptr %comm, i32 %n)
  ret void
}

; CHECK-LABEL: define internal void @diffef_waitall_split(
; CHECK: %cnt = extractvalue { ptr, i32 }
; CHECK: icmp ne i32 %cnt, 0

; CHECK-LABEL: define internal void @diffef_waitall_int(
; CHECK: invertentry:
; CHECK: %[[REQP:[0-9a-zA-Z_.]+]] = inttoptr i64 %reqi to ptr
; CHECK: invertentry_loop:
; CHECK: getelementptr inbounds ptr, ptr %[[REQP]], i32

; CHECK-LABEL: define internal void @diffef_recv_bundle(
; CHECK: call i32 @MPI_Send(ptr %"buf'", i32 1, ptr @ompi_mpi_double, i32 0, i32 0, ptr %comm) [ "jl_roots"(ptr %root, ptr %"root'") ]
; CHECK: call void @llvm.memset.p0.i64(ptr nonnull %"buf'", i8 0, i64 8, i1 false) [ "jl_roots"(ptr %"root'") ]

; CHECK-LABEL: define internal void @diffef_gather_loop(
; CHECK: %cnt_cache = alloca ptr
; CHECK: tail call noalias nonnull ptr @malloc(i64 %mallocsize), !enzyme_cache_alloc
; CHECK: invertloop:
; CHECK: %[[CACHED_CNT:[0-9a-zA-Z_.]+]] = load i32, ptr
; CHECK: %[[GALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull ptr @malloc
; CHECK: call i32 @MPI_Scatter(ptr %"rbuf'", i32 1, ptr @ompi_mpi_double, ptr %[[GALLOC]], i32 %[[CACHED_CNT]], ptr @ompi_mpi_double, i32 0, ptr %comm)
