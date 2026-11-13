; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

@ompi_mpi_double = external global i8
@ompi_mpi_float = external global i8

declare i32 @MPI_Send(ptr, i32, ptr, i32, i32, ptr)
declare i32 @MPI_Isend(ptr, i32, ptr, i32, i32, ptr, ptr)
declare i32 @MPI_Wait(ptr, ptr)
declare void @__enzyme_autodiff(...)
declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)

; 1. Multi-segment struct: GEPs must offset from the base allocation/shadow,
; not accumulate across iterations (which previously produced +48 on a 40-byte buffer).
define void @f_multi(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Float@double, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Float@double}" %buf, ptr %comm) {
  %r = call i32 @MPI_Send(ptr %buf, i32 5, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm)
  ret void
}

define void @df_multi(ptr %buf, ptr %dbuf, ptr %comm) {
  call void (...) @__enzyme_autodiff(ptr @f_multi, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_const", ptr %comm)
  ret void
}

; 2. Zero-length MPI_Send must not fail with CannotDeduceType.
define void @f_zero(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %buf, ptr %comm) {
  %r = call i32 @MPI_Send(ptr %buf, i32 0, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm)
  ret void
}

define void @df_zero(ptr %buf, ptr %dbuf, ptr %comm) {
  call void (...) @__enzyme_autodiff(ptr @f_zero, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_const", ptr %comm)
  ret void
}

; 3. Anything padding between floats must not poison segments and drop gradients.
define void @f_pad_any(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything, [-1,8]:Float@double}" %buf, ptr %comm) {
  %r = call i32 @MPI_Send(ptr %buf, i32 4, ptr @ompi_mpi_float, i32 1, i32 0, ptr %comm)
  ret void
}

define void @df_pad_any(ptr %buf, ptr %dbuf, ptr %comm) {
  call void (...) @__enzyme_autodiff(ptr @f_pad_any, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_const", ptr %comm)
  ret void
}

; 4. Leading Anything followed by Float must still accumulate the trailing Float.
define void @f_pad_lead(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Anything, [-1,1]:Anything, [-1,2]:Anything, [-1,3]:Anything, [-1,4]:Anything, [-1,5]:Anything, [-1,6]:Anything, [-1,7]:Anything, [-1,8]:Float@double}" %buf, ptr %comm) {
  %r = call i32 @MPI_Send(ptr %buf, i32 2, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm)
  ret void
}

define void @df_pad_lead(ptr %buf, ptr %dbuf, ptr %comm) {
  call void (...) @__enzyme_autodiff(ptr @f_pad_lead, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_const", ptr %comm)
  ret void
}

; 5. Wildcard Float with specific Anything hole must split cleanly without looping.
define void @f_pad_wild(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double, [-1,8]:Anything, [-1,9]:Anything, [-1,10]:Anything, [-1,11]:Anything, [-1,12]:Anything, [-1,13]:Anything, [-1,14]:Anything, [-1,15]:Anything}" %buf, ptr %comm) {
  %r = call i32 @MPI_Send(ptr %buf, i32 3, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm)
  ret void
}

define void @df_pad_wild(ptr %buf, ptr %dbuf, ptr %comm) {
  call void (...) @__enzyme_autodiff(ptr @f_pad_wild, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_const", ptr %comm)
  ret void
}

; 6. Runtime activity on MPI_Send must guard accumulation with shadow == primal check.
define void @f_ra(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %buf, ptr %comm) {
  %r = call i32 @MPI_Send(ptr %buf, i32 1, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm)
  ret void
}

define void @df_ra(ptr %buf, ptr %dbuf, ptr %comm) {
  call void (...) @__enzyme_autodiff(ptr @f_ra, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_const", ptr %comm)
  ret void
}

; 7. Runtime activity on MPI_Isend + MPI_Wait in split reverse mode must preserve primal buffer.
define void @f_isend_ra(ptr %pp, ptr %comm, ptr %req, ptr %st) {
  %buf = load ptr, ptr %pp, align 8
  store double 7.0, ptr %buf, align 8
  %c1 = call i32 @MPI_Isend(ptr %buf, i32 1, ptr @ompi_mpi_double, i32 1, i32 0, ptr %comm, ptr %req)
  %c2 = call i32 @MPI_Wait(ptr %req, ptr %st)
  ret void
}

define void @df_isend_ra(ptr %pp, ptr %dpp, ptr %comm, ptr %req, ptr %dreq, ptr %st) {
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @f_isend_ra, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %pp, ptr %dpp, metadata !"enzyme_const", ptr %comm, metadata !"enzyme_dup", ptr %req, ptr %dreq, metadata !"enzyme_const", ptr %st)
  call void (...) @__enzyme_reverse(ptr @f_isend_ra, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %pp, ptr %dpp, metadata !"enzyme_const", ptr %comm, metadata !"enzyme_dup", ptr %req, ptr %dreq, metadata !"enzyme_const", ptr %st, ptr %tape)
  ret void
}

; CHECK-LABEL: define internal void @diffef_multi(
; CHECK: %[[MALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull dereferenceable(40) dereferenceable_or_null(40) ptr @malloc(i64 40)
; CHECK: call i32 @MPI_Recv(ptr %[[MALLOC]], i32 5, ptr @ompi_mpi_double
; CHECK: getelementptr inbounds double, ptr %[[MALLOC]]
; CHECK: getelementptr inbounds i8, ptr %[[MALLOC]], i64 16
; CHECK: getelementptr inbounds i8, ptr %"buf'", i64 16
; CHECK: getelementptr inbounds i8, ptr %[[MALLOC]], i64 32
; CHECK: getelementptr inbounds i8, ptr %"buf'", i64 32

; CHECK-LABEL: define internal void @diffef_zero(
; CHECK: %[[ZALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull ptr @malloc(i64 0)
; CHECK: call i32 @MPI_Recv(ptr %[[ZALLOC]], i32 0, ptr @ompi_mpi_double
; CHECK-NOT: fadd
; CHECK: tail call void @free(ptr nonnull %[[ZALLOC]])
; CHECK-NEXT: ret void

; CHECK-LABEL: define internal void @diffef_pad_any(
; CHECK: %[[PALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) ptr @malloc(i64 16)
; CHECK: fadd fast float
; CHECK: getelementptr inbounds i8, ptr %[[PALLOC]], i64 8
; CHECK: getelementptr inbounds i8, ptr %"buf'", i64 8
; CHECK: fadd fast double

; CHECK-LABEL: define internal void @diffef_pad_lead(
; CHECK: %[[LALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) ptr @malloc(i64 16)
; CHECK: getelementptr inbounds i8, ptr %[[LALLOC]], i64 8
; CHECK: getelementptr inbounds i8, ptr %"buf'", i64 8
; CHECK: fadd fast double

; CHECK-LABEL: define internal void @diffef_pad_wild(
; CHECK: %[[WALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull dereferenceable(24) dereferenceable_or_null(24) ptr @malloc(i64 24)
; CHECK: fadd fast double
; CHECK: getelementptr inbounds i8, ptr %[[WALLOC]], i64 16
; CHECK: getelementptr inbounds i8, ptr %"buf'", i64 16
; CHECK: fadd fast double

; CHECK-LABEL: define internal void @diffef_ra(
; CHECK: %[[RALLOC:[0-9a-zA-Z_.]+]] = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) ptr @malloc(i64 8)
; CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp eq ptr %"buf'", %buf
; CHECK: br i1 %[[CMP]], label %memset_dst.i, label %for.body.i
; CHECK: memset_dst.i:
; CHECK: call void @llvm.memset.p0.i64(ptr align 1 %[[RALLOC]], i8 0, i64 8, i1 false)

; CHECK-LABEL: define internal void @diffef_isend_ra(
; CHECK: icmp eq ptr
; CHECK: memset_dst.i
