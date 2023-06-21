; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1

define dso_local void @mpi_alltoallv_test(float* %sendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, i32* %recvcounts, i32* %recvdispls) {
  %sendbuf.bc = bitcast float* %sendbuf to i8*
  %recvbuf.bc = bitcast float* %recvbuf to i8*
  %result = call i32 @MPI_Alltoallv(i8* %sendbuf.bc, i32* %sendcounts, i32* %senddispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %recvbuf.bc, i32* %recvcounts, i32* %recvdispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Alltoallv(i8*, i32*, i32*, %struct.ompi_datatype_t*, i8*, i32*, i32*, %struct.ompi_datatype_t*, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define void @caller(float* %sendbuf, float* %dsendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %drecvbuf, i32* %recvcounts, i32* %recvdispls) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (float*, i32*, i32*, float*, i32*, i32*)* @mpi_alltoallv_test to i8*), float* %sendbuf, float* %dsendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %drecvbuf, i32* %recvcounts, i32* %recvdispls)

  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffempi_alltoallv_test(float* %sendbuf, float* %"sendbuf'", i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %"recvbuf'", i32* %recvcounts, i32* %recvdispls) #0 {
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   %2 = alloca i32
; CHECK-NEXT:   %"sendbuf.bc'ipc" = bitcast float* %"sendbuf'" to i8*
; CHECK-NEXT:   %sendbuf.bc = bitcast float* %sendbuf to i8*
; CHECK-NEXT:   %"recvbuf.bc'ipc" = bitcast float* %"recvbuf'" to i8*
; CHECK-NEXT:   %recvbuf.bc = bitcast float* %recvbuf to i8*
; CHECK-NEXT:   %result = call i32 @MPI_Alltoallv(i8* %sendbuf.bc, i32* %sendcounts, i32* %senddispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %recvbuf.bc, i32* %recvcounts, i32* %recvdispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   br label %invert

; CHECK: invert:                                           ; preds = %0
; CHECK-NEXT:   %3 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %1)
; CHECK-NEXT:   %4 = load i32, i32* %1
; CHECK-NEXT:   %5 = alloca i64, align 8
; CHECK-NEXT:   store i64 0, i64* %5
; CHECK-NEXT:   br label %invert_loop

; CHECK: invert_loop:                                      ; preds = %invert_loop, %invert
; CHECK-NEXT:   %6 = phi i32 [ 0, %invert ], [ %12, %invert_loop ]
; CHECK-NEXT:   %7 = getelementptr inbounds i32, i32* %sendcounts, i32 %6
; CHECK-NEXT:   %8 = load i32, i32* %7
; CHECK-NEXT:   %9 = zext i32 %8 to i64
; CHECK-NEXT:   %10 = load i64, i64* %5
; CHECK-NEXT:   %11 = add i64 %10, %9
; CHECK-NEXT:   store i64 %11, i64* %5
; CHECK-NEXT:   %12 = add nuw nsw i32 %6, 1
; CHECK-NEXT:   %13 = icmp eq i32 %12, %4
; CHECK-NEXT:   br i1 %13, label %invert_endloop, label %invert_loop

; CHECK: invert_endloop:                                   ; preds = %invert_loop
; CHECK-NEXT:   %14 = load i64, i64* %5
; CHECK-NEXT:   %15 = mul i64 %14, 4
; CHECK-NEXT:   store i64 %15, i64* %5
; CHECK-NEXT:   %16 = load i64, i64* %5
; CHECK-NEXT:   %17 = tail call noalias nonnull i8* @malloc(i64 %16)
; CHECK-NEXT:   %18 = call i32 @MPI_Alltoallv(i8* %"recvbuf.bc'ipc", i32* %recvcounts, i32* %recvdispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %17, i32* %sendcounts, i32* %senddispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %19 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %2)
; CHECK-NEXT:   %20 = load i32, i32* %2
; CHECK-NEXT:   %21 = alloca i64, align 8
; CHECK-NEXT:   store i64 0, i64* %21
; CHECK-NEXT:   br label %invert_endloop_loop

; CHECK: invert_endloop_loop:                              ; preds = %invert_endloop_loop, %invert_endloop
; CHECK-NEXT:   %22 = phi i32 [ 0, %invert_endloop ], [ %28, %invert_endloop_loop ]
; CHECK-NEXT:   %23 = getelementptr inbounds i32, i32* %recvcounts, i32 %22
; CHECK-NEXT:   %24 = load i32, i32* %23
; CHECK-NEXT:   %25 = zext i32 %24 to i64
; CHECK-NEXT:   %26 = load i64, i64* %21
; CHECK-NEXT:   %27 = add i64 %26, %25
; CHECK-NEXT:   store i64 %27, i64* %21
; CHECK-NEXT:   %28 = add nuw nsw i32 %22, 1
; CHECK-NEXT:   %29 = icmp eq i32 %28, %20
; CHECK-NEXT:   br i1 %29, label %invert_endloop_endloop, label %invert_endloop_loop

; CHECK: invert_endloop_endloop:                           ; preds = %invert_endloop_loop
; CHECK-NEXT:   %30 = load i64, i64* %21
; CHECK-NEXT:   %31 = mul i64 %30, 4
; CHECK-NEXT:   store i64 %31, i64* %21
; CHECK-NEXT:   %32 = load i64, i64* %21
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recvbuf.bc'ipc", i8 0, i64 %32, i1 false)
; CHECK-NEXT:   %33 = bitcast i8* %17 to float*
; CHECK-NEXT:   %34 = bitcast i8* %"sendbuf.bc'ipc" to float*
; CHECK-NEXT:   %35 = udiv i64 %16, 4
; CHECK-NEXT:   %36 = icmp eq i64 %35, 0
; CHECK-NEXT:   br i1 %36, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invert_endloop_endloop
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invert_endloop_endloop ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %33, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i, align 1
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %34, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i, align 1
; CHECK-NEXT:   %37 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %37, float* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %38 = icmp eq i64 %35, %idx.next.i
; CHECK-NEXT:   br i1 %38, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %invert_endloop_endloop, %for.body.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %17)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
