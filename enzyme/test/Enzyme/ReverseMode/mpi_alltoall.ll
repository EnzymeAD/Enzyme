; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1

define void @mpi_alltoall_test(float* %sendbuf, i32 %count, float* %recvbuf) {
  %sendbuf.bc = bitcast float* %sendbuf to i8*
  %recvbuf.bc = bitcast float* %recvbuf to i8*
  %result = call i32 @MPI_Alltoall(i8* %sendbuf.bc, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %recvbuf.bc, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Alltoall(i8*, i32, %struct.ompi_datatype_t*, i8*, i32, %struct.ompi_datatype_t*, %struct.ompi_communicator_t*)

; Function Attrs: nounwind uwtable
define void @caller(float* %sendbuf, float* %dsendbuf, i32 %count, float* %recvbuf, float* %drecvbuf) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (float*, i32, float*)* @mpi_alltoall_test to i8*), float* %sendbuf, float* %dsendbuf, i32 %count, float* %recvbuf, float* %drecvbuf)

  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffempi_alltoall_test(float* %sendbuf, float* %"sendbuf'", i32 %count, float* %recvbuf, float* %"recvbuf'") #0 {
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   %2 = alloca i32
; CHECK-NEXT:   %"sendbuf.bc'ipc" = bitcast float* %"sendbuf'" to i8*
; CHECK-NEXT:   %sendbuf.bc = bitcast float* %sendbuf to i8*
; CHECK-NEXT:   %"recvbuf.bc'ipc" = bitcast float* %"recvbuf'" to i8*
; CHECK-NEXT:   %recvbuf.bc = bitcast float* %recvbuf to i8*
; CHECK-NEXT:   %result = call i32 @MPI_Alltoall(i8* %sendbuf.bc, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %recvbuf.bc, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   br label %invert

; CHECK: invert:                                           ; preds = %0
; CHECK-NEXT:   %3 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %1)
; CHECK-NEXT:   %4 = load i32, i32* %1
; CHECK-NEXT:   %5 = zext i32 %4 to i64
; CHECK-NEXT:   %6 = zext i32 %count to i64
; CHECK-NEXT:   %7 = mul nuw nsw i64 %6, %5
; CHECK-NEXT:   %8 = mul nuw nsw i64 %7, 4
; CHECK-NEXT:   %9 = tail call noalias nonnull i8* @malloc(i64 %8)
; CHECK-NEXT:   %10 = call i32 @MPI_Alltoall(i8* %"recvbuf.bc'ipc", i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %9, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %11 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %2)
; CHECK-NEXT:   %12 = load i32, i32* %2
; CHECK-NEXT:   %13 = zext i32 %12 to i64
; CHECK-NEXT:   %14 = zext i32 %count to i64
; CHECK-NEXT:   %15 = mul nuw nsw i64 %14, %13
; CHECK-NEXT:   %16 = mul nuw nsw i64 %15, 4
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recvbuf.bc'ipc", i8 0, i64 %16, i1 false)
; CHECK-NEXT:   %17 = bitcast i8* %9 to float*
; CHECK-NEXT:   %18 = bitcast i8* %"sendbuf.bc'ipc" to float*
; CHECK-NEXT:   %19 = udiv i64 %8, 4
; CHECK-NEXT:   %20 = icmp eq i64 %19, 0
; CHECK-NEXT:   br i1 %20, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invert
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invert ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %17, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %18, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i
; CHECK-NEXT:   %21 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %21, float* %src.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %22 = icmp eq i64 %19, %idx.next.i
; CHECK-NEXT:   br i1 %22, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %invert, %for.body.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %9)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

