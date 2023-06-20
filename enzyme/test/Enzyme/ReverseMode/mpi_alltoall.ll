; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1

define void @mpi_alltoall_test(float* %0, i32 %1, float* %2) {
  %4 = bitcast float* %0 to i8*
  %5 = bitcast float* %2 to i8*
  %6 = tail call i32 @MPI_Alltoall(i8* %4, i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %5, i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
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

; CHECK: define internal void @diffempi_alltoall_test(float* %0, float* %"'", i32 %1, float* %2, float* %"'1") #0 {
; CHECK-NEXT:   %4 = alloca i32, align 4
; CHECK-NEXT:   %5 = alloca i32, align 4
; CHECK-NEXT:   %"'ipc" = bitcast float* %"'" to i8*
; CHECK-NEXT:   %6 = bitcast float* %0 to i8*
; CHECK-NEXT:   %"'ipc2" = bitcast float* %"'1" to i8*
; CHECK-NEXT:   %7 = bitcast float* %2 to i8*
; CHECK-NEXT:   %8 = tail call i32 @MPI_Alltoall(i8* %6, i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %7, i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #0
; CHECK-NEXT:   br label %invert

; CHECK: invert:                                           ; preds = %3
; CHECK-NEXT:   %9 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %4)
; CHECK-NEXT:   %10 = load i32, i32* %4, align 4
; CHECK-NEXT:   %11 = zext i32 %10 to i64
; CHECK-NEXT:   %12 = zext i32 %1 to i64
; CHECK-NEXT:   %13 = mul nuw nsw i64 %12, %11
; CHECK-NEXT:   %14 = mul nuw nsw i64 %13, 4
; CHECK-NEXT:   %15 = tail call noalias nonnull i8* @malloc(i64 %14)
; CHECK-NEXT:   %16 = call i32 @MPI_Alltoall(i8* %"'ipc2", i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %15, i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %17 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %5)
; CHECK-NEXT:   %18 = load i32, i32* %5, align 4
; CHECK-NEXT:   %19 = zext i32 %18 to i64
; CHECK-NEXT:   %20 = zext i32 %1 to i64
; CHECK-NEXT:   %21 = mul nuw nsw i64 %20, %19
; CHECK-NEXT:   %22 = mul nuw nsw i64 %21, 4
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"'ipc2", i8 0, i64 %22, i1 false)
; CHECK-NEXT:   %23 = bitcast i8* %15 to float*
; CHECK-NEXT:   %24 = bitcast i8* %"'ipc" to float*
; CHECK-NEXT:   %25 = udiv i64 %14, 4
; CHECK-NEXT:   %26 = icmp eq i64 %25, 0
; CHECK-NEXT:   br i1 %26, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invert
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invert ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %23, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i, align 1
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %24, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i, align 1
; CHECK-NEXT:   %27 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %27, float* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %28 = icmp eq i64 %25, %idx.next.i
; CHECK-NEXT:   br i1 %28, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %invert, %for.body.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %15)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

