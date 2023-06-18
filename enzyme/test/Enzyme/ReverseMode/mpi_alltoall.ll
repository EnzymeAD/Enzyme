; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1

define void @mpi_alltoall_test(float* noundef %0, i32 noundef %1, float* noundef %2) {
  %4 = bitcast float* %0 to i8*
  %5 = bitcast float* %2 to i8*
  %6 = tail call i32 @MPI_Alltoall(i8* noundef %4, i32 noundef %1, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* noundef %5, i32 noundef %1, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* noundef bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Alltoall(i8* noundef, i32 noundef, %struct.ompi_datatype_t* noundef, i8* noundef, i32 noundef, %struct.ompi_datatype_t* noundef, %struct.ompi_communicator_t* noundef)

; Function Attrs: nounwind uwtable
define void @caller(float* %sendbuf, float* %dsendbuf, i32 %count, float* %recvbuf, float* %drecvbuf) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (float*, i32, float*)* @mpi_alltoall_test to i8*), float* noundef %sendbuf, float* noundef %dsendbuf, i32 noundef %count, float* noundef %recvbuf, float* noundef %drecvbuf)

  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffempi_alltoall_test(float* noundef %0, float* %"'", i32 noundef %1, float* noundef %2, float* %"'1") #0 {
; CHECK-NEXT: invert:
; CHECK-NEXT:   %3 = alloca i32, align 4
; CHECK-NEXT:   %4 = alloca i32, align 4
; CHECK-NEXT:   %5 = bitcast float* %0 to i8*
; CHECK-NEXT:   %"'ipc2" = bitcast float* %"'1" to i8*
; CHECK-NEXT:   %6 = bitcast float* %2 to i8*
; CHECK-NEXT:   %7 = tail call i32 @MPI_Alltoall(i8* noundef %5, i32 noundef %1, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* noundef %6, i32 noundef %1, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* noundef bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #0
; CHECK-NEXT:   %8 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %3)
; CHECK-NEXT:   %9 = load i32, i32* %3, align 4
; CHECK-NEXT:   %10 = zext i32 %9 to i64
; CHECK-NEXT:   %11 = zext i32 %1 to i64
; CHECK-NEXT:   %12 = mul nuw nsw i64 %11, %10
; CHECK-NEXT:   %13 = mul nuw nsw i64 %12, 4
; CHECK-NEXT:   %14 = tail call noalias nonnull i8* @malloc(i64 %13)
; CHECK-NEXT:   %15 = call i32 @MPI_Alltoall(i8* %"'ipc2", i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %14, i32 %1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %16 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %4)
; CHECK-NEXT:   %17 = load i32, i32* %4, align 4
; CHECK-NEXT:   %18 = zext i32 %17 to i64
; CHECK-NEXT:   %19 = zext i32 %1 to i64
; CHECK-NEXT:   %20 = mul nuw nsw i64 %19, %18
; CHECK-NEXT:   %21 = mul nuw nsw i64 %20, 4
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"'ipc2", i8 0, i64 %21, i1 false)
; CHECK-NEXT:   %22 = bitcast i8* %14 to float*
; CHECK-NEXT:   %23 = icmp eq i64 %12, 0
; CHECK-NEXT:   br i1 %23, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invert
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invert ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %22, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i, align 1
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %"'", i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i, align 1
; CHECK-NEXT:   %24 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %24, float* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %25 = icmp eq i64 %12, %idx.next.i
; CHECK-NEXT:   br i1 %25, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %invert, %for.body.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %14)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


