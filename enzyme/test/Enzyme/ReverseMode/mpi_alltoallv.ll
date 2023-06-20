; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1

define dso_local void @mpi_alltoallv_test(float* %0, i32* %1, i32* %2, float* %3, i32* %4, i32* %5) #6 {
  %7 = bitcast float* %0 to i8*
  %8 = bitcast float* %3 to i8*
  %9 = tail call i32 @MPI_Alltoallv(i8* %7, i32* %1, i32* %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %8, i32* %4, i32* %5, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Alltoallv(i8*, i32*, i32*, %struct.ompi_datatype_t*, i8*, i32*, i32*, %struct.ompi_datatype_t*, %struct.ompi_communicator_t*) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
define void @caller(float* %sendbuf, float* %dsendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %drecvbuf, i32* %recvcounts, i32* %recvdispls) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (float*, i32*, i32*, float*, i32*, i32*)* @mpi_alltoallv_test to i8*), float* %sendbuf, float* %dsendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %drecvbuf, i32* %recvcounts, i32* %recvdispls)

  ret void
}

declare void @__enzyme_autodiff(i8*, ...)


; CHECK: define internal void @diffempi_alltoallv_test(float* %0, float* %"'", i32* %1, i32* %2, float* %3, float* %"'1", i32* %4, i32* %5) #0 {
; CHECK-NEXT:   %7 = alloca i32, align 4
; CHECK-NEXT:   %8 = alloca i32, align 4
; CHECK-NEXT:   %"'ipc" = bitcast float* %"'" to i8*
; CHECK-NEXT:   %9 = bitcast float* %0 to i8*
; CHECK-NEXT:   %"'ipc2" = bitcast float* %"'1" to i8*
; CHECK-NEXT:   %10 = bitcast float* %3 to i8*
; CHECK-NEXT:   %11 = tail call i32 @MPI_Alltoallv(i8* %9, i32* %1, i32* %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %10, i32* %4, i32* %5, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #0
; CHECK-NEXT:   br label %invert

; CHECK: invert:                                           ; preds = %6
; CHECK-NEXT:   %12 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %7)
; CHECK-NEXT:   %13 = load i32, i32* %7, align 4
; CHECK-NEXT:   %14 = alloca i64, align 8
; CHECK-NEXT:   store i64 0, i64* %14, align 4
; CHECK-NEXT:   br label %invert_loop

; CHECK: invert_loop:                                      ; preds = %invert_loop, %invert
; CHECK-NEXT:   %15 = phi i32 [ 0, %invert ], [ %21, %invert_loop ]
; CHECK-NEXT:   %16 = getelementptr inbounds i32, i32* %1, i32 %15
; CHECK-NEXT:   %17 = load i32, i32* %16, align 4
; CHECK-NEXT:   %18 = zext i32 %17 to i64
; CHECK-NEXT:   %19 = load i64, i64* %14, align 4
; CHECK-NEXT:   %20 = add i64 %19, %18
; CHECK-NEXT:   store i64 %20, i64* %14, align 4
; CHECK-NEXT:   %21 = add nuw nsw i32 %15, 1
; CHECK-NEXT:   %22 = icmp eq i32 %21, %13
; CHECK-NEXT:   br i1 %22, label %invert_endloop, label %invert_loop

; CHECK: invert_endloop:                                   ; preds = %invert_loop
; CHECK-NEXT:   %23 = load i64, i64* %14, align 4
; CHECK-NEXT:   %24 = mul i64 %23, 4
; CHECK-NEXT:   store i64 %24, i64* %14, align 4
; CHECK-NEXT:   %25 = load i64, i64* %14, align 4
; CHECK-NEXT:   %26 = tail call noalias nonnull i8* @malloc(i64 %25)
; CHECK-NEXT:   %27 = call i32 @MPI_Alltoallv(i8* %"'ipc2", i32* %4, i32* %5, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %26, i32* %1, i32* %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %28 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %8)
; CHECK-NEXT:   %29 = load i32, i32* %8, align 4
; CHECK-NEXT:   %30 = alloca i64, align 8
; CHECK-NEXT:   store i64 0, i64* %30, align 4
; CHECK-NEXT:   br label %invert_endloop_loop

; CHECK: invert_endloop_loop:                              ; preds = %invert_endloop_loop, %invert_endloop
; CHECK-NEXT:   %31 = phi i32 [ 0, %invert_endloop ], [ %37, %invert_endloop_loop ]
; CHECK-NEXT:   %32 = getelementptr inbounds i32, i32* %4, i32 %31
; CHECK-NEXT:   %33 = load i32, i32* %32, align 4
; CHECK-NEXT:   %34 = zext i32 %33 to i64
; CHECK-NEXT:   %35 = load i64, i64* %30, align 4
; CHECK-NEXT:   %36 = add i64 %35, %34
; CHECK-NEXT:   store i64 %36, i64* %30, align 4
; CHECK-NEXT:   %37 = add nuw nsw i32 %31, 1
; CHECK-NEXT:   %38 = icmp eq i32 %37, %29
; CHECK-NEXT:   br i1 %38, label %invert_endloop_endloop, label %invert_endloop_loop

; CHECK: invert_endloop_endloop:                           ; preds = %invert_endloop_loop
; CHECK-NEXT:   %39 = load i64, i64* %30, align 4
; CHECK-NEXT:   %40 = mul i64 %39, 4
; CHECK-NEXT:   store i64 %40, i64* %30, align 4
; CHECK-NEXT:   %41 = load i64, i64* %30, align 4
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"'ipc2", i8 0, i64 %41, i1 false)
; CHECK-NEXT:   %42 = bitcast i8* %26 to float*
; CHECK-NEXT:   %43 = bitcast i8* %"'ipc" to float*
; CHECK-NEXT:   %44 = udiv i64 %25, 4
; CHECK-NEXT:   %45 = icmp eq i64 %44, 0
; CHECK-NEXT:   br i1 %45, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invert_endloop_endloop
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invert_endloop_endloop ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %42, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i, align 1
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %43, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i, align 1
; CHECK-NEXT:   %46 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %46, float* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %47 = icmp eq i64 %44, %idx.next.i
; CHECK-NEXT:   br i1 %47, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %invert_endloop_endloop, %for.body.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %26)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


