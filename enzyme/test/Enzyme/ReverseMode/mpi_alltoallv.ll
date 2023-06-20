; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

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
; CHECK-NEXT: invert:
; CHECK-NEXT:   %6 = alloca i32, align 4
; CHECK-NEXT:   %7 = alloca i32, align 4
; CHECK-NEXT:   %8 = bitcast float* %0 to i8*
; CHECK-NEXT:   %"'ipc2" = bitcast float* %"'1" to i8*
; CHECK-NEXT:   %9 = bitcast float* %3 to i8*
; CHECK-NEXT:   %10 = tail call i32 @MPI_Alltoallv(i8* %8, i32* %1, i32* %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %9, i32* %4, i32* %5, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #0
; CHECK-NEXT:   %11 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %6)
; CHECK-NEXT:   %12 = load i32, i32* %6, align 4
; CHECK-NEXT:   %13 = alloca i64, align 8
; CHECK-NEXT:   store i64 0, i64* %13, align 4
; CHECK-NEXT:   br label %invert_loop

; CHECK: invert_loop:                                      ; preds = %invert_loop, %invert
; CHECK-NEXT:   %14 = phi i32 [ 0, %invert ], [ %20, %invert_loop ]
; CHECK-NEXT:   %15 = getelementptr inbounds i32, i32* %1, i32 %14
; CHECK-NEXT:   %16 = load i32, i32* %15, align 4
; CHECK-NEXT:   %17 = zext i32 %16 to i64
; CHECK-NEXT:   %18 = load i64, i64* %13, align 4
; CHECK-NEXT:   %19 = add i64 %18, %17
; CHECK-NEXT:   store i64 %19, i64* %13, align 4
; CHECK-NEXT:   %20 = add nuw nsw i32 %14, 1
; CHECK-NEXT:   %21 = icmp eq i32 %20, %12
; CHECK-NEXT:   br i1 %21, label %invert_endloop, label %invert_loop

; CHECK: invert_endloop:                                   ; preds = %invert_loop
; CHECK-NEXT:   %22 = load i64, i64* %13, align 4
; CHECK-NEXT:   %23 = mul i64 %22, 4
; CHECK-NEXT:   store i64 %23, i64* %13, align 4
; CHECK-NEXT:   %24 = load i64, i64* %13, align 4
; CHECK-NEXT:   %25 = tail call noalias nonnull i8* @malloc(i64 %24)
; CHECK-NEXT:   %26 = call i32 @MPI_Alltoallv(i8* %"'ipc2", i32* %4, i32* %5, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %25, i32* %1, i32* %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %27 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %7)
; CHECK-NEXT:   %28 = load i32, i32* %7, align 4
; CHECK-NEXT:   %29 = alloca i64, align 8
; CHECK-NEXT:   store i64 0, i64* %29, align 4
; CHECK-NEXT:   br label %invert_endloop_loop

; CHECK: invert_endloop_loop:                              ; preds = %invert_endloop_loop, %invert_endloop
; CHECK-NEXT:   %30 = phi i32 [ 0, %invert_endloop ], [ %36, %invert_endloop_loop ]
; CHECK-NEXT:   %31 = getelementptr inbounds i32, i32* %4, i32 %30
; CHECK-NEXT:   %32 = load i32, i32* %31, align 4
; CHECK-NEXT:   %33 = zext i32 %32 to i64
; CHECK-NEXT:   %34 = load i64, i64* %29, align 4
; CHECK-NEXT:   %35 = add i64 %34, %33
; CHECK-NEXT:   store i64 %35, i64* %29, align 4
; CHECK-NEXT:   %36 = add nuw nsw i32 %30, 1
; CHECK-NEXT:   %37 = icmp eq i32 %36, %28
; CHECK-NEXT:   br i1 %37, label %invert_endloop_endloop, label %invert_endloop_loop

; CHECK: invert_endloop_endloop:                           ; preds = %invert_endloop_loop
; CHECK-NEXT:   %38 = load i64, i64* %29, align 4
; CHECK-NEXT:   %39 = mul i64 %38, 4
; CHECK-NEXT:   store i64 %39, i64* %29, align 4
; CHECK-NEXT:   %40 = load i64, i64* %29, align 4
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"'ipc2", i8 0, i64 %40, i1 false)
; CHECK-NEXT:   %41 = bitcast i8* %25 to float*
; CHECK-NEXT:   %42 = udiv i64 %24, 4
; CHECK-NEXT:   %43 = icmp eq i64 %42, 0
; CHECK-NEXT:   br i1 %43, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invert_endloop_endloop
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invert_endloop_endloop ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %41, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i, align 1
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %"'", i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i, align 1
; CHECK-NEXT:   %44 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %44, float* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %45 = icmp eq i64 %42, %idx.next.i
; CHECK-NEXT:   br i1 %45, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %invert_endloop_endloop, %for.body.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %25)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

