; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_predefined_op_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque
%struct.ompi_op_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_op_max = external global %struct.ompi_predefined_op_t, align 1

define void @mpi_allreduce_max_test(float* %sendbuf, float* %recvbuf, i32 %count) {
  %sendbuf.bc = bitcast float* %sendbuf to i8*
  %recvbuf.bc = bitcast float* %recvbuf to i8*
  %result = call i32 @MPI_Allreduce(i8* %sendbuf.bc, i8* %recvbuf.bc, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_max to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Allreduce(i8*, i8*, i32, %struct.ompi_datatype_t*, %struct.ompi_op_t*, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define void @caller(float* %sendbuf, float* %dsendbuf, float* %recvbuf, float* %drecvbuf, i32 %count) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (float*, float*, i32)* @mpi_allreduce_max_test to i8*), float* %sendbuf, float* %dsendbuf, float* %recvbuf, float* %drecvbuf, i32 %count)

  ret void
}

declare void @__enzyme_autodiff(i8*, ...)

; CHECK: define internal void @diffempi_allreduce_max_test(float* %sendbuf, float* %"sendbuf'", float* %recvbuf, float* %"recvbuf'", i32 %count) #0 {
; CHECK-NEXT:   %1 = alloca i32
; CHECK-NEXT:   %2 = alloca i32
; CHECK-NEXT:   %3 = alloca i32
; CHECK-NEXT:   %"sendbuf.bc'ipc" = bitcast float* %"sendbuf'" to i8*
; CHECK-NEXT:   %sendbuf.bc = bitcast float* %sendbuf to i8*
; CHECK-NEXT:   %"recvbuf.bc'ipc" = bitcast float* %"recvbuf'" to i8*
; CHECK-NEXT:   %recvbuf.bc = bitcast float* %recvbuf to i8*
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %count, 4
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* bitcast (i8* (i64)* @malloc to i8* (i32)*)(i32 %mallocsize)
; CHECK-NEXT:   %4 = bitcast i8* %malloccall to i32*
; CHECK-NEXT:   %5 = bitcast i8* %sendbuf.bc to float*
; CHECK-NEXT:   %6 = bitcast i8* %recvbuf.bc to float*
; CHECK-NEXT:   %7 = bitcast i32* %1 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 4, i8* %7)
; CHECK-NEXT:   %8 = bitcast i32* %2 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 4, i8* %8)
; CHECK-NEXT:   %9 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float_int to i8*), i32* %1)
; CHECK-NEXT:   %10 = zext i32 %count to i64
; CHECK-NEXT:   %mallocsize.i = mul nuw nsw i64 %10, 8
; CHECK-NEXT:   %malloccall.i = call noalias nonnull i8* @malloc(i64 %mallocsize.i)
; CHECK-NEXT:   %11 = bitcast i8* %malloccall.i to { float, i32 }*
; CHECK-NEXT:   %mallocsize1.i = mul nuw nsw i64 %10, 8
; CHECK-NEXT:   %malloccall2.i = call noalias nonnull i8* @malloc(i64 %mallocsize1.i)
; CHECK-NEXT:   %12 = bitcast i8* %malloccall2.i to { float, i32 }*
; CHECK-NEXT:   %13 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %2)
; CHECK-NEXT:   %14 = load i32, i32* %2
; CHECK-NEXT:   %15 = icmp sgt i32 %count, 0
; CHECK-NEXT:   br i1 %15, label %loop.i, label %end.i

; CHECK: loop.i:                                           ; preds = %loop.i, %0
; CHECK-NEXT:   %16 = phi i32 [ 0, %0 ], [ %21, %loop.i ]
; CHECK-NEXT:   %17 = getelementptr inbounds { float, i32 }, { float, i32 }* %11, i32 %16, i32 0
; CHECK-NEXT:   %18 = getelementptr inbounds float, float* %5, i32 %16
; CHECK-NEXT:   %19 = load float, float* %18
; CHECK-NEXT:   store float %19, float* %17
; CHECK-NEXT:   %20 = getelementptr inbounds { float, i32 }, { float, i32 }* %11, i32 %16, i32 1
; CHECK-NEXT:   store i32 %14, i32* %20
; CHECK-NEXT:   %21 = add i32 %16, 1
; CHECK-NEXT:   %22 = icmp slt i32 %16, %count
; CHECK-NEXT:   br i1 %22, label %loop.i, label %end.i

; CHECK: end.i:                                            ; preds = %loop.i, %0
; CHECK-NEXT:   %23 = call i32 @MPI_Allreduce(i8* %malloccall.i, i8* %malloccall2.i, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float_int to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_maxloc to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %24 = icmp sgt i32 %count, 0
; CHECK-NEXT:   br i1 %24, label %loop3.i, label %__enzyme_mpi_allreduce_comploc_float.exit

; CHECK: loop3.i:                                          ; preds = %loop3.i, %end.i
; CHECK-NEXT:   %25 = phi i32 [ 0, %end.i ], [ %32, %loop3.i ]
; CHECK-NEXT:   %26 = getelementptr inbounds { float, i32 }, { float, i32 }* %12, i32 %25, i32 0
; CHECK-NEXT:   %27 = getelementptr inbounds float, float* %6, i32 %25
; CHECK-NEXT:   %28 = load float, float* %26
; CHECK-NEXT:   store float %28, float* %27
; CHECK-NEXT:   %29 = getelementptr inbounds { float, i32 }, { float, i32 }* %12, i32 %25, i32 1
; CHECK-NEXT:   %30 = getelementptr inbounds i32, i32* %4, i32 %25
; CHECK-NEXT:   %31 = load i32, i32* %29
; CHECK-NEXT:   store i32 %31, i32* %30
; CHECK-NEXT:   %32 = add i32 %25, 1
; CHECK-NEXT:   %33 = icmp slt i32 %25, %count
; CHECK-NEXT:   br i1 %33, label %loop3.i, label %__enzyme_mpi_allreduce_comploc_float.exit

; CHECK: __enzyme_mpi_allreduce_comploc_float.exit:        ; preds = %end.i, %loop3.i
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall.i)
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall2.i)
; CHECK-NEXT:   %34 = bitcast i32* %1 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 4, i8* %34)
; CHECK-NEXT:   %35 = bitcast i32* %2 to i8*
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 4, i8* %35)
; CHECK-NEXT:   br label %invert

; CHECK: invert:                                           ; preds = %__enzyme_mpi_allreduce_comploc_float.exit
; CHECK-NEXT:   %36 = zext i32 %count to i64
; CHECK-NEXT:   %37 = mul nuw i64 %36, 4
; CHECK-NEXT:   %38 = tail call noalias nonnull i8* @malloc(i64 %37)
; CHECK-NEXT:   %39 = call i32 @MPI_Allreduce(i8* %"recvbuf.bc'ipc", i8* %38, i32 %count, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_sum to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %40 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %3)
; CHECK-NEXT:   %41 = load i32, i32* %3
; CHECK-NEXT:   %42 = icmp sgt i32 %count, 0
; CHECK-NEXT:   br i1 %42, label %invert_loop, label %invert_endloop

; CHECK: invert_loop:                                      ; preds = %invert_endthen, %invert
; CHECK-NEXT:   %43 = phi i32 [ 0, %invert ], [ %54, %invert_endthen ]
; CHECK-NEXT:   %44 = getelementptr inbounds i32, i32* %4, i32 %43
; CHECK-NEXT:   %45 = load i32, i32* %44
; CHECK-NEXT:   %46 = icmp eq i32 %45, %41
; CHECK-NEXT:   br i1 %46, label %invert_then, label %invert_endthen

; CHECK: invert_then:                                      ; preds = %invert_loop
; CHECK-NEXT:   %47 = bitcast i8* %38 to float*
; CHECK-NEXT:   %48 = getelementptr inbounds float, float* %47, i32 %43
; CHECK-NEXT:   %49 = bitcast i8* %"sendbuf.bc'ipc" to float*
; CHECK-NEXT:   %50 = getelementptr inbounds float, float* %49, i32 %43
; CHECK-DAG:   %[[r0:.+]] = load float, float* %48
; CHECK-DAG:   %[[r1:.+]] = load float, float* %50
; CHECK-NEXT:   %53 = fadd fast float %[[r0]], %[[r1]]
; CHECK-NEXT:   store float %53, float* %50
; CHECK-NEXT:   br label %invert_endthen

; CHECK: invert_endthen:                                   ; preds = %invert_then, %invert_loop
; CHECK-NEXT:   %54 = add i32 %43, 1
; CHECK-NEXT:   %55 = icmp slt i32 %43, %count
; CHECK-NEXT:   br i1 %55, label %invert_loop, label %invert_endloop

; CHECK: invert_endloop:                                   ; preds = %invert_endthen, %invert
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"recvbuf.bc'ipc", i8 0, i64 %37, i1 false)
; CHECK-NEXT:   tail call void @free(i8* nonnull %38)
; CHECK-NEXT:   %56 = bitcast i32* %4 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %56)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


