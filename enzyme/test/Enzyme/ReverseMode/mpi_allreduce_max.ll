; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_predefined_op_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque
%struct.ompi_op_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_op_max = external global %struct.ompi_predefined_op_t, align 1

define void @mpi_allreduce_max_test(float* %0, float* %1, i32 %2) {
  %4 = bitcast float* %0 to i8*
  %5 = bitcast float* %1 to i8*
  %6 = tail call i32 @MPI_Allreduce(i8* %4, i8* %5, i32 %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_max to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
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


; CHECK: define internal void @diffempi_allreduce_max_test(float* %0, float* %"'", float* %1, float* %"'1", i32 %2) #0 {
; CHECK-NEXT: invert:
; CHECK-NEXT:   %3 = alloca i32, align 4
; CHECK-NEXT:   %"'ipc" = bitcast float* %"'1" to i8*
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %2, 4
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* bitcast (i8* (i64)* @malloc to i8* (i32)*)(i32 %mallocsize)
; CHECK-NEXT:   %4 = bitcast i8* %malloccall to i32*
; CHECK-NEXT:   %5 = call i32 @__enzyme_mpi_allreduce_comploc_float(float* %0, float* %1, i32* %4, i32 %2, %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_maxloc to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %6 = zext i32 %2 to i64
; CHECK-NEXT:   %7 = mul nuw i64 %6, 4
; CHECK-NEXT:   %8 = tail call noalias nonnull i8* @malloc(i64 %7)
; CHECK-NEXT:   %9 = call i32 @MPI_Allreduce(i8* %"'ipc", i8* %8, i32 %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_op_t* bitcast (%struct.ompi_predefined_op_t* @ompi_mpi_op_sum to %struct.ompi_op_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %10 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* %3)
; CHECK-NEXT:   %11 = load i32, i32* %3, align 4
; CHECK-NEXT:   %12 = icmp sgt i32 %2, 0
; CHECK-NEXT:   br i1 %12, label %invert_loop, label %invert_endloop

; CHECK: invert_loop:                                      ; preds = %invert_endthen, %invert
; CHECK-NEXT:   %13 = phi i32 [ 0, %invert ], [ %23, %invert_endthen ]
; CHECK-NEXT:   %14 = getelementptr inbounds i32, i32* %4, i32 %13
; CHECK-NEXT:   %15 = load i32, i32* %14, align 4
; CHECK-NEXT:   %16 = icmp eq i32 %15, %11
; CHECK-NEXT:   br i1 %16, label %invert_then, label %invert_endthen

; CHECK: invert_then:                                      ; preds = %invert_loop
; CHECK-NEXT:   %17 = bitcast i8* %8 to float*
; CHECK-NEXT:   %18 = getelementptr inbounds float, float* %17, i32 %13
; CHECK-NEXT:   %19 = getelementptr inbounds float, float* %"'", i32 %13
; CHECK-NEXT:   %20 = load float, float* %18, align 4
; CHECK-NEXT:   %21 = load float, float* %19, align 4
; CHECK-NEXT:   %22 = fadd fast float %20, %21
; CHECK-NEXT:   store float %22, float* %19, align 4
; CHECK-NEXT:   br label %invert_endthen

; CHECK: invert_endthen:                                   ; preds = %invert_then, %invert_loop
; CHECK-NEXT:   %23 = add i32 %13, 1
; CHECK-NEXT:   %24 = icmp slt i32 %13, %2
; CHECK-NEXT:   br i1 %24, label %invert_loop, label %invert_endloop

; CHECK: invert_endloop:                                   ; preds = %invert_endthen, %invert
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %"'ipc", i8 0, i64 %7, i1 false)
; CHECK-NEXT:   tail call void @free(i8* nonnull %8)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: ; Function Attrs: nounwind
; CHECK-NEXT: define internal i32 @__enzyme_mpi_allreduce_comploc_float(float* nocapture %0, float* nocapture %1, i32* nocapture %2, i32 %3, %struct.ompi_op_t* nocapture %4, %struct.ompi_communicator_t* nocapture %5) #1 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %6 = alloca i32, align 4
; CHECK-NEXT:   %7 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float_int to i8*), i32* %6) #4
; CHECK-NEXT:   %8 = zext i32 %3 to i64
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %8, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %9 = bitcast i8* %malloccall to { float, i32 }*
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i64 %8, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:   %10 = bitcast i8* %malloccall2 to { float, i32 }*
; CHECK-NEXT:   %11 = alloca i32, align 4
; CHECK-NEXT:   %12 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* %5, i32* %11)
; CHECK-NEXT:   %13 = load i32, i32* %11, align 4
; CHECK-NEXT:   %14 = icmp sgt i32 %3, 0
; CHECK-NEXT:   br i1 %14, label %loop, label %end

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %15 = phi i32 [ 0, %entry ], [ %20, %loop ]
; CHECK-NEXT:   %16 = getelementptr inbounds { float, i32 }, { float, i32 }* %9, i32 %15, i32 0
; CHECK-NEXT:   %17 = getelementptr inbounds float, float* %0, i32 %15
; CHECK-NEXT:   %18 = load float, float* %17, align 4
; CHECK-NEXT:   store float %18, float* %16, align 4
; CHECK-NEXT:   %19 = getelementptr inbounds { float, i32 }, { float, i32 }* %9, i32 %15, i32 1
; CHECK-NEXT:   store i32 %13, i32* %19, align 4
; CHECK-NEXT:   %20 = add i32 %15, 1
; CHECK-NEXT:   %21 = icmp slt i32 %15, %3
; CHECK-NEXT:   br i1 %21, label %loop, label %end

; CHECK: end:                                              ; preds = %loop, %entry
; CHECK-NEXT:   %22 = call i32 @MPI_Allreduce(i8* %malloccall, i8* %malloccall2, i32 %3, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float_int to %struct.ompi_datatype_t*), %struct.ompi_op_t* %4, %struct.ompi_communicator_t* %5)
; CHECK-NEXT:   %23 = icmp sgt i32 %3, 0
; CHECK-NEXT:   br i1 %23, label %loop3, label %end4

; CHECK: loop3:                                            ; preds = %loop3, %end
; CHECK-NEXT:   %24 = phi i32 [ 0, %end ], [ %31, %loop3 ]
; CHECK-NEXT:   %25 = getelementptr inbounds { float, i32 }, { float, i32 }* %10, i32 %24, i32 0
; CHECK-NEXT:   %26 = getelementptr inbounds float, float* %1, i32 %24
; CHECK-NEXT:   %27 = load float, float* %25, align 4
; CHECK-NEXT:   store float %27, float* %26, align 4
; CHECK-NEXT:   %28 = getelementptr inbounds { float, i32 }, { float, i32 }* %10, i32 %24, i32 1
; CHECK-NEXT:   %29 = getelementptr inbounds i32, i32* %2, i32 %24
; CHECK-NEXT:   %30 = load i32, i32* %28, align 4
; CHECK-NEXT:   store i32 %30, i32* %29, align 4
; CHECK-NEXT:   %31 = add i32 %24, 1
; CHECK-NEXT:   %32 = icmp slt i32 %24, %3
; CHECK-NEXT:   br i1 %32, label %loop3, label %end4

; CHECK: end4:                                             ; preds = %loop3, %end
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall2)
; CHECK-NEXT:   ret i32 %22
; CHECK-NEXT: }
