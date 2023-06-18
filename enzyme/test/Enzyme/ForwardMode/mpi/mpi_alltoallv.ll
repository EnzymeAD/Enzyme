; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1

define dso_local void @mpi_alltoallv_test(float* noundef %0, i32* noundef %1, i32* noundef %2, float* noundef %3, i32* noundef %4, i32* noundef %5) #6 {
  %7 = bitcast float* %0 to i8*
  %8 = bitcast float* %3 to i8*
  %9 = tail call i32 @MPI_Alltoallv(i8* noundef %7, i32* noundef %1, i32* noundef %2, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* noundef %8, i32* noundef %4, i32* noundef %5, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* noundef bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Alltoallv(i8* noundef, i32* noundef, i32* noundef, %struct.ompi_datatype_t* noundef, i8* noundef, i32* noundef, i32* noundef, %struct.ompi_datatype_t* noundef, %struct.ompi_communicator_t* noundef) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
define void @caller(float* %sendbuf, float* %dsendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %drecvbuf, i32* %recvcounts, i32* %recvdispls) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_fwddiff(i8* noundef bitcast (void (float*, i32*, i32*, float*, i32*, i32*)* @mpi_alltoallv_test to i8*), float* noundef %sendbuf, float* noundef %dsendbuf, i32* noundef %sendcounts, i32* noundef %senddispls, float* noundef %recvbuf, float* noundef %drecvbuf, i32* noundef %recvcounts, i32* noundef %recvdispls)

  ret void
}

declare void @__enzyme_fwddiff(i8*, ...)


; CHECK: define internal void @fwddiffempi_alltoallv_test(float* noundef %0, float* %"'", i32* noundef %1, i32* noundef %2, float* noundef %3, float* %"'1", i32* noundef %4, i32* noundef %5) #0 {
; CHECK-NEXT:   %"'ipc" = bitcast float* %"'" to i8*
; CHECK-NEXT:   %7 = bitcast float* %0 to i8*
; CHECK-NEXT:   %"'ipc3" = bitcast float* %"'1" to i8*
; CHECK-NEXT:   %8 = bitcast float* %3 to i8*
; CHECK-NEXT:   %9 = tail call i32 @MPI_Alltoallv(i8* noundef %7, i32* noundef %1, i32* noundef %2, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* noundef %8, i32* noundef %4, i32* noundef %5, %struct.ompi_datatype_t* noundef bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* noundef bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #0
; CHECK-NEXT:   %10 = call i32 @MPI_Alltoallv(i8* %"'ipc", i32* %1, i32* %2, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %"'ipc3", i32* %4, i32* %5, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
