; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@ompi_mpi_float = external global %struct.ompi_predefined_datatype_t, align 1

define void @mpi_alltoallv_test(float* %sendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, i32* %recvcounts, i32* %recvdispls) {
  %sendbuf.bc = bitcast float* %sendbuf to i8*
  %recvbuf.bc = bitcast float* %recvbuf to i8*
  %result = tail call i32 @MPI_Alltoallv(i8* %sendbuf.bc, i32* %sendcounts, i32* %senddispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %recvbuf.bc, i32* %recvcounts, i32* %recvdispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  ret void
}

declare i32 @MPI_Alltoallv(i8*, i32*, i32*, %struct.ompi_datatype_t*, i8*, i32*, i32*, %struct.ompi_datatype_t*, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define void @caller(float* %sendbuf, float* %dsendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %drecvbuf, i32* %recvcounts, i32* %recvdispls) local_unnamed_addr  {
entry:
  call void (i8*, ...) @__enzyme_fwddiff(i8* bitcast (void (float*, i32*, i32*, float*, i32*, i32*)* @mpi_alltoallv_test to i8*), float* %sendbuf, float* %dsendbuf, i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %drecvbuf, i32* %recvcounts, i32* %recvdispls)

  ret void
}

declare void @__enzyme_fwddiff(i8*, ...)


; CHECK: define internal void @fwddiffempi_alltoallv_test(float* %sendbuf, float* %"sendbuf'", i32* %sendcounts, i32* %senddispls, float* %recvbuf, float* %"recvbuf'", i32* %recvcounts, i32* %recvdispls) #0 {
; CHECK-NEXT:   %"sendbuf.bc'ipc" = bitcast float* %"sendbuf'" to i8*
; CHECK-NEXT:   %sendbuf.bc = bitcast float* %sendbuf to i8*
; CHECK-NEXT:   %"recvbuf.bc'ipc" = bitcast float* %"recvbuf'" to i8*
; CHECK-NEXT:   %recvbuf.bc = bitcast float* %recvbuf to i8*
; CHECK-NEXT:   %result = tail call i32 @MPI_Alltoallv(i8* %sendbuf.bc, i32* %sendcounts, i32* %senddispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %recvbuf.bc, i32* %recvcounts, i32* %recvdispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #0
; CHECK-NEXT:   %1 = call i32 @MPI_Alltoallv(i8* %"sendbuf.bc'ipc", i32* %sendcounts, i32* %senddispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), i8* %"recvbuf.bc'ipc", i32* %recvcounts, i32* %recvdispls, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_float to %struct.ompi_datatype_t*), %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   ret void
; CHECK-NEXT: }



