; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; ModuleID = 'test/mpi2.c'
source_filename = "test/mpi2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_request_t = type opaque
%struct.ompi_status_public_t = type { i32, i32, i32, i32, i64 }
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@ompi_mpi_real = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1
@.str = private unnamed_addr constant [33 x i8] c"Process %d: vald %f, valeurd %f\0A\00", align 1
@.str.1 = private unnamed_addr constant [31 x i8] c"Process %d: val %f, valeur %f\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local void @msg1(float* %val1, float* %val2, i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette) #0 {
entry:
  %r1 = alloca %struct.ompi_request_t*, align 8
  %s1 = alloca %struct.ompi_status_public_t, align 8
  %r2 = alloca %struct.ompi_request_t*, align 8
  %s2 = alloca %struct.ompi_status_public_t, align 8
  %0 = bitcast %struct.ompi_request_t** %r1 to i8*
  %1 = bitcast float* %val1 to i8*
  %call = call i32 @MPI_Isend(i8* %1, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r1) #4
  %2 = bitcast %struct.ompi_status_public_t* %s1 to i8*
  %call1 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r1, %struct.ompi_status_public_t* nonnull %s1) #4
  %3 = bitcast %struct.ompi_request_t** %r2 to i8*
  %4 = bitcast float* %val2 to i8*
  %call2 = call i32 @MPI_Irecv(i8* %4, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r2) #4
  %5 = bitcast %struct.ompi_status_public_t* %s2 to i8*
  %call3 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r2, %struct.ompi_status_public_t* nonnull %s2) #4
  ret void
}

declare dso_local i32 @MPI_Isend(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**)

declare dso_local i32 @MPI_Wait(%struct.ompi_request_t**, %struct.ompi_status_public_t*) 

declare dso_local i32 @MPI_Irecv(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_request_t**)

; Function Attrs: nounwind uwtable
define dso_local void @msg2(float* %val1, float* %val2, i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette) #0 {
entry:
  %statut = alloca %struct.ompi_status_public_t, align 8
  %0 = bitcast %struct.ompi_status_public_t* %statut to i8*
  %1 = bitcast float* %val2 to i8*
  %call = call i32 @MPI_Recv(i8* %1, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_status_public_t* nonnull %statut) #4
  %2 = bitcast float* %val1 to i8*
  %call1 = call i32 @MPI_Send(i8* %2, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*)) #4
  ret void
}

declare dso_local i32 @MPI_Recv(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*, %struct.ompi_status_public_t*)

declare dso_local i32 @MPI_Send(i8*, i32, %struct.ompi_datatype_t*, i32, i32, %struct.ompi_communicator_t*)

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) local_unnamed_addr #0 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %nb_processus = alloca i32, align 4
  %rang = alloca i32, align 4
  %val = alloca float, align 4
  %valeur = alloca float, align 4
  %vald = alloca float, align 4
  %valeurd = alloca float, align 4
  store i32 %argc, i32* %argc.addr, align 4, !tbaa !2
  store i8** %argv, i8*** %argv.addr, align 8, !tbaa !6
  %0 = bitcast i32* %nb_processus to i8*
  %1 = bitcast i32* %rang to i8*
  %2 = bitcast float* %val to i8*
  %3 = bitcast float* %valeur to i8*
  %call = call i32 @MPI_Init(i32* nonnull %argc.addr, i8*** nonnull %argv.addr) #4
  %call1 = call i32 @MPI_Comm_rank(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* nonnull %rang) #4
  %call2 = call i32 @MPI_Comm_size(%struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), i32* nonnull %nb_processus) #4
  %4 = load i32, i32* %nb_processus, align 4, !tbaa !2
  %5 = load i32, i32* %rang, align 4, !tbaa !2
  %add = add i32 %4, -1
  %sub = add i32 %add, %5
  %rem = srem i32 %sub, %4
  %add3 = add nsw i32 %5, 1
  %rem4 = srem i32 %add3, %4
  %add5 = add nsw i32 %5, 1000
  %conv = sitofp i32 %add5 to float
  store float %conv, float* %val, align 4, !tbaa !8
  %6 = bitcast float* %vald to i8*
  %7 = bitcast float* %valeurd to i8*
  %add6 = add nsw i32 %5, 2000
  %conv7 = sitofp i32 %add6 to float
  store float %conv7, float* %valeurd, align 4, !tbaa !8
  %cmp = icmp eq i32 %5, 0
  %.sink = select i1 %cmp, i8* bitcast (void (float*, float*, i32, i32, i32)* @msg1 to i8*), i8* bitcast (void (float*, float*, i32, i32, i32)* @msg2 to i8*)
  call void (i8*, ...) @__enzyme_autodiff(i8* %.sink, float* nonnull %val, float* nonnull %vald, float* nonnull %valeur, float* nonnull %valeurd, i32 %rem, i32 %rem4, i32 100) #4
  %8 = load i32, i32* %rang, align 4, !tbaa !2
  %9 = load float, float* %vald, align 4, !tbaa !8
  %conv9 = fpext float %9 to double
  %10 = load float, float* %valeurd, align 4, !tbaa !8
  %conv10 = fpext float %10 to double
  %call11 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([33 x i8], [33 x i8]* @.str, i64 0, i64 0), i32 %8, double %conv9, double %conv10)
  %11 = load i32, i32* %rang, align 4, !tbaa !2
  %12 = load float, float* %val, align 4, !tbaa !8
  %conv12 = fpext float %12 to double
  %13 = load float, float* %valeur, align 4, !tbaa !8
  %conv13 = fpext float %13 to double
  %call14 = call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([31 x i8], [31 x i8]* @.str.1, i64 0, i64 0), i32 %11, double %conv12, double %conv13)
  %call15 = call i32 @MPI_Finalize() #4
  ret i32 0
}

declare dso_local i32 @MPI_Init(i32*, i8***)

declare dso_local i32 @MPI_Comm_rank(%struct.ompi_communicator_t*, i32*)

declare dso_local i32 @MPI_Comm_size(%struct.ompi_communicator_t*, i32*)

declare dso_local void @__enzyme_autodiff(i8*, ...) 

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) 

declare dso_local i32 @MPI_Finalize() 

attributes #0 = { nounwind uwtable }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (git@github.com:llvm/llvm-project ef32c611aa214dea855364efd7ba451ec5ec3f74)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !4, i64 0}


; CHECK: define internal void @diffemsg1(float* %val1, float* %"val1'", float* %val2, float* %"val2'", i32 %numprocprec, i32 %numprocsuiv, i32 %etiquette)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca i32
; CHECK-NEXT:   %1 = alloca %struct.ompi_status_public_t
; CHECK-NEXT:   %2 = alloca i32
; CHECK-NEXT:   %3 = alloca i32
; CHECK-NEXT:   %4 = alloca %struct.ompi_status_public_t
; CHECK-NEXT:   %"r1'ipa" = alloca %struct.ompi_request_t*
; CHECK-NEXT:   store %struct.ompi_request_t* null, %struct.ompi_request_t** %"r1'ipa"
; CHECK-NEXT:   %r1 = alloca %struct.ompi_request_t*
; CHECK-NEXT:   %s1 = alloca %struct.ompi_status_public_t
; CHECK-NEXT:   %"r2'ipa" = alloca %struct.ompi_request_t*
; CHECK-NEXT:   store %struct.ompi_request_t* null, %struct.ompi_request_t** %"r2'ipa"
; CHECK-NEXT:   %r2 = alloca %struct.ompi_request_t*
; CHECK-NEXT:   %s2 = alloca %struct.ompi_status_public_t
; CHECK-NEXT:   %5 = bitcast float* %val1 to i8*
; CHECK-NEXT:   %malloccall3 = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) i8* @malloc(i64 64)
; CHECK-NEXT:   %6 = bitcast i8* %malloccall3 to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*
; CHECK-NEXT:   %7 = bitcast %struct.ompi_request_t** %"r1'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }**
; CHECK-NEXT:   %8 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %7
; CHECK-DAG:    %[[a9:.+]] = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 7
; CHECK-DAG:    %[[a10:.+]] = bitcast { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %8 to i8*
; CHECK-NEXT:   store i8* %[[a10]], i8** %[[a9]]
; CHECK-NEXT:   store { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %7
; CHECK-NEXT:   %11 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i32* %2)
; CHECK-NEXT:   %12 = load i32, i32* %2
; CHECK-NEXT:   %13 = zext i32 %12 to i64
; CHECK-NEXT:   %malloccall4 = tail call noalias nonnull i8* @malloc(i64 %13)
; CHECK-NEXT:   %14 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 0
; CHECK-NEXT:   store i8* %malloccall4, i8** %14
; CHECK-NEXT:   %15 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 1
; CHECK-NEXT:   store i64 1, i64* %15
; CHECK-NEXT:   %16 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 2
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i8** %16
; CHECK-DAG:    %[[a17:.+]] = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 3
; CHECK-DAG:    %[[a18:.+]] = zext i32 %numprocsuiv to i64
; CHECK-NEXT:   store i64 %[[a18]], i64* %[[a17]]
; CHECK-DAG:    %[[a19:.+]] = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 4
; CHECK-DAG:    %[[a20:.+]] = zext i32 %etiquette to i64
; CHECK-NEXT:   store i64 %[[a20]], i64* %[[a19]]
; CHECK-NEXT:   %21 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 5
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to i8*), i8** %21
; CHECK-NEXT:   %22 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %6, i64 0, i32 6
; CHECK-NEXT:   store i8 1, i8* %22
; CHECK-NEXT:   %call = call i32 @MPI_Isend(i8* %5, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocsuiv, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r1)
; CHECK-NEXT:   %23 = bitcast %struct.ompi_request_t** %"r1'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }**
; CHECK-NEXT:   %24 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %23
; CHECK-NEXT:   %call1 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r1, %struct.ompi_status_public_t* nonnull %s1)
; CHECK-NEXT:   %"'ipc" = bitcast float* %"val2'" to i8*
; CHECK-NEXT:   %25 = bitcast float* %val2 to i8*
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(64) dereferenceable_or_null(64) i8* @malloc(i64 64)
; CHECK-NEXT:   %26 = bitcast i8* %malloccall to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*
; CHECK-NEXT:   %27 = bitcast %struct.ompi_request_t** %"r2'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }**
; CHECK-NEXT:   %28 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %27
; CHECK-DAG:    %[[a29:.+]] = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 7
; CHECK-DAG:    %[[a30:.+]] = bitcast { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %28 to i8*
; CHECK-NEXT:   store i8* %[[a30]], i8** %[[a29]]
; CHECK-NEXT:   store { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %27
; CHECK-NEXT:   %31 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 0
; CHECK-NEXT:   store i8* %"'ipc", i8** %31
; CHECK-NEXT:   %32 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 1
; CHECK-NEXT:   store i64 1, i64* %32
; CHECK-NEXT:   %33 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 2
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i8** %33
; CHECK-DAG:    %[[a34:.+]] = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 3
; CHECK-DAG:    %[[a35:.+]] = zext i32 %numprocprec to i64
; CHECK-NEXT:   store i64 %[[a35]], i64* %[[a34]]
; CHECK-DAG:    %[[a36:.+]] = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 4
; CHECK-DAG:    %[[a37:.+]] = zext i32 %etiquette to i64
; CHECK-NEXT:   store i64 %[[a37]], i64* %[[a36]]
; CHECK-NEXT:   %38 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 5
; CHECK-NEXT:   store i8* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to i8*), i8** %38
; CHECK-NEXT:   %39 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %26, i64 0, i32 6
; CHECK-NEXT:   store i8 2, i8* %39
; CHECK-NEXT:   %call2 = call i32 @MPI_Irecv(i8* %25, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to %struct.ompi_datatype_t*), i32 %numprocprec, i32 %etiquette, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*), %struct.ompi_request_t** nonnull %r2)
; CHECK-NEXT:   %40 = bitcast %struct.ompi_request_t** %"r2'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }**
; CHECK-NEXT:   %41 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %40
; CHECK-NEXT:   %call3 = call i32 @MPI_Wait(%struct.ompi_request_t** nonnull %r2, %struct.ompi_status_public_t* nonnull %s2)
; CHECK-NEXT:   %42 = icmp eq { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %41, null
; CHECK-NEXT:   br i1 %42, label %invertentry_end, label %invertentry_nonnull

; CHECK: invertentry_nonnull:                              ; preds = %entry
; CHECK-NEXT:   %43 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %41
; CHECK-NEXT:   %44 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %43, 0
; CHECK-NEXT:   %45 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %43, 1
; CHECK-NEXT:   %46 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %43, 2
; CHECK-NEXT:   %47 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %43, 3
; CHECK-NEXT:   %48 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %43, 4
; CHECK-NEXT:   %49 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %43, 5
; CHECK-NEXT:   %50 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %43, 6
; CHECK-NEXT:   %51 = trunc i64 %45 to i32
; CHECK-NEXT:   %52 = bitcast i8* %46 to %struct.ompi_datatype_t*
; CHECK-NEXT:   %53 = trunc i64 %47 to i32
; CHECK-NEXT:   %54 = trunc i64 %48 to i32
; CHECK-NEXT:   %55 = bitcast i8* %49 to %struct.ompi_communicator_t*
; CHECK-NEXT:   %56 = icmp eq i8 %50, 1
; CHECK-NEXT:   br i1 %56, label %invertISend.i, label %invertIRecv.i

; CHECK: invertISend.i:                                    ; preds = %invertentry_nonnull
; CHECK-NEXT:   %57 = call i32 @MPI_Irecv(i8* %44, i32 %51, %struct.ompi_datatype_t* %52, i32 %53, i32 %54, %struct.ompi_communicator_t* %55, %struct.ompi_request_t** %r2)
; CHECK-NEXT:   br label %invertentry_end

; CHECK: invertIRecv.i:                                    ; preds = %invertentry_nonnull
; CHECK-NEXT:   %58 = call i32 @MPI_Isend(i8* %44, i32 %51, %struct.ompi_datatype_t* %52, i32 %53, i32 %54, %struct.ompi_communicator_t* %55, %struct.ompi_request_t** %r2)
; CHECK-NEXT:   br label %invertentry_end

; CHECK: invertentry_end:                                  ; preds = %invertISend.i, %invertIRecv.i, %entry
; CHECK-NEXT:   %59 = bitcast %struct.ompi_request_t** %"r2'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }**
; CHECK-NEXT:   %60 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %59
; CHECK-NEXT:   %61 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %60, i64 0, i32 0
; CHECK-NEXT:   %62 = load i8*, i8** %61
; CHECK-NEXT:   %63 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %60, i64 0, i32 7
; CHECK-NEXT:   %64 = load i8*, i8** %63
; CHECK-NEXT:   %65 = bitcast %struct.ompi_request_t** %"r2'ipa" to i8**
; CHECK-NEXT:   store i8* %64, i8** %65
; CHECK-NEXT:   %66 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i32* %0)
; CHECK-NEXT:   %67 = load i32, i32* %0
; CHECK-NEXT:   %68 = call i32 @MPI_Wait(%struct.ompi_request_t** %r2, %struct.ompi_status_public_t* %1)
; CHECK-NEXT:   %69 = zext i32 %67 to i64
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull %62, i8 0, i64 %69, i1 false)
; CHECK-NEXT:   %70 = bitcast { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %60 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %70)
; CHECK-NEXT:   %71 = icmp eq { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %24, null
; CHECK-NEXT:   br i1 %71, label %invertentry_end_end, label %invertentry_end_nonnull

; CHECK: invertentry_end_nonnull:                          ; preds = %invertentry_end
; CHECK-NEXT:   %72 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %24
; CHECK-NEXT:   %73 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %72, 0
; CHECK-NEXT:   %74 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %72, 1
; CHECK-NEXT:   %75 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %72, 2
; CHECK-NEXT:   %76 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %72, 3
; CHECK-NEXT:   %77 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %72, 4
; CHECK-NEXT:   %78 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %72, 5
; CHECK-NEXT:   %79 = extractvalue { i8*, i64, i8*, i64, i64, i8*, i8, i8* } %72, 6
; CHECK-NEXT:   %80 = trunc i64 %74 to i32
; CHECK-NEXT:   %81 = bitcast i8* %75 to %struct.ompi_datatype_t*
; CHECK-NEXT:   %82 = trunc i64 %76 to i32
; CHECK-NEXT:   %83 = trunc i64 %77 to i32
; CHECK-NEXT:   %84 = bitcast i8* %78 to %struct.ompi_communicator_t*
; CHECK-NEXT:   %85 = icmp eq i8 %79, 1
; CHECK-NEXT:   br i1 %85, label %invertISend.i7, label %invertIRecv.i8

; CHECK: invertISend.i7:                                   ; preds = %invertentry_end_nonnull
; CHECK-NEXT:   %86 = call i32 @MPI_Irecv(i8* %73, i32 %80, %struct.ompi_datatype_t* %81, i32 %82, i32 %83, %struct.ompi_communicator_t* %84, %struct.ompi_request_t** %r1)
; CHECK-NEXT:   br label %invertentry_end_end

; CHECK: invertIRecv.i8:                                   ; preds = %invertentry_end_nonnull
; CHECK-NEXT:   %87 = call i32 @MPI_Isend(i8* %73, i32 %80, %struct.ompi_datatype_t* %81, i32 %82, i32 %83, %struct.ompi_communicator_t* %84, %struct.ompi_request_t** %r1)
; CHECK-NEXT:   br label %invertentry_end_end

; CHECK: invertentry_end_end:                              ; preds = %invertISend.i7, %invertIRecv.i8, %invertentry_end
; CHECK-NEXT:   %88 = bitcast %struct.ompi_request_t** %"r1'ipa" to { i8*, i64, i8*, i64, i64, i8*, i8, i8* }**
; CHECK-NEXT:   %89 = load { i8*, i64, i8*, i64, i64, i8*, i8, i8* }*, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }** %88
; CHECK-NEXT:   %90 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %89, i64 0, i32 0
; CHECK-NEXT:   %91 = load i8*, i8** %90
; CHECK-NEXT:   %92 = getelementptr inbounds { i8*, i64, i8*, i64, i64, i8*, i8, i8* }, { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %89, i64 0, i32 7
; CHECK-NEXT:   %93 = load i8*, i8** %92
; CHECK-NEXT:   %94 = bitcast %struct.ompi_request_t** %"r1'ipa" to i8**
; CHECK-NEXT:   store i8* %93, i8** %94
; CHECK-NEXT:   %95 = call i32 @MPI_Type_size(i8* bitcast (%struct.ompi_predefined_datatype_t* @ompi_mpi_real to i8*), i32* %3)
; CHECK-NEXT:   %96 = load i32, i32* %3
; CHECK-NEXT:   %97 = call i32 @MPI_Wait(%struct.ompi_request_t** %r1, %struct.ompi_status_public_t* %4)
; CHECK-NEXT:   %98 = zext i32 %96 to i64
; CHECK-NEXT:   %99 = bitcast i8* %91 to float*
; CHECK-NEXT:   %100 = udiv i64 %98, 4
; CHECK-NEXT:   %101 = icmp eq i64 %100, 0
; CHECK-NEXT:   br i1 %101, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invertentry_end_end
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invertentry_end_end ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %99, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %"val1'", i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i
; CHECK-NEXT:   %102 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %102, float* %src.i.i
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %103 = icmp eq i64 %100, %idx.next.i
; CHECK-NEXT:   br i1 %103, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %invertentry_end_end, %for.body.i
; CHECK-NEXT:   tail call void @free(i8* nonnull %91)
; CHECK-NEXT:   %104 = bitcast { i8*, i64, i8*, i64, i64, i8*, i8, i8* }* %89 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %104)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_differential_mpi_wait(i8* %buf, i64 %count, i8* %datatype, i64 %source, i64 %tag, i8* %comm, i8 %fn, %struct.ompi_request_t** %d_req)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = trunc i64 %count to i32
; CHECK-NEXT:   %1 = bitcast i8* %datatype to %struct.ompi_datatype_t*
; CHECK-NEXT:   %2 = trunc i64 %source to i32
; CHECK-NEXT:   %3 = trunc i64 %tag to i32
; CHECK-NEXT:   %4 = bitcast i8* %comm to %struct.ompi_communicator_t*
; CHECK-NEXT:   %5 = icmp eq i8 %fn, 1
; CHECK-NEXT:   br i1 %5, label %invertISend, label %invertIRecv

; CHECK: invertISend:                                      ; preds = %entry
; CHECK-NEXT:   %6 = call i32 @MPI_Irecv(i8* %buf, i32 %0, %struct.ompi_datatype_t* %1, i32 %2, i32 %3, %struct.ompi_communicator_t* %4, %struct.ompi_request_t** %d_req)

; CHECK: invertIRecv:                                      ; preds = %entry
; CHECK-NEXT:   %7 = call i32 @MPI_Isend(i8* %buf, i32 %0, %struct.ompi_datatype_t* %1, i32 %2, i32 %3, %struct.ompi_communicator_t* %4, %struct.ompi_request_t** %d_req)
