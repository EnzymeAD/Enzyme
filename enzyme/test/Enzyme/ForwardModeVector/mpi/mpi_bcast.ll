; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(adce)" -enzyme-preopt=false -S | FileCheck %s

%struct.ompi_predefined_datatype_t = type opaque
%struct.ompi_predefined_communicator_t = type opaque
%struct.ompi_datatype_t = type opaque
%struct.ompi_communicator_t = type opaque

@random_datatype = external dso_local global %struct.ompi_predefined_datatype_t, align 1
@ompi_mpi_comm_world = external dso_local global %struct.ompi_predefined_communicator_t, align 1

define double @mpi_bcast_test(double %b) {
entry:
  %b.addr = alloca double, align 8
  store double %b, double* %b.addr, align 8
  %0 = bitcast double* %b.addr to i8*
  %call = call i32 @MPI_Bcast(i8* nonnull %0, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
  %1 = load double, double* %b.addr, align 8
  ret double %1
}

declare i32 @MPI_Bcast(i8*, i32, %struct.ompi_datatype_t*, i32, %struct.ompi_communicator_t*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define double @caller(double %x) local_unnamed_addr  {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @mpi_bcast_test to i8*), metadata !"enzyme_width", i64 2, double %x, double 1.0, double 1.0)
  ret double %call
}

declare double @__enzyme_fwddiff(i8*, ...)


; CHECK: define internal [2 x double] @fwddiffe2mpi_bcast_test(double %b, [2 x double] %"b'") #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"b.addr'ipa" = alloca double, align 8
; CHECK-NEXT:   %"b.addr'ipa2" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"b.addr'ipa2", align 8
; CHECK-NEXT:   %b.addr = alloca double, align 8
; CHECK-NEXT:   %0 = extractvalue [2 x double] %"b'", 0
; CHECK-NEXT:   store double %0, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   %1 = extractvalue [2 x double] %"b'", 1
; CHECK-NEXT:   store double %1, double* %"b.addr'ipa2", align 8
; CHECK-NEXT:   store double %b, double* %b.addr, align 8
; CHECK-NEXT:   %"'ipc" = bitcast double* %"b.addr'ipa" to i8*
; CHECK-NEXT:   %"'ipc3" = bitcast double* %"b.addr'ipa2" to i8*
; CHECK-NEXT:   %2 = bitcast double* %b.addr to i8*
; CHECK-NEXT:   %call = call i32 @MPI_Bcast(i8* nonnull %2, i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %3 = call i32 @MPI_Bcast(i8* %"'ipc", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %4 = call i32 @MPI_Bcast(i8* %"'ipc3", i32 1, %struct.ompi_datatype_t* bitcast (%struct.ompi_predefined_datatype_t* @random_datatype to %struct.ompi_datatype_t*), i32 0, %struct.ompi_communicator_t* bitcast (%struct.ompi_predefined_communicator_t* @ompi_mpi_comm_world to %struct.ompi_communicator_t*))
; CHECK-NEXT:   %"'ipl" = load double, double* %"b.addr'ipa", align 8
; CHECK-NEXT:   %5 = insertvalue [2 x double] undef, double %"'ipl", 0
; CHECK-NEXT:   %"'ipl4" = load double, double* %"b.addr'ipa2", align 8
; CHECK-NEXT:   %6 = insertvalue [2 x double] %5, double %"'ipl4", 1
; CHECK-NEXT:   ret [2 x double] %6
; CHECK-NEXT: }