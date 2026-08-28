; RUN: if [ %llvmver -eq 15 ]; then %opt < %s %loadEnzyme -enzyme -opaque-pointers=1 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -opaque-pointers=1 -S | FileCheck %s; fi

; ModuleID = 'enzyme-repro.ll'
source_filename = "dot_enzyme.3df87ea89a38df43-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@RSMPI_DOUBLE = external local_unnamed_addr global ptr
@RSMPI_COMM_WORLD = external local_unnamed_addr global ptr
@RSMPI_COMM_SELF = external local_unnamed_addr global ptr

; Function Attrs: noinline nonlazybind sanitize_hwaddress uwtable
define hidden noundef "enzyme_type"="{[-1]:Float@double}" double @_ZN10dot_enzyme12dot_parallel17h7dfcd86d9e8c176bE(ptr noalias nocapture noundef readonly align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer}" %0, ptr noalias nocapture noundef nonnull readonly align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %1, i64 noundef "enzyme_type"="{[-1]:Integer}" %2, ptr noalias nocapture noundef nonnull readonly align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %3, i64 noundef "enzyme_type"="{[-1]:Integer}" %4, ptr noundef "enzyme_type"="{[0]:Pointer}" %5) unnamed_addr #1 personality ptr @rust_eh_personality {
  %7 = alloca double, align 8
  %8 = alloca double, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %8)
  %9 = alloca double, align 8
  store double 1.000, ptr %8, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %7)
  store double 0.000000e+00, ptr %7, align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !7)
  %10 = load ptr, ptr @RSMPI_DOUBLE, align 8, !noalias !10, !noundef !13
  %11 = load i64, ptr %0, align 8, !range !14, !alias.scope !15, !noalias !18, !noundef !13
  switch i64 %11, label %12 [
    i64 0, label %20
    i64 1, label %13
    i64 2, label %14
    i64 3, label %16
    i64 4, label %18
  ]

12:                                               ; preds = %6
  unreachable

13:                                               ; preds = %6
  br label %20

14:                                               ; preds = %6
  %15 = getelementptr inbounds { i64, ptr }, ptr %0, i64 0, i32 1
  br label %20

16:                                               ; preds = %6
  %17 = getelementptr inbounds { i64, ptr }, ptr %0, i64 0, i32 1
  br label %20

18:                                               ; preds = %6
  %19 = getelementptr inbounds { i64, ptr }, ptr %0, i64 0, i32 1
  br label %20

20:                                               ; preds = %18, %16, %14, %13, %6
  %21 = phi ptr [ %19, %18 ], [ %17, %16 ], [ %15, %14 ], [ @RSMPI_COMM_WORLD, %13 ], [ @RSMPI_COMM_SELF, %6 ]
  %22 = load ptr, ptr %21, align 8, !noalias !18, !noundef !13
  %23 = alloca i32, align 4
  ;%23 = call noundef i32 @MPI_Allreduce(ptr noundef nonnull %8, ptr noundef nonnull %7, i32 noundef 1, ptr noundef %10, ptr noundef %5, ptr noundef %22), !noalias !7
  %24 = load double, ptr %7, align 8, !noundef !13
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %7)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %8)
  ret double %24
}

; Function Attrs: nonlazybind sanitize_hwaddress uwtable
declare noundef i32 @MPI_Allreduce(ptr noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef, ptr noundef) unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #4

; Function Attrs: nounwind nonlazybind sanitize_hwaddress uwtable
declare noundef i32 @rust_eh_personality(i32 noundef, i32 noundef, i64, ptr noundef, ptr noundef) unnamed_addr #5

declare double @__enzyme_autodiff(...)

define double @enzyme_opt_helper_0(ptr %0, ptr %1, i64 %2, ptr %3, i64 %4, ptr %5) {
  %7 = call double (...) @__enzyme_autodiff(ptr @_ZN10dot_enzyme12dot_parallel17h7dfcd86d9e8c176bE, metadata !"enzyme_const", ptr %0, metadata !"enzyme_dup", ptr %1, ptr %1, metadata !"enzyme_const", i64 %2, metadata !"enzyme_dup", ptr %3, ptr %3, metadata !"enzyme_const", i64 %4, metadata !"enzyme_const", ptr %5)
  ret double %7
}

attributes #0 = { noinline nounwind nonlazybind sanitize_hwaddress uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #1 = { noinline nonlazybind sanitize_hwaddress uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #2 = { nonlazybind sanitize_hwaddress uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #5 = { nounwind nonlazybind sanitize_hwaddress uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.ident = !{!6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6, !6}
!llvm.dbg.cu = !{}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 2, !"RtLibUseGOT", i32 1}
!3 = !{i32 1, !"LTOPostLink", i32 1}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"rustc version 1.77.0-nightly (ecb2f9cdf 2024-07-30)"}
!7 = !{!8}
!8 = distinct !{!8, !9, !"_ZN3mpi10collective23CommunicatorCollectives15all_reduce_into17h5bd43ff3d0a82648E: argument 0"}
!9 = distinct !{!9, !"_ZN3mpi10collective23CommunicatorCollectives15all_reduce_into17h5bd43ff3d0a82648E"}
!10 = !{!8, !11, !12}
!11 = distinct !{!11, !9, !"_ZN3mpi10collective23CommunicatorCollectives15all_reduce_into17h5bd43ff3d0a82648E: argument 1"}
!12 = distinct !{!12, !9, !"_ZN3mpi10collective23CommunicatorCollectives15all_reduce_into17h5bd43ff3d0a82648E: argument 2"}
!13 = !{}
!14 = !{i64 0, i64 5}
!15 = !{!16, !8}
!16 = distinct !{!16, !17, !"_ZN69_$LT$mpi..topology..SimpleCommunicator$u20$as$u20$mpi..raw..AsRaw$GT$6as_raw17h5ddd9d255d268465E: argument 0"}
!17 = distinct !{!17, !"_ZN69_$LT$mpi..topology..SimpleCommunicator$u20$as$u20$mpi..raw..AsRaw$GT$6as_raw17h5ddd9d255d268465E"}
!18 = !{!11, !12}

; CHECK: define internal void @diffe_ZN10dot_enzyme12dot_parallel17h7dfcd86d9e8c176bE(ptr noalias nocapture noundef readonly align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer}" %0, ptr noalias nocapture noundef nonnull readonly align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %1, ptr nocapture align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %"'", i64 noundef "enzyme_type"="{[-1]:Integer}" %2, ptr noalias nocapture noundef nonnull readonly align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %3, ptr nocapture align 8 "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %"'1", i64 noundef "enzyme_type"="{[-1]:Integer}" %4, ptr noundef "enzyme_type"="{[0]:Pointer}" %5, double %differeturn)
