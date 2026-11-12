; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck %s; fi

declare void @_ZN6Kokkos4Impl29hostspace_parallel_zeromemsetEPvm(ptr, i64)
declare void @_ZN6Kokkos4Impl27hostspace_parallel_deepcopyEPvPKvl(ptr, ptr, i64)
declare void @_ZN6Kokkos5fenceERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(ptr)
declare void @_ZNK6Kokkos4Impl20HostThreadTeamMemberINS_6SerialEE12team_barrierEv(ptr)
declare void @_ZN6Kokkos4Impl11HostBarrier29impl_backoff_wait_until_equalEPiib(ptr, i32, i1)
declare void @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define double @kokkos_deepcopy_ops(ptr %dst, ptr %src, ptr %label, ptr %team, ptr %bar) {
entry:
  call void @_ZN6Kokkos4Impl29hostspace_parallel_zeromemsetEPvm(ptr %dst, i64 8)
  %x = load double, ptr %src, align 8
  %sq = fmul double %x, %x
  store double %sq, ptr %src, align 8
  call void @_ZN6Kokkos4Impl27hostspace_parallel_deepcopyEPvPKvl(ptr %dst, ptr %src, i64 8)
  %step = load i32, ptr %bar, align 4
  %next_step = add nsw i32 %step, 1
  call void @_ZN6Kokkos4Impl11HostBarrier29impl_backoff_wait_until_equalEPiib(ptr %bar, i32 %next_step, i1 false)
  call void @_ZNK6Kokkos4Impl20HostThreadTeamMemberINS_6SerialEE12team_barrierEv(ptr %team)
  call void @_ZN6Kokkos5fenceERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(ptr %label)
  %res = load double, ptr %dst, align 8
  ret double %res
}

define void @d_deepcopy_rev(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc, ptr %label, ptr %team, ptr %bar) {
entry:
  call void (...) @__enzyme_autodiff(ptr @kokkos_deepcopy_ops, metadata !"enzyme_dup", ptr %dst, ptr %ddst, metadata !"enzyme_dup", ptr %src, ptr %dsrc, metadata !"enzyme_const", ptr %label, metadata !"enzyme_const", ptr %team, metadata !"enzyme_const", ptr %bar)
  ret void
}

define double @d_deepcopy_fwd(ptr %dst, ptr %ddst, ptr %src, ptr %dsrc, ptr %label, ptr %team, ptr %bar) {
entry:
  %dr = call double (...) @__enzyme_fwddiff(ptr @kokkos_deepcopy_ops, metadata !"enzyme_dup", ptr %dst, ptr %ddst, metadata !"enzyme_dup", ptr %src, ptr %dsrc, metadata !"enzyme_const", ptr %label, metadata !"enzyme_const", ptr %team, metadata !"enzyme_const", ptr %bar)
  ret double %dr
}

; CHECK-LABEL: define internal void @diffekokkos_deepcopy_ops(
; CHECK: call void @_ZN6Kokkos4Impl29hostspace_parallel_zeromemsetEPvm(ptr %dst, i64 8)
; CHECK: call void @llvm.memset.p0.i64(ptr {{.*}}%"dst'", i8 0, i64 8, i1 false)
; CHECK: call void @_ZN6Kokkos4Impl27hostspace_parallel_deepcopyEPvPKvl(ptr %dst, ptr %src, i64 8)
; CHECK: fence seq_cst
; CHECK: call void @_ZNK6Kokkos4Impl20HostThreadTeamMemberINS_6SerialEE12team_barrierEv(ptr %team)
; CHECK: call void @llvm.memset.p0.i64(ptr {{.*}}%"dst'", i8 0, i64 8, i1 false)
; CHECK: ret void

; CHECK-LABEL: define internal double @fwddiffekokkos_deepcopy_ops(
; CHECK: call void @llvm.memset.p0.i64(ptr {{.*}}%"dst'", i8 0, i64 8, i1 false)
; CHECK: call void @_ZN6Kokkos4Impl27hostspace_parallel_deepcopyEPvPKvl(ptr %"dst'", ptr %"src'", i64 8)
; CHECK: ret double
