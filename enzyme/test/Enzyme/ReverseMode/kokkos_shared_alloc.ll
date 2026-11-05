; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck %s; fi

%"struct.Kokkos::Impl::SharedAllocationTracker" = type { i64 }

declare ptr @_ZN6Kokkos4Impl29make_shared_allocation_recordINS_9HostSpaceEvEEPNS0_22SharedAllocationRecordIvvEERKS2_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm(ptr, ptr, i64)
declare ptr @_ZNK6Kokkos4Impl22SharedAllocationRecordIvvE4dataEv(ptr)
declare void @_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9incrementEPS2_(ptr)
declare ptr @_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9decrementEPS2_(ptr)
declare void @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define double @kokkos_view_lifecycle(double %x, ptr %space, ptr %label) {
entry:
  %trk1 = alloca %"struct.Kokkos::Impl::SharedAllocationTracker", align 8
  %trk2 = alloca %"struct.Kokkos::Impl::SharedAllocationTracker", align 8
  ; Default-construct tracker with DO_NOT_DEREF_FLAG (1)
  store i64 1, ptr %trk1, align 8
  store i64 1, ptr %trk2, align 8
  %rec = call ptr @_ZN6Kokkos4Impl29make_shared_allocation_recordINS_9HostSpaceEvEEPNS0_22SharedAllocationRecordIvvEERKS2_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm(ptr %space, ptr %label, i64 8)
  %buf = call ptr @_ZNK6Kokkos4Impl22SharedAllocationRecordIvvE4dataEv(ptr %rec)
  %rec_int = ptrtoint ptr %rec to i64
  store i64 %rec_int, ptr %trk1, align 8
  ; Copy-assign tracker: load record bits, increment, store into trk2, then clear trk1 to sentinel 1
  %bits = load i64, ptr %trk1, align 8
  %rec_copy = inttoptr i64 %bits to ptr
  call void @_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9incrementEPS2_(ptr %rec_copy)
  store i64 %bits, ptr %trk2, align 8
  store i64 1, ptr %trk1, align 8
  call ptr @_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9decrementEPS2_(ptr %rec)
  %sq = fmul double %x, %x
  store double %sq, ptr %buf, align 8
  %res = load double, ptr %buf, align 8
  %bits2 = load i64, ptr %trk2, align 8
  %rec2 = inttoptr i64 %bits2 to ptr
  call ptr @_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9decrementEPS2_(ptr %rec2)
  ret double %res
}

define double @d_view_rev(double %x, ptr %space, ptr %label) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kokkos_view_lifecycle, metadata !"enzyme_out", double %x, metadata !"enzyme_const", ptr %space, metadata !"enzyme_const", ptr %label)
  ret double %dx
}

define double @d_view_fwd(double %x, double %dx, ptr %space, ptr %label) {
entry:
  %dr = call double (...) @__enzyme_fwddiff(ptr @kokkos_view_lifecycle, double %x, double %dx, metadata !"enzyme_const", ptr %space, metadata !"enzyme_const", ptr %label)
  ret double %dr
}

; CHECK-LABEL: define internal { double } @diffekokkos_view_lifecycle(
; CHECK: call ptr @_ZN6Kokkos4Impl29make_shared_allocation_recordINS_9HostSpaceEvEEPNS0_22SharedAllocationRecordIvvEERKS2_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm(
; CHECK: call void @__enzyme_kokkos_record_inc_d(
; CHECK: call ptr @_ZN6Kokkos4Impl29make_shared_allocation_recordINS_9HostSpaceEvEEPNS0_22SharedAllocationRecordIvvEERKS2_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm(
; CHECK: call void @__enzyme_kokkos_zero_record_data_host(
; CHECK: call ptr @_ZNK6Kokkos4Impl22SharedAllocationRecordIvvE4dataEv(
; CHECK: call void @__enzyme_kokkos_record_dec_d(
; CHECK: ret { double }

; CHECK-LABEL: define internal double @fwddiffekokkos_view_lifecycle(
; CHECK: call ptr @_ZN6Kokkos4Impl29make_shared_allocation_recordINS_9HostSpaceEvEEPNS0_22SharedAllocationRecordIvvEERKS2_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm(
; CHECK: call ptr @_ZN6Kokkos4Impl29make_shared_allocation_recordINS_9HostSpaceEvEEPNS0_22SharedAllocationRecordIvvEERKS2_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm(
; CHECK: call void @__enzyme_kokkos_zero_record_data_host(
; CHECK: call void @__enzyme_kokkos_record_dec_d(
; CHECK: ret double
