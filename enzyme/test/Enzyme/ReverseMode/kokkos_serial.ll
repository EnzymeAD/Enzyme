; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck %s; fi

%"class.Kokkos::Serial" = type { ptr, ptr }

declare void @_ZN6Kokkos6SerialC1Ev(ptr)
declare void @_ZN6Kokkos6SerialD1Ev(ptr)
declare void @_ZN6Kokkos6Serial5fenceERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(ptr, ptr)
declare void @_ZN6Kokkos4Impl14SerialInternal23resize_thread_team_dataEmmmm(ptr, i64, i64, i64, i64)
declare void @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define double @kokkos_serial_ops(double %x, ptr %label) {
entry:
  %exec = alloca %"class.Kokkos::Serial", align 8
  call void @_ZN6Kokkos6SerialC1Ev(ptr %exec)
  %internal = load ptr, ptr %exec, align 8
  call void @_ZN6Kokkos4Impl14SerialInternal23resize_thread_team_dataEmmmm(ptr %internal, i64 64, i64 64, i64 128, i64 64)
  %scr_slot = getelementptr inbounds i8, ptr %internal, i64 24
  %scr = load ptr, ptr %scr_slot, align 8
  %sq = fmul double %x, %x
  store double %sq, ptr %scr, align 8
  call void @_ZN6Kokkos6Serial5fenceERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(ptr %exec, ptr %label)
  %res = load double, ptr %scr, align 8
  call void @_ZN6Kokkos6SerialD1Ev(ptr %exec)
  ret double %res
}

define double @d_serial_rev(double %x, ptr %label) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kokkos_serial_ops, metadata !"enzyme_out", double %x, metadata !"enzyme_const", ptr %label)
  ret double %dx
}

define double @d_serial_fwd(double %x, double %dx, ptr %label) {
entry:
  %dr = call double (...) @__enzyme_fwddiff(ptr @kokkos_serial_ops, double %x, double %dx, metadata !"enzyme_const", ptr %label)
  ret double %dr
}

; CHECK-LABEL: define internal { double } @diffekokkos_serial_ops(
; CHECK: call void @_ZN6Kokkos6SerialC1Ev(
; CHECK: call void @_ZN6Kokkos4Impl14SerialInternalC1Ev(
; CHECK: call void @__enzyme_kokkos_serial_resize_d(
; CHECK: call void @_ZN6Kokkos4Impl14SerialInternalD1Ev(
; CHECK: ret { double }

; CHECK-LABEL: define internal double @fwddiffekokkos_serial_ops(
; CHECK: call void @__enzyme_kokkos_serial_init_d(
; CHECK: call void @__enzyme_kokkos_serial_resize_d(
; CHECK: ret double
