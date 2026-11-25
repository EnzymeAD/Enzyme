; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Desul's linear atomic fetch_oper (Kokkos::atomic_add / atomic_sub on floats)
; is lowered to atomicrmw fadd / fsub before differentiation, with the memory
; order taken from the MemoryOrder template argument.

declare double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_add_fetch_operatorIvvEEdNS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr, ptr, double, i8, i8)
declare double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_sub_fetch_operatorIvvEEdNS_17MemoryOrderSeqCstENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr, ptr, double, i8, i8)
declare void @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define double @kernel(ptr %buf, double %x) {
entry:
  %op = alloca i8, align 1
  %sq = fmul double %x, %x
  %old1 = call double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_add_fetch_operatorIvvEEdNS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %op, ptr %buf, double %sq, i8 0, i8 0)
  %old2 = call double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_sub_fetch_operatorIvvEEdNS_17MemoryOrderSeqCstENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %op, ptr %buf, double %x, i8 0, i8 0)
  %res = load double, ptr %buf, align 8
  ret double %res
}

define double @dkernel_rev(ptr %buf, ptr %dbuf, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_out", double %x)
  ret double %dx
}

define double @dkernel_fwd(ptr %buf, ptr %dbuf, double %x, double %dx) {
entry:
  %dr = call double (...) @__enzyme_fwddiff(ptr @kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, double %x, double %dx)
  ret double %dr
}

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK-NOT: host_atomic_fetch_oper
; CHECK: %old1 = atomicrmw fadd ptr %buf, double %sq monotonic, align 8
; CHECK: %old2 = atomicrmw fsub ptr %buf, double %x seq_cst, align 8
; CHECK-NOT: host_atomic_fetch_oper
; CHECK: load atomic double, ptr %"buf'" seq_cst
; CHECK: fneg fast double
; CHECK: load atomic double, ptr %"buf'" monotonic
; CHECK: ret { double }

; CHECK-LABEL: define internal double @fwddiffekernel(
; CHECK-NOT: host_atomic_fetch_oper
; CHECK: atomicrmw fadd ptr %"buf'"
; CHECK: atomicrmw fsub ptr %"buf'"
; CHECK: ret double
