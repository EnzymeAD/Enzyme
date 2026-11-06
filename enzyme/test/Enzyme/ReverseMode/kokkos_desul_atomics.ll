; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck %s; fi

%struct.AddOp = type { i8 }
%struct.SubOp = type { i8 }

declare double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_add_fetch_operatorIvvEEdNS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr, ptr, double, i8, i8)
declare double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_sub_fetch_operatorIvvEEdNS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr, ptr, double, i8, i8)
declare i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr, i8)
declare void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr, i8)
declare void @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define double @desul_kernel(ptr %buf, double %x) {
entry:
  %add_op = alloca %struct.AddOp, align 1
  %sub_op = alloca %struct.SubOp, align 1
  %locked = call i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
  %sq = fmul double %x, %x
  %old1 = call double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_add_fetch_operatorIvvEEdNS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %add_op, ptr %buf, double %sq, i8 0, i8 0)
  %old2 = call double @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_sub_fetch_operatorIvvEEdNS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %sub_op, ptr %buf, double %x, i8 0, i8 0)
  call void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0)
  %res = load double, ptr %buf, align 8
  ret double %res
}

define double @d_desul_rev(ptr %buf, ptr %dbuf, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @desul_kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_out", double %x)
  ret double %dx
}

define double @d_desul_fwd(ptr %buf, ptr %dbuf, double %x, double %dx) {
entry:
  %dr = call double (...) @__enzyme_fwddiff(ptr @desul_kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, double %x, double %dx)
  ret double %dr
}

; CHECK-LABEL: define internal { double } @diffedesul_kernel(
; CHECK: call i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
; CHECK: call void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0)
; CHECK: call void @__enzyme_kokkos_desul_spinlock__ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
; CHECK: load atomic double, ptr %"buf'" monotonic, align 8
; CHECK: fneg fast double
; CHECK: load atomic double, ptr %"buf'" monotonic, align 8
; CHECK: fence release
; CHECK: call void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0)
; CHECK: ret { double }

; CHECK-LABEL: define internal double @fwddiffedesul_kernel(
; CHECK: atomicrmw fadd ptr %"buf'"
; CHECK: atomicrmw fsub ptr %"buf'"
; CHECK: ret double
