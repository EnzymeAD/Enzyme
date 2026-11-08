; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; The adjoint of a Desul (Kokkos atomics) thread fence call is a thread fence
; at the corresponding location in the reverse pass, with acquire and release
; exchanged (as for a fence instruction): the reverse pass traverses the
; synchronized accesses in the opposite order.

declare void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderReleaseENS_17MemoryScopeDeviceEEEvT_T0_(i8, i8)
declare void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderAcquireENS_17MemoryScopeDeviceEEEvT_T0_(i8, i8)
declare void @_ZN5desul4Impl25host_atomic_thread_fenceINS_17MemoryOrderSeqCstENS_17MemoryScopeDeviceEEEvT_T0_(i8, i8)
declare double @__enzyme_autodiff(...)

define double @kernel(ptr %buf, double %x) {
entry:
  call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderAcquireENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
  %v = load double, ptr %buf, align 8
  %r = fmul double %v, %x
  store double %r, ptr %buf, align 8
  call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderReleaseENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
  call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_17MemoryOrderSeqCstENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
  ret double %r
}

define double @dkernel(ptr %buf, ptr %dbuf, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_out", double %x)
  ret double %dx
}

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK: call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderAcquireENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
; CHECK: call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderReleaseENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
; CHECK-NEXT: call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_17MemoryOrderSeqCstENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)

; Reverse: seq_cst, then acquire (for the release), ..., release (for the
; acquire).
; CHECK: invertentry:
; CHECK: call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_17MemoryOrderSeqCstENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
; CHECK-NEXT: call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderAcquireENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
; CHECK: store double {{.*}}, ptr %"buf'"
; CHECK: call void @_ZN5desul4Impl25host_atomic_thread_fenceINS_18MemoryOrderReleaseENS_17MemoryScopeDeviceEEEvT_T0_(i8 0, i8 0)
; CHECK-NOT: call void @_ZN5desul4Impl25host_atomic_thread_fence
; CHECK: ret { double }
