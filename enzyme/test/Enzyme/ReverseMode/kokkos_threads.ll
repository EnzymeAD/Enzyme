; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck %s; fi

%struct.Closure = type { ptr, ptr, i64 }

declare ptr @_ZN6Kokkos4Impl15ThreadsInternal14resize_scratchEmm(i64, i64)
declare void @_ZN6Kokkos4Impl15ThreadsInternal5startEPFvRS1_PKvES4_(ptr, ptr)
declare void @_ZN6Kokkos4Impl15ThreadsInternal5fenceEv()
declare void @__enzyme_autodiff(...)

; ParallelFor RangePolicy<Threads>::driver exercises the multi-threaded parallel launch path.
define internal void @_ZN6Kokkos4Impl11ParallelForI7FunctorNS_11RangePolicyIJNS_7ThreadsEEEES4_E6driverERNS0_15ThreadsInternalEPKv(ptr %instance, ptr %arg) {
entry:
  %out_slot = getelementptr inbounds %struct.Closure, ptr %arg, i64 0, i32 0
  %in_slot = getelementptr inbounds %struct.Closure, ptr %arg, i64 0, i32 1
  %out = load ptr, ptr %out_slot, align 8
  %in = load ptr, ptr %in_slot, align 8
  %x = load double, ptr %in, align 8
  %sq = fmul double %x, %x
  store double %sq, ptr %out, align 8
  ret void
}

; ParallelReduce RangePolicy<Threads>::driver exercises the solo scratch-shadowed execution path.
define internal void @_ZN6Kokkos4Impl14ParallelReduceI7FunctorNS_11RangePolicyIJNS_7ThreadsEEEEdE6driverERNS0_15ThreadsInternalEPKv(ptr %instance, ptr %arg) {
entry:
  %scr_slot = getelementptr inbounds i8, ptr %instance, i64 8
  %scr_ptr = load ptr, ptr %scr_slot, align 8
  %out_slot = getelementptr inbounds %struct.Closure, ptr %arg, i64 0, i32 0
  %x_ptr = load ptr, ptr %out_slot, align 8
  %v = load double, ptr %x_ptr, align 8
  %scaled = fmul double %v, 3.000000e+00
  store double %scaled, ptr %scr_ptr, align 8
  %red = load double, ptr %scr_ptr, align 8
  store double %red, ptr %x_ptr, align 8
  ret void
}

define double @launch_threads(ptr %out, ptr %in) {
entry:
  %closure = alloca %struct.Closure, align 8
  %scratch = call ptr @_ZN6Kokkos4Impl15ThreadsInternal14resize_scratchEmm(i64 8, i64 0)
  %out_slot = getelementptr inbounds %struct.Closure, ptr %closure, i64 0, i32 0
  %in_slot = getelementptr inbounds %struct.Closure, ptr %closure, i64 0, i32 1
  %n_slot = getelementptr inbounds %struct.Closure, ptr %closure, i64 0, i32 2
  store ptr %out, ptr %out_slot, align 8
  store ptr %in, ptr %in_slot, align 8
  store i64 1, ptr %n_slot, align 8
  call void @_ZN6Kokkos4Impl15ThreadsInternal5startEPFvRS1_PKvES4_(ptr @_ZN6Kokkos4Impl11ParallelForI7FunctorNS_11RangePolicyIJNS_7ThreadsEEEES4_E6driverERNS0_15ThreadsInternalEPKv, ptr %closure)
  call void @_ZN6Kokkos4Impl15ThreadsInternal5fenceEv()
  call void @_ZN6Kokkos4Impl15ThreadsInternal5startEPFvRS1_PKvES4_(ptr @_ZN6Kokkos4Impl14ParallelReduceI7FunctorNS_11RangePolicyIJNS_7ThreadsEEEEdE6driverERNS0_15ThreadsInternalEPKv, ptr %closure)
  call void @_ZN6Kokkos4Impl15ThreadsInternal5fenceEv()
  %r = load double, ptr %out, align 8
  ret double %r
}

define void @d_launch_threads(ptr %out, ptr %dout, ptr %in, ptr %din) {
entry:
  call void (...) @__enzyme_autodiff(ptr @launch_threads, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_dup", ptr %in, ptr %din)
  ret void
}

; CHECK-LABEL: define internal void @diffelaunch_threads(
; CHECK: call ptr @_ZN6Kokkos4Impl15ThreadsInternal14resize_scratchEmm(i64 8, i64 0)
; CHECK: call void @_ZN6Kokkos4Impl15ThreadsInternal5startEPFvRS1_PKvES4_(ptr {{.*}}.kokkos_threads_aug
; CHECK: call {{.*}}@augmented__ZN6Kokkos4Impl14ParallelReduce
; CHECK: call void @diffe_ZN6Kokkos4Impl14ParallelReduce
; CHECK: call void @_ZN6Kokkos4Impl15ThreadsInternal5startEPFvRS1_PKvES4_(ptr {{.*}}.kokkos_threads_rev
; CHECK: call void @_ZN6Kokkos4Impl15ThreadsInternal14internal_fenceEv()
; CHECK: ret void
