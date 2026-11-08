; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Desul (Kokkos atomics) lock-array setup and std::mutex / pthread mutex
; construction and destruction are inactive: replayed in the forward pass
; only, never differentiated, and not wrapped in nofree_ clones. Reads of the
; lock arrays carry no derivative.

%"class.std::__1::mutex" = type { %union.pthread_mutex_t }
%union.pthread_mutex_t = type { [40 x i8] }

@_ZN5desul4Impl30HOST_SPACE_ATOMIC_LOCKS_DEVICEE = external global ptr, align 8

declare void @_ZNSt3__15mutexD1Ev(ptr)
declare i32 @pthread_mutex_init(ptr, ptr)
declare i32 @pthread_mutex_destroy(ptr)
declare void @_ZN5desul4Impl16init_lock_arraysEv()
declare void @_ZN5desul4Impl20finalize_lock_arraysEv()
declare double @__enzyme_autodiff(...)

define double @kernel(ptr %m, ptr %pm, ptr %x) {
entry:
  call void @_ZN5desul4Impl16init_lock_arraysEv()
  %i = call i32 @pthread_mutex_init(ptr %pm, ptr null)
  %locks = load ptr, ptr @_ZN5desul4Impl30HOST_SPACE_ATOMIC_LOCKS_DEVICEE, align 8
  %lock0 = load i32, ptr %locks, align 4
  %free = icmp eq i32 %lock0, 0
  %v = load double, ptr %x, align 8
  %sq = fmul double %v, %v
  %res = select i1 %free, double %sq, double 0.0
  %d = call i32 @pthread_mutex_destroy(ptr %pm)
  call void @_ZNSt3__15mutexD1Ev(ptr %m)
  call void @_ZN5desul4Impl20finalize_lock_arraysEv()
  ret double %res
}

define double @dkernel(ptr %m, ptr %pm, ptr %x, ptr %dx) {
entry:
  %dr = call double (...) @__enzyme_autodiff(ptr @kernel, metadata !"enzyme_const", ptr %m, metadata !"enzyme_const", ptr %pm, metadata !"enzyme_dup", ptr %x, ptr %dx)
  ret double %dr
}

; CHECK-LABEL: define internal void @diffekernel(
; CHECK-NOT: @nofree_
; CHECK: call void @_ZN5desul4Impl16init_lock_arraysEv()
; CHECK-NEXT: call i32 @pthread_mutex_init(ptr %pm, ptr null)
; CHECK: call i32 @pthread_mutex_destroy(ptr %pm)
; CHECK-NEXT: call void @_ZNSt3__15mutexD1Ev(ptr %m)
; CHECK-NEXT: call void @_ZN5desul4Impl20finalize_lock_arraysEv()
; CHECK-NOT: @nofree_
; CHECK: fadd fast double
; CHECK-NOT: call void @_ZNSt3__15mutex
; CHECK-NOT: call i32 @pthread_mutex
; CHECK-NOT: call void @_ZN5desul4Impl
