; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EXEC; fi

; Desul (Kokkos atomics) guards atomics on types without a lock-free atomic
; with address locks. A region locked in the forward pass is locked again in
; the reverse pass: the reverse of unlock_address spins on lock_address, the
; reverse of lock_address is a release fence followed by unlock_address.

@locks = internal global [1 x i32] zeroinitializer, align 4

define i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %p, i8 %scope) noinline {
  %old = cmpxchg ptr @locks, i32 0, i32 1 acquire monotonic, align 4
  %ok = extractvalue { i32, i1 } %old, 1
  ret i1 %ok
}

define void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %p, i8 %scope) noinline {
  store atomic i32 0, ptr @locks release, align 4
  ret void
}

declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
@fmt = private constant [7 x i8] c"%f %f\0A\00"

define double @kernel(ptr %buf, double %x) {
entry:
  br label %spin

spin:
  %locked = call i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
  br i1 %locked, label %body, label %spin

body:
  %v = load double, ptr %buf, align 8
  %r = fmul double %v, %x
  store double %r, ptr %buf, align 8
  call void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0)
  ret double %r
}

define double @dkernel(ptr %buf, ptr %dbuf, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_out", double %x)
  ret double %dx
}

define i32 @main() {
  %buf = alloca double, align 8
  %dbuf = alloca double, align 8
  store double 5.0, ptr %buf
  store double 0.0, ptr %dbuf
  %dx = call double @dkernel(ptr %buf, ptr %dbuf, double 3.0)
  %db = load double, ptr %dbuf
  %lock = load i32, ptr @locks
  %unlocked = icmp eq i32 %lock, 0
  %dbz = select i1 %unlocked, double %db, double -1.0
  call i32 (ptr, ...) @printf(ptr @fmt, double %dx, double %dbz)
  ret i32 0
}

; r = buf * x: dx = buf = 5, d(buf) = x = 3 (and the lock is released again).
; EXEC: 5.000000 3.000000

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK: %locked = call i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
; CHECK: call void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0)
; CHECK-DAG: call void @__enzyme_spinlock__ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
; CHECK-DAG: call void @__enzyme_unlock_if__ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0, i1 %{{.+}})

; CHECK-LABEL: define internal void @__enzyme_unlock_if__ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %lock, i8 %arg1, i1 %acquired)
; CHECK: br i1 %acquired, label %unlock, label %exit
; CHECK: unlock:
; CHECK-NEXT: fence release
; CHECK-NEXT: call void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %lock, i8 %arg1)

; CHECK-LABEL: define internal void @__enzyme_spinlock__ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %lock, i8 %arg1)
; CHECK: spin:
; CHECK-NEXT: %[[GOT:.+]] = call i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %lock, i8 %arg1)
; CHECK-NEXT: br i1 %[[GOT]], label %exit, label %spin
; CHECK: exit:
; CHECK-NEXT: fence acquire
; CHECK-NEXT: ret void
