; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EXEC; fi

; Desul address locks in a callee differentiated in split mode: whether each
; lock_address attempt acquired the lock is taped by the augmented primal and
; read back by the gradient, both for the gradient's replay of the primal spin
; loop and to decide whether to unlock in reverse.

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

define double @inner(ptr %buf, double %x) noinline {
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

define double @kernel(ptr %buf, double %x) {
entry:
  %r = call double @inner(ptr %buf, double %x)
  %r2 = fmul double %r, %x
  ret double %r2
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
  call i32 (ptr, ...) @printf(ptr @fmt, double %dx, double %db)
  ret i32 0
}

; kernel = (buf * x) * x: d/dx = 2 * buf * x = 30, d/dbuf = x * x = 9.
; EXEC: 30.000000 9.000000

; The augmented primal tapes the result of every lock_address attempt.
; CHECK: define internal { { ptr, double }, double } @augmented_inner(ptr nocapture %buf, ptr nocapture %"buf'", double %x)
; CHECK:   %locked = call i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
; CHECK-NEXT:   %[[cache:.+]] = load ptr, ptr %locked_cache
; CHECK-NEXT:   %[[slot:.+]] = getelementptr inbounds i1, ptr %[[cache]], i64 %iv
; CHECK-NEXT:   store i1 %locked, ptr %[[slot]]
; CHECK-NEXT:   br i1 %locked, label %body, label %spin
; CHECK:   call void @_ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0)

; The gradient replays the spin loop from the tape instead of locking again,
; and only unlocks in reverse for the attempt that acquired the lock.
; CHECK: define internal { double } @diffeinner(ptr nocapture %buf, ptr nocapture %"buf'", double %x, double %differeturn, { ptr, double } %tapeArg)
; CHECK-NOT: call i1 @_ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_
; CHECK: spin:
; CHECK:   %locked = load i1, ptr %
; CHECK-NEXT:   br i1 %locked, label %body, label %spin
; CHECK: invertspin:
; CHECK:   %[[acq:.+]] = load i1, ptr %
; CHECK-NEXT:   call void @__enzyme_unlock_if__ZN5desul4Impl14unlock_addressINS_17MemoryScopeDeviceEEEvPvT_(ptr %buf, i8 0, i1 %[[acq]])
; CHECK: invertbody:
; CHECK:   call void @__enzyme_spinlock__ZN5desul4Impl12lock_addressINS_17MemoryScopeDeviceEEEbPvT_(ptr %buf, i8 0)
