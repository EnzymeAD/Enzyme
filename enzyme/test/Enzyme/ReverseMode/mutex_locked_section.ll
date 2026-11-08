; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EXEC; fi

; A section locked with std::mutex or a pthread mutex in the forward pass is
; locked again, in reverse order, in the reverse pass: the reverse of unlock
; is lock, the reverse of lock is unlock, and the reverse of try_lock is an
; unlock only if the lock was acquired.

%"class.std::__1::mutex" = type { i32 }

; Stand-ins for the libc++ std::mutex methods (a counter of the lock depth).
define void @_ZNSt3__15mutex4lockEv(ptr %this) noinline {
  %d = load i32, ptr %this, align 4
  %n = add i32 %d, 1
  store i32 %n, ptr %this, align 4
  ret void
}

define void @_ZNSt3__15mutex6unlockEv(ptr %this) noinline {
  %d = load i32, ptr %this, align 4
  %n = sub i32 %d, 1
  store i32 %n, ptr %this, align 4
  ret void
}

define i1 @_ZNSt3__15mutex8try_lockEv(ptr %this) noinline {
  %d = load i32, ptr %this, align 4
  %free = icmp eq i32 %d, 0
  br i1 %free, label %take, label %exit
take:
  store i32 1, ptr %this, align 4
  br label %exit
exit:
  ret i1 %free
}

declare i32 @pthread_mutex_lock(ptr)
declare i32 @pthread_mutex_unlock(ptr)
declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
@fmt = private constant [13 x i8] c"%f %f %d %d\0A\00"

; buf *= x under the std::mutex, then buf *= x again if the try_lock succeeds,
; then buf *= x under the pthread mutex.
define double @kernel(ptr %m, ptr %pm, ptr %buf, double %x) {
entry:
  call void @_ZNSt3__15mutex4lockEv(ptr %m)
  %v = load double, ptr %buf, align 8
  %r = fmul double %v, %x
  store double %r, ptr %buf, align 8
  call void @_ZNSt3__15mutex6unlockEv(ptr %m)
  %got = call i1 @_ZNSt3__15mutex8try_lockEv(ptr %m)
  br i1 %got, label %locked, label %join
locked:
  %v2 = load double, ptr %buf, align 8
  %r2 = fmul double %v2, %x
  store double %r2, ptr %buf, align 8
  call void @_ZNSt3__15mutex6unlockEv(ptr %m)
  br label %join
join:
  %p = call i32 @pthread_mutex_lock(ptr %pm)
  %v3 = load double, ptr %buf, align 8
  %r3 = fmul double %v3, %x
  store double %r3, ptr %buf, align 8
  %q = call i32 @pthread_mutex_unlock(ptr %pm)
  ret double %r3
}

define double @dkernel(ptr %m, ptr %pm, ptr %buf, ptr %dbuf, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kernel, metadata !"enzyme_const", ptr %m, metadata !"enzyme_const", ptr %pm, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_out", double %x)
  ret double %dx
}

define i32 @main() {
  %m = alloca i32, align 4
  %pm = alloca [64 x i8], align 8
  %buf = alloca double, align 8
  %dbuf = alloca double, align 8
  store i32 0, ptr %m
  call void @llvm.memset.p0.i64(ptr %pm, i8 0, i64 64, i1 false)
  store double 5.0, ptr %buf
  store double 0.0, ptr %dbuf
  %dx = call double @dkernel(ptr %m, ptr %pm, ptr %buf, ptr %dbuf, double 2.0)
  %db = load double, ptr %dbuf
  %depth = load i32, ptr %m
  %buf2 = load double, ptr %buf
  %bufi = fptosi double %buf2 to i32
  call i32 (ptr, ...) @printf(ptr @fmt, double %dx, double %db, i32 %depth, i32 %bufi)
  ret i32 0
}

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; kernel = buf * x^3: d/dx = 3 * 5 * 4 = 60, d/dbuf = 8; the std::mutex is
; balanced (depth 0) after forward and reverse; buf = 40.
; EXEC: 60.000000 8.000000 0 40

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK: call void @_ZNSt3__15mutex4lockEv(ptr %m)
; CHECK: call void @_ZNSt3__15mutex6unlockEv(ptr %m)
; CHECK: %got = call i1 @_ZNSt3__15mutex8try_lockEv(ptr %m)

; Reverse blocks are emitted in the order invertentry, invertlocked,
; invertjoin (the reverse runs them in the opposite order).

; Reverse of the std::mutex section, preceded by the reverse of the try_lock
; (also in entry): unlock only if it acquired the lock.
; CHECK: invertentry:
; CHECK: call void @__enzyme_unlock_if__ZNSt3__15mutex6unlockEv(ptr %m, i1 %got)
; CHECK: call void @_ZNSt3__15mutex4lockEv(ptr %m)
; CHECK: store double {{.*}}, ptr %"buf'"
; CHECK: call void @_ZNSt3__15mutex6unlockEv(ptr %m)

; Reverse of the section guarded by try_lock: its unlock becomes a lock.
; CHECK: invertlocked:
; CHECK: call void @_ZNSt3__15mutex4lockEv(ptr %m)
; CHECK: store double {{.*}}, ptr %"buf'"
; CHECK: br label %invertentry

; Reverse of the pthread-locked section.
; CHECK: invertjoin:
; CHECK: call i32 @pthread_mutex_lock(ptr %pm)
; CHECK: store double {{.*}}, ptr %"buf'"
; CHECK: call i32 @pthread_mutex_unlock(ptr %pm)

; CHECK-LABEL: define internal void @__enzyme_unlock_if__ZNSt3__15mutex6unlockEv(ptr %lock, i1 %acquired)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 %acquired, label %unlock, label %exit
; CHECK: unlock:
; CHECK-NEXT:   call void @_ZNSt3__15mutex6unlockEv(ptr %lock)
