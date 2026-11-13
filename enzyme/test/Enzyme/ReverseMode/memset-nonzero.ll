; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

%struct.mixed = type { double, i64 }

@.pat_2d = private unnamed_addr constant [2 x double] [double 1.5, double 2.5], align 16

declare void @__enzyme_autodiff(...)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1)
declare void @memset_pattern16(ptr nocapture writeonly, ptr nocapture readonly, i64)
declare ptr @memset(ptr, i32, i64)
declare ptr @__memset_chk(ptr, i32, i64, i64)

define void @f_nonzero(ptr nocapture %y) {
  call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 8, i1 false)
  %v = load double, ptr %y, align 8
  %v2 = fmul double %v, 2.0
  store double %v2, ptr %y, align 8
  ret void
}

define double @f_mixed(ptr nocapture %s, double %x) {
  call void @llvm.memset.p0.i64(ptr nonnull align 16 dereferenceable(16) %s, i8 7, i64 16, i1 false)
  %d = load double, ptr %s, align 8
  %ip = getelementptr inbounds %struct.mixed, ptr %s, i32 0, i32 1
  %iv = load i64, ptr %ip, align 8
  %ivf = sitofp i64 %iv to double
  %cmp = fcmp oeq double %ivf, 0.0
  %res0 = fadd double %d, %x
  %res = select i1 %cmp, double 0.0, double %res0
  ret double %res
}

define double @f_zerolen(ptr nocapture %y) {
  %v = load double, ptr %y, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 0, i1 false)
  %v2 = fmul double %v, %v
  ret double %v2
}

define void @f_pattern(ptr nocapture writeonly %x) {
  call void @memset_pattern16(ptr align 16 %x, ptr align 16 @.pat_2d, i64 16)
  ret void
}

define void @f_chk_loop(ptr %arr, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %p = getelementptr double, ptr %arr, i64 %i
  %v = load double, ptr %p, align 8
  %is_even = and i64 %i, 1
  %cmp = icmp eq i64 %is_even, 0
  br i1 %cmp, label %do_ms, label %skip

do_ms:
  %rem = sub i64 %n, %i
  %rem_bytes = shl i64 %rem, 3
  %c = call ptr @__memset_chk(ptr %p, i32 63, i64 8, i64 %rem_bytes)
  br label %latch

skip:
  %v2 = fmul double %v, 3.0
  store double %v2, ptr %p, align 8
  br label %latch

latch:
  %inc = add i64 %i, 1
  %cond = icmp slt i64 %inc, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

define double @f_ret(ptr %p, double %x) {
  %r = call ptr @memset(ptr %p, i32 63, i64 8)
  %old = load double, ptr %r, align 8
  %sum = fadd double %old, %x
  store double %sum, ptr %r, align 8
  %v = load double, ptr %r, align 8
  ret double %v
}

define double @f_rt(ptr nocapture %y) {
  call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 8, i1 false)
  %v = load double, ptr %y, align 8
  ret double %v
}

define void @f_huge(ptr nocapture %y) {
  call void @llvm.memset.p0.i64(ptr align 8 %y, i8 63, i64 4294967296, i1 false)
  %v = load double, ptr %y, align 8
  %v2 = fmul double %v, 2.0
  store double %v2, ptr %y, align 8
  ret void
}

define void @caller(ptr %y, ptr %dy, ptr %s, ptr %ds, i64 %n, double %x) {
  call void (...) @__enzyme_autodiff(ptr @f_nonzero, metadata !"enzyme_dup", ptr %y, ptr %dy)
  call void (...) @__enzyme_autodiff(ptr @f_mixed, metadata !"enzyme_dup", ptr %s, ptr %ds, metadata !"enzyme_const", double %x)
  call void (...) @__enzyme_autodiff(ptr @f_zerolen, metadata !"enzyme_dup", ptr %y, ptr %dy)
  call void (...) @__enzyme_autodiff(ptr @f_pattern, metadata !"enzyme_dup", ptr %s, ptr %ds)
  call void (...) @__enzyme_autodiff(ptr @f_chk_loop, metadata !"enzyme_dup", ptr %y, ptr %dy, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @f_ret, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_const", double %x)
  call void (...) @__enzyme_autodiff(ptr @f_rt, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %y, ptr %dy)
  call void (...) @__enzyme_autodiff(ptr @f_huge, metadata !"enzyme_dup", ptr %y, ptr %dy)
  ret void
}

; CHECK: define internal void @diffef_nonzero(ptr nocapture %y, ptr nocapture %"y'")
; CHECK: invert:
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %"y'", i8 0, i64 8, i1 false)

; CHECK: define internal void @diffef_mixed(ptr nocapture %s, ptr nocapture %"s'", double %x, double %differeturn)
; CHECK:   call void @llvm.memset.p0.i64(ptr nonnull align 16 dereferenceable(16) %s, i8 7, i64 16, i1 false)
; CHECK-NEXT:   %[[GEP:.+]] = getelementptr inbounds i8, ptr %"s'", i32 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull align 8 dereferenceable(8) %[[GEP]], i8 7, i64 8, i1 false)
; CHECK: invert:
; CHECK:   call void @llvm.memset.p0.i64(ptr nonnull align 16 dereferenceable(16) %"s'", i8 0, i64 8, i1 false)

; CHECK: define internal void @diffef_zerolen(ptr nocapture %y, ptr nocapture %"y'", double %differeturn)
; CHECK: invert:
; CHECK-NOT: llvm.memset
; CHECK:   ret void

; CHECK: define internal void @diffef_pattern(ptr nocapture writeonly %x, ptr nocapture %"x'")
; CHECK:   call void @memset_pattern16(ptr align 16 %x, ptr align 16 @.pat_2d, i64 16)
; CHECK: invert:
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 16 %"x'", i8 0, i64 16, i1 false)

; CHECK: define internal void @diffef_chk_loop(ptr %arr, ptr %"arr'", i64 %n)
; CHECK: invertdo_ms:
; CHECK:   %[[REM:.+]] = sub i64 %n, %{{.+}}
; CHECK-NEXT:   %[[REMBYTES:.+]] = shl i64 %[[REM]], 3
; CHECK:   %[[PTR:.+]] = getelementptr double, ptr %"arr'", i64 %{{.+}}
; CHECK-NEXT:   %{{.+}} = call ptr @__memset_chk(ptr %[[PTR]], i32 0, i64 8, i64 %[[REMBYTES]])

; CHECK: define internal void @diffef_ret(ptr %p, ptr %"p'", double %x, double %differeturn)
; CHECK: invert:
; CHECK:   %{{.+}} = call ptr @memset(ptr %"p'", i32 0, i64 8)

; CHECK: define internal void @diffef_rt(ptr nocapture %y, ptr nocapture %"y'", double %differeturn)
; CHECK: invert_amerge:
; CHECK-NEXT:   %[[EQ:.+]] = icmp eq ptr %"y'", %y
; CHECK-NEXT:   %[[SEL:.+]] = select i1 %[[EQ]], i64 0, i64 8
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 8 %"y'", i8 0, i64 %[[SEL]], i1 false)

; CHECK: define internal void @diffef_huge(ptr nocapture %y, ptr nocapture %"y'")
; CHECK: invert:
; CHECK:   call void @llvm.memset.p0.i64(ptr align 8 %"y'", i8 0, i64 4294967296, i1 false)
