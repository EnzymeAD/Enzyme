; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,simplifycfg)" -S | FileCheck %s; fi

; Reduced from MITgcm's checkpointed time step, where myIter = iloop - 1 +
; nIter0 and nIter0 = NINT(...), which flang lowers to llvm.lround. If the
; rounded value has no type, neither does the counter stored in %it, so %it
; may hold active data. Then body's load of it is active, and its type cannot
; be deduced.

@param = global double 2.000000e+00, align 8
@flag = global i1 false, align 1

define internal void @body(ptr %x, ptr %it) {
entry:
  %v = load i32, ptr %it, align 4
  %f = load i1, ptr @flag, align 1
  %w = add i32 %v, -1
  %m = select i1 %f, i32 %v, i32 %w
  call void @use(ptr %x, i32 %m)
  %n = add i32 %v, 1
  store i32 %n, ptr %it, align 4
  ret void
}

define internal void @use(ptr %x, i32 %n) {
entry:
  %c = icmp sgt i32 %n, 0
  br i1 %c, label %then, label %exit

then:
  %y = load double, ptr %x, align 8
  %z = fmul double %y, %y
  store double %z, ptr %x, align 8
  br label %exit

exit:
  ret void
}

define void @step(i64 %i, ptr %x) {
entry:
  %it = alloca i32, align 4
  %p = load double, ptr @param, align 8
  %r = call i32 @llvm.lround.i32.f64(double %p)
  %t = trunc i64 %i to i32
  %a = add i32 %t, -1
  %s = add i32 %a, %r
  store i32 %s, ptr %it, align 4
  call void @body(ptr %x, ptr %it)
  ret void
}

declare i32 @llvm.lround.i32.f64(double)

define void @df(ptr %x, ptr %dx) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @step, i64 3, ptr %x, ptr %dx)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; The counter is an integer, so it has no shadow.
; CHECK: define internal void @diffestep(i64 %i, ptr {{.*}}%x, ptr {{.*}}%"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %it = alloca i32, align 4
; CHECK-NOT:    alloca
; CHECK:        call void @diffebody(ptr %x, ptr %"x'", ptr %it)
; CHECK-NEXT:   ret void

; CHECK: define internal void @diffebody(ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%it)
