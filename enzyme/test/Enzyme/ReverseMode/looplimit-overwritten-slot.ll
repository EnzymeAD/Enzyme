; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

; Both bounds of the inner loop are loaded from a stack slot that the outer
; loop rewrites in every iteration, and @kern is called in a loop, so it is
; differentiated in split mode. The reverse pass needs the inner trip count,
; hi - lo, which the min-cut must therefore keep in the tape. It used to only
; see the exit branch, cached that per inner iteration instead, and left the
; reverse pass to reload lo and hi from the slot, which does not exist there
; ("Illegal replace ficticious phi" for the slot).

; out[k] += x[0] * (hi - lo + 1) with lo = k and hi = 2k, for k = 0, 1, 2:
; d/dx[0] = 1 + 2 + 3 per call of @kern

; EVAL: dx0=6.000000 dx1=6.000000 dx2=0.000000

@fmt = private unnamed_addr constant [22 x i8] c"dx0=%f dx1=%f dx2=%f\0A\00", align 1

declare i32 @printf(ptr, ...)

declare void @__enzyme_autodiff(...)

define void @outer(ptr %x, ptr %out, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %xi = getelementptr inbounds double, ptr %x, i64 %i
  call void @kern(ptr %xi, ptr %out, i64 %n)
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 2
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @kern(ptr %x, ptr %out, i64 %n) {
entry:
  %slot = alloca [2 x i64], align 8
  %hi.ptr = getelementptr inbounds i64, ptr %slot, i64 1
  %x0 = load double, ptr %x, align 8
  br label %outer

outer:
  %k = phi i64 [ 0, %entry ], [ %knext, %latch ]
  store i64 %k, ptr %slot, align 8
  %k2 = shl i64 %k, 1
  store i64 %k2, ptr %hi.ptr, align 8
  %lo = load i64, ptr %slot, align 8
  %hi = load i64, ptr %hi.ptr, align 8
  br label %inner

inner:
  %j = phi i64 [ %lo, %outer ], [ %jnext, %inner ]
  %s = phi double [ 0.000000e+00, %outer ], [ %snext, %inner ]
  %snext = fadd double %s, %x0
  %jnext = add i64 %j, 1
  %idone = icmp eq i64 %j, %hi
  br i1 %idone, label %latch, label %inner

latch:
  %op = getelementptr inbounds double, ptr %out, i64 %k
  %old = load double, ptr %op, align 8
  %new = fadd double %old, %snext
  store double %new, ptr %op, align 8
  %knext = add nuw nsw i64 %k, 1
  %odone = icmp eq i64 %knext, %n
  br i1 %odone, label %end, label %outer

end:
  ret void
}

define i32 @main() {
entry:
  %x = alloca [3 x double], align 8
  %dx = alloca [3 x double], align 8
  %out = alloca [3 x double], align 8
  %dout = alloca [3 x double], align 8
  store [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], ptr %x, align 8
  store [3 x double] zeroinitializer, ptr %dx, align 8
  store [3 x double] zeroinitializer, ptr %out, align 8
  store [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], ptr %dout, align 8
  call void (...) @__enzyme_autodiff(ptr @outer, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %out, ptr %dout, metadata !"enzyme_const", i64 3)
  %dx0 = load double, ptr %dx, align 8
  %dx1.ptr = getelementptr inbounds double, ptr %dx, i64 1
  %dx1 = load double, ptr %dx1.ptr, align 8
  %dx2.ptr = getelementptr inbounds double, ptr %dx, i64 2
  %dx2 = load double, ptr %dx2.ptr, align 8
  %r = call i32 (ptr, ...) @printf(ptr @fmt, double %dx0, double %dx1, double %dx2)
  ret i32 0
}
