; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-strict-aliasing=0 -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

; Numeric check for https://github.com/EnzymeAD/Enzyme/issues/3174 (see memset-unknown-loop.ll).
declare void @__enzyme_autodiff(...)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare i32 @printf(ptr, ...)

@fmt = private constant [11 x i8] c"dx[%d]=%f\0A\00"

; Each iteration re-zeroes acc, copies x[i] into it on even iterations only,
; and writes acc[0] back to x[i]. So x[i] is unchanged for even i and becomes 0
; for odd i. With every dx seeded to 1 the correct result is dx = [1, 0, 1, 0].
; Without the reverse-sweep zeroing, the adjoint of the odd iteration's read of
; acc leaks into the preceding even iteration and dx = [2, 0, 2, 0].
define void @f(ptr %x, i64 %n) {
entry:
  %acc = alloca [32 x i8], align 4
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %latch ]
  call void @llvm.memset.p0.i64(ptr align 4 %acc, i8 0, i64 32, i1 false)
  %xp = getelementptr float, ptr %x, i64 %i
  %rem = and i64 %i, 1
  %even = icmp eq i64 %rem, 0
  br i1 %even, label %doStore, label %writeback

doStore:
  %v = load float, ptr %xp, align 4
  store float %v, ptr %acc, align 4
  br label %writeback

writeback:
  %a = load float, ptr %acc, align 4
  store float %a, ptr %xp, align 4
  br label %latch

latch:
  %inext = add nuw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define i32 @main() {
  %x = alloca [4 x float], align 4
  %dx = alloca [4 x float], align 4
  store [4 x float] [float 1.0, float 2.0, float 3.0, float 4.0], ptr %x
  store [4 x float] [float 1.0, float 1.0, float 1.0, float 1.0], ptr %dx
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i64 4)
  br label %ploop
ploop:
  %i = phi i64 [ 0, %0 ], [ %in, %ploop ]
  %p = getelementptr float, ptr %dx, i64 %i
  %v = load float, ptr %p
  %vd = fpext float %v to double
  %it = trunc i64 %i to i32
  call i32 (ptr, ...) @printf(ptr @fmt, i32 %it, double %vd)
  %in = add i64 %i, 1
  %c = icmp eq i64 %in, 4
  br i1 %c, label %done, label %ploop
done:
  ret i32 0
}

; EVAL: dx[0]=1.000000
; EVAL-NEXT: dx[1]=0.000000
; EVAL-NEXT: dx[2]=1.000000
; EVAL-NEXT: dx[3]=0.000000
