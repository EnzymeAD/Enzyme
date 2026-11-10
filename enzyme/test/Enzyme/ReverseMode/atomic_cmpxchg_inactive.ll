; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: rev1_dx=12.000000 flag=1
; EVAL-NEXT: rev2_dx=4.000000 flag=1
; EVAL-NEXT: cas_dx=10.000000 mask=7
; EVAL-NEXT: fwd_dy=12.000000 flag=1
; EVAL-NEXT: split_dy=12.000000 flag=1
; EVAL-NEXT: prev_dy=2.000000 counter=5

@.fmt_rev1 = private unnamed_addr constant [20 x i8] c"rev1_dx=%f flag=%d\0A\00", align 1
@.fmt_rev2 = private unnamed_addr constant [20 x i8] c"rev2_dx=%f flag=%d\0A\00", align 1
@.fmt_cas = private unnamed_addr constant [19 x i8] c"cas_dx=%f mask=%d\0A\00", align 1
@.fmt_fwd = private unnamed_addr constant [19 x i8] c"fwd_dy=%f flag=%d\0A\00", align 1
@.fmt_split = private unnamed_addr constant [21 x i8] c"split_dy=%f flag=%d\0A\00", align 1
@.fmt_prev = private unnamed_addr constant [23 x i8] c"prev_dy=%f counter=%d\0A\00", align 1

declare double @__enzyme_autodiff(ptr, ...)
declare double @__enzyme_fwddiff(ptr, ...)
declare { ptr, double } @__enzyme_augmentfwd(ptr, ...)
declare double @__enzyme_fwdsplit(ptr, ...)
declare i32 @printf(ptr, ...)

define double @claim_once_scale(double %x, ptr %flag) {
entry:
  %cas = cmpxchg ptr %flag, i32 0, i32 1 monotonic monotonic, align 4
  %won = extractvalue { i32, i1 } %cas, 1
  br i1 %won, label %first, label %later

first:
  %sq = fmul double %x, %x
  %scaled = fmul double 3.000000e+00, %sq
  ret double %scaled

later:
  %lin = fmul double 4.000000e+00, %x
  ret double %lin
}

define double @cas_or_loop(double %x, ptr %mask, i32 %bit) {
entry:
  %init = load atomic i32, ptr %mask monotonic, align 4
  br label %loop

loop:
  %expected = phi i32 [ %init, %entry ], [ %prev, %loop ]
  %desired = or i32 %expected, %bit
  %cas = cmpxchg ptr %mask, i32 %expected, i32 %desired monotonic monotonic, align 4
  %prev = extractvalue { i32, i1 } %cas, 0
  %ok = extractvalue { i32, i1 } %cas, 1
  br i1 %ok, label %done, label %loop

done:
  %res = fmul double %x, %x
  ret double %res
}

define double @scale_by_prev(double %x, ptr %counter) {
entry:
  %cas = cmpxchg ptr %counter, i32 2, i32 5 seq_cst monotonic, align 4
  %prev = extractvalue { i32, i1 } %cas, 0
  %fp = sitofp i32 %prev to double
  %res = fmul double %x, %fp
  ret double %res
}

define i32 @main() {
entry:
  %flag = alloca i32, align 4
  %mask = alloca i32, align 4
  %counter = alloca i32, align 4

  store i32 0, ptr %flag, align 4
  %dx1 = call double (ptr, ...) @__enzyme_autodiff(ptr @claim_once_scale, double 2.000000e+00, metadata !"enzyme_const", ptr %flag)
  %f1 = load i32, ptr %flag, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_rev1, double %dx1, i32 %f1)

  %dx2 = call double (ptr, ...) @__enzyme_autodiff(ptr @claim_once_scale, double 2.000000e+00, metadata !"enzyme_const", ptr %flag)
  %f2 = load i32, ptr %flag, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_rev2, double %dx2, i32 %f2)

  store i32 1, ptr %mask, align 4
  %dx_cas = call double (ptr, ...) @__enzyme_autodiff(ptr @cas_or_loop, double 5.000000e+00, metadata !"enzyme_const", ptr %mask, metadata !"enzyme_const", i32 6)
  %m_val = load i32, ptr %mask, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_cas, double %dx_cas, i32 %m_val)

  store i32 0, ptr %flag, align 4
  %dy_fwd = call double (ptr, ...) @__enzyme_fwddiff(ptr @claim_once_scale, metadata !"enzyme_dup", double 2.000000e+00, double 1.000000e+00, metadata !"enzyme_const", ptr %flag)
  %f3 = load i32, ptr %flag, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_fwd, double %dy_fwd, i32 %f3)

  store i32 0, ptr %flag, align 4
  %aug = call { ptr, double } (ptr, ...) @__enzyme_augmentfwd(ptr @claim_once_scale, metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", double 2.000000e+00, double 1.000000e+00, metadata !"enzyme_const", ptr %flag)
  %tape = extractvalue { ptr, double } %aug, 0
  %dy_split = call double (ptr, ...) @__enzyme_fwdsplit(ptr @claim_once_scale, metadata !"enzyme_nofree", metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", double 2.000000e+00, double 1.000000e+00, metadata !"enzyme_const", ptr %flag, ptr %tape)
  %f4 = load i32, ptr %flag, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt_split, double %dy_split, i32 %f4)

  store i32 2, ptr %counter, align 4
  %aug_prev = call { ptr, double } (ptr, ...) @__enzyme_augmentfwd(ptr @scale_by_prev, metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", double 3.000000e+00, double 1.000000e+00, metadata !"enzyme_const", ptr %counter)
  %tape_prev = extractvalue { ptr, double } %aug_prev, 0
  store i32 2, ptr %counter, align 4
  %dy_prev = call double (ptr, ...) @__enzyme_fwdsplit(ptr @scale_by_prev, metadata !"enzyme_nofree", metadata !"enzyme_allocated", i64 -1, metadata !"enzyme_dup", double 3.000000e+00, double 1.000000e+00, metadata !"enzyme_const", ptr %counter, ptr %tape_prev)
  %c_after = load i32, ptr %counter, align 4
  %expected_2_changed = add i32 %c_after, 3
  call i32 (ptr, ...) @printf(ptr @.fmt_prev, double %dy_prev, i32 %expected_2_changed)

  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffeclaim_once_scale(
; CHECK: cmpxchg ptr %flag, i32 0, i32 1
; CHECK-NOT: cmpxchg
; CHECK: ret { double }

; CHECK-LABEL: define internal { double } @diffecas_or_loop(
; CHECK: cmpxchg ptr %mask, i32 %expected, i32 %desired

; CHECK-LABEL: define internal double @fwddiffeclaim_once_scale(
; CHECK: cmpxchg ptr %flag, i32 0, i32 1

; CHECK-LABEL: define internal double @fwddiffeclaim_once_scale.{{[0-9]+}}(double %x, double %"x'", ptr %flag, ptr %tapeArg)
; CHECK-NOT: cmpxchg
; CHECK: ret double

; CHECK-LABEL: define internal double @fwddiffescale_by_prev(double %x, double %"x'", ptr %counter, ptr %tapeArg)
; CHECK-NOT: cmpxchg
; CHECK: ret double
