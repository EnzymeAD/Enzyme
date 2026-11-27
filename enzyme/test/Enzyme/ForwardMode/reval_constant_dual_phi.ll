; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s --check-prefix=PREOPT; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EXEC; fi

; Regression test for https://github.com/EnzymeAD/Enzyme/issues/3841:
; When an internal global's activity is initially conservatively active during
; forceAugmentedReturns() and later re-evaluated to constant while visiting
; instructions (e.g. during invertPointerM()), forward-mode 'dual_phi
; placeholders created for instructions depending on that load must be resolved
; or erased cleanly instead of being left orphaned in the differentiated IR,
; and float shadows loaded from constant globals must be zeroed rather than
; reading primal values.
;
; Each test function uses its own dedicated internal global so the global has no
; external users and re-evaluates to constant during differentiation.

@g_orig = internal global double 3.000000e+00, align 8
@g_direct = internal global double 3.000000e+00, align 8
@g_fmul = internal global double 3.000000e+00, align 8
@g_extra = internal global double 3.000000e+00, align 8
@g_f32 = internal global float 2.500000e+00, align 4
@g_arr = internal global [2 x double] [double 3.000000e+00, double 5.000000e+00], align 8
@g_struct = internal global double 9.000000e+00, align 8
@g_select = internal global double 9.000000e+00, align 8
@g_vec = internal global double 3.000000e+00, align 8
@g_split_mul = internal global double 3.000000e+00, align 8
@g_split_loop = internal global double 3.000000e+00, align 8

@fmt_f = private constant [6 x i8] c"%.1f\0A\00", align 1
@fmt_pair = private constant [9 x i8] c"%d %.1f\0A\00", align 1
@fmt_vec = private constant [11 x i8] c"%.1f %.1f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare double @llvm.sin.f64(double)
declare double @__enzyme_fwddiff(ptr, ...)
declare [2 x double] @__enzyme_fwddiff_vec(ptr, ...)
declare { ptr, double } @__enzyme_fwddiff_mixed(ptr, ...)
declare double @__enzyme_fwdsplit(ptr, ...)

; 1. Original issue #3841 reproducer: load from internal global + fmul.
define double @f_orig() {
entry:
  %0 = load double, ptr @g_orig, align 8
  %1 = fmul double %0, 0.000000e+00
  ret double %1
}

; 2. Direct load from internal global feeding a loop PHI (must zero shadow, not load primal 3.0).
define double @f_loop_phi_direct(double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ %x, %entry ], [ %0, %loop ]
  %0 = load double, ptr @g_direct, align 8
  %dummy = fadd double %0, 0.000000e+00
  %inc = add i64 %i, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret double %acc
}

; 3. Re-evaluated fmul feeding a loop PHI.
define double @f_loop_phi_fmul(double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ %x, %entry ], [ %1, %loop ]
  %0 = load double, ptr @g_fmul, align 8
  %1 = fmul double %0, 2.000000e+00
  %inc = add i64 %i, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret double %acc
}

; 4. TableGen instruction/intrinsic derivatives (fneg, llvm.sin), fpext, ConstantExpr GEP.
define double @f_extra_ops(i1 %cond) {
entry:
  %v0 = load double, ptr @g_extra, align 8
  %neg = fneg double %v0
  %sin = call double @llvm.sin.f64(double %neg)
  %f32 = load float, ptr @g_f32, align 4
  %ext = fpext float %f32 to double
  %arr0 = load double, ptr @g_arr, align 8
  %gepv = load double, ptr getelementptr inbounds ([2 x double], ptr @g_arr, i64 0, i64 1), align 8
  %arrsum = fadd double %arr0, %gepv
  %sum = fadd double %sin, %ext
  %sel = select i1 %cond, double %sum, double %arrsum
  ret double %sel
}

; 5. Mixed { ptr, double } containing constant ptr + re-evaluated constant double:
;    must preserve primal pointer field while zeroing the float derivative field.
define { ptr, double } @f_mixed_struct(ptr %x, double %y, i64 %n) {
entry:
  %init0 = insertvalue { ptr, double } undef, ptr %x, 0
  %init1 = insertvalue { ptr, double } %init0, double %y, 1
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi { ptr, double } [ %init1, %entry ], [ %s, %loop ]
  %0 = load double, ptr @g_struct, align 8
  %dummy = fadd double %0, 0.000000e+00
  %s0 = insertvalue { ptr, double } undef, ptr %x, 0
  %s = insertvalue { ptr, double } %s0, double %0, 1
  %next = add i64 %i, 1
  %cmp = icmp slt i64 %next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret { ptr, double } %acc
}

; 6. Mixed { ptr, double } select feeding loop PHI and extractvalue of float field.
define double @f_mixed_select(ptr %x, double %y, i1 %c, i64 %n) {
entry:
  %init0 = insertvalue { ptr, double } undef, ptr %x, 0
  %init1 = insertvalue { ptr, double } %init0, double %y, 1
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi { ptr, double } [ %init1, %entry ], [ %sel, %loop ]
  %0 = load double, ptr @g_select, align 8
  %dummy = fadd double %0, 0.000000e+00
  %s0 = insertvalue { ptr, double } undef, ptr %x, 0
  %s = insertvalue { ptr, double } %s0, double %0, 1
  %sel = select i1 %c, { ptr, double } %s, { ptr, double } %s
  %next = add i64 %i, 1
  %cmp = icmp slt i64 %next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  %res = extractvalue { ptr, double } %acc, 1
  ret double %res
}

; 7. Vector forward mode (width = 2) with loop PHI.
define double @f_vec_loop(double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ %x, %entry ], [ %0, %loop ]
  %0 = load double, ptr @g_vec, align 8
  %dummy = fadd double %0, 0.000000e+00
  %inc = add i64 %i, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret double %acc
}

; 8. ForwardModeSplit straight-line fmul (activity re-evaluated mid-visitor).
define double @f_split_mul() {
entry:
  %0 = load double, ptr @g_split_mul, align 8
  %1 = fmul double %0, 2.000000e+00
  ret double %1
}

; 9. ForwardModeSplit loop PHI load.
define double @f_split_loop(double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ %x, %entry ], [ %0, %loop ]
  %0 = load double, ptr @g_split_loop, align 8
  %dummy = fadd double %0, 0.000000e+00
  %inc = add i64 %i, 1
  %cmp = icmp slt i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret double %acc
}

define double @df_orig() {
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @f_orig)
  ret double %r
}

define double @df_loop_phi_direct(double %x, i64 %n) {
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @f_loop_phi_direct, metadata !"enzyme_dup", double %x, double 1.000000e+00, metadata !"enzyme_const", i64 %n)
  ret double %r
}

define double @df_loop_phi_fmul(double %x, i64 %n) {
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @f_loop_phi_fmul, metadata !"enzyme_dup", double %x, double 1.000000e+00, metadata !"enzyme_const", i64 %n)
  ret double %r
}

define double @df_extra_ops(i1 %cond) {
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @f_extra_ops, metadata !"enzyme_const", i1 %cond)
  ret double %r
}

define { ptr, double } @df_mixed_struct(ptr %x, double %y, i64 %n) {
  %r = call { ptr, double } (ptr, ...) @__enzyme_fwddiff_mixed(ptr @f_mixed_struct, metadata !"enzyme_const", ptr %x, metadata !"enzyme_dup", double %y, double 1.000000e+00, metadata !"enzyme_const", i64 %n)
  ret { ptr, double } %r
}

define double @df_mixed_select(ptr %x, double %y, i1 %c, i64 %n) {
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @f_mixed_select, metadata !"enzyme_const", ptr %x, metadata !"enzyme_dup", double %y, double 1.000000e+00, metadata !"enzyme_const", i1 %c, metadata !"enzyme_const", i64 %n)
  ret double %r
}

define [2 x double] @df_vec_loop(double %x, i64 %n) {
  %r = call [2 x double] (ptr, ...) @__enzyme_fwddiff_vec(ptr @f_vec_loop, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", double %x, double 1.000000e+00, double 2.000000e+00, metadata !"enzyme_const", i64 %n)
  ret [2 x double] %r
}

define double @df_split_mul() {
  %r = call double (ptr, ...) @__enzyme_fwdsplit(ptr @f_split_mul, ptr null)
  ret double %r
}

define double @df_split_loop(double %x, i64 %n) {
  %r = call double (ptr, ...) @__enzyme_fwdsplit(ptr @f_split_loop, double %x, double 1.000000e+00, i64 %n, ptr null)
  ret double %r
}

define i32 @main() {
entry:
  %buf = alloca double, align 8
  store double 1.000000e+00, ptr %buf, align 8

  %r0 = call double @df_orig()
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %r0)

  %r1 = call double @df_loop_phi_direct(double 5.000000e+00, i64 1)
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %r1)
  %r2 = call double @df_loop_phi_direct(double 5.000000e+00, i64 2)
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %r2)

  %r3 = call double @df_loop_phi_fmul(double 5.000000e+00, i64 2)
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %r3)

  %r4 = call double @df_extra_ops(i1 true)
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %r4)

  %rm = call { ptr, double } @df_mixed_struct(ptr %buf, double 5.000000e+00, i64 2)
  %rm_p = extractvalue { ptr, double } %rm, 0
  %rm_d = extractvalue { ptr, double } %rm, 1
  %eq = icmp eq ptr %rm_p, %buf
  %eq_i = zext i1 %eq to i32
  call i32 (ptr, ...) @printf(ptr @fmt_pair, i32 %eq_i, double %rm_d)

  %rsel1 = call double @df_mixed_select(ptr %buf, double 5.000000e+00, i1 true, i64 1)
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %rsel1)
  %rsel2 = call double @df_mixed_select(ptr %buf, double 5.000000e+00, i1 true, i64 2)
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %rsel2)

  %rv1 = call [2 x double] @df_vec_loop(double 5.000000e+00, i64 1)
  %rv1_0 = extractvalue [2 x double] %rv1, 0
  %rv1_1 = extractvalue [2 x double] %rv1, 1
  call i32 (ptr, ...) @printf(ptr @fmt_vec, double %rv1_0, double %rv1_1)

  %rv2 = call [2 x double] @df_vec_loop(double 5.000000e+00, i64 2)
  %rv2_0 = extractvalue [2 x double] %rv2, 0
  %rv2_1 = extractvalue [2 x double] %rv2, 1
  call i32 (ptr, ...) @printf(ptr @fmt_vec, double %rv2_0, double %rv2_1)

  %rsp1 = call double @df_split_mul()
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %rsp1)

  %rsp2 = call double @df_split_loop(double 5.000000e+00, i64 2)
  call i32 (ptr, ...) @printf(ptr @fmt_f, double %rsp2)

  ret i32 0
}

; CHECK-NOT: dual_phi
; PREOPT-NOT: dual_phi

; CHECK-LABEL: define internal double @fwddiffef_orig()
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }

; CHECK-LABEL: define internal double @fwddiffef_loop_phi_direct(double %x, double %"x'", i64 %n)
; CHECK: loop:
; CHECK:   phi {{(fast )?}}double [ %"x'", %entry ], [ 0.000000e+00, %loop ]
; CHECK-NOT: load double, ptr @g_direct

; CHECK-LABEL: define internal double @fwddiffef_loop_phi_fmul(double %x, double %"x'", i64 %n)
; CHECK: loop:
; CHECK:   phi {{(fast )?}}double [ %"x'", %entry ], [ 0.000000e+00, %loop ]
; CHECK-NOT: load double, ptr @g_fmul

; CHECK-LABEL: define internal double @fwddiffef_extra_ops(i1 %cond)
; CHECK: entry:
; CHECK:   ret double 0.000000e+00
; CHECK-NEXT: }

; CHECK-LABEL: define internal { ptr, double } @fwddiffef_mixed_struct(ptr %x, double %y, double %"y'", i64 %n)
; CHECK: loop:
; CHECK:   %[[S0:.+]] = insertvalue { ptr, double } zeroinitializer, ptr %x, 0
; CHECK:   %[[S:.+]] = insertvalue { ptr, double } %[[S0]], double 0.000000e+00, 1

; CHECK-LABEL: define internal double @fwddiffef_mixed_select(ptr %x, double %y, double %"y'", i1 %c, i64 %n)
; CHECK: loop:
; CHECK:   phi {{(fast )?}}double [ %"y'", %entry ], [ 0.000000e+00, %loop ]

; CHECK-LABEL: define internal [2 x double] @fwddiffe2f_vec_loop(double %x, [2 x double] %"x'", i64 %n)
; CHECK: loop:
; CHECK:   phi {{(fast )?}}double [ %{{.+}}, %entry ], [ 0.000000e+00, %loop ]
; CHECK:   phi {{(fast )?}}double [ %{{.+}}, %entry ], [ 0.000000e+00, %loop ]

; CHECK-LABEL: define internal double @fwddiffef_split_mul(ptr %tapeArg)
; CHECK: ret double 0.000000e+00

; CHECK-LABEL: define internal double @fwddiffef_split_loop(double %x, double %"x'", i64 %n, ptr %tapeArg)
; CHECK: loop:
; CHECK:   phi {{(fast )?}}double [ %"x'", %entry ], [ 0.000000e+00, %loop ]

; EXEC: 0.0
; EXEC-NEXT: 1.0
; EXEC-NEXT: 0.0
; EXEC-NEXT: 0.0
; EXEC-NEXT: 0.0
; EXEC-NEXT: 1 0.0
; EXEC-NEXT: 1.0
; EXEC-NEXT: 0.0
; EXEC-NEXT: 1.0 2.0
; EXEC-NEXT: 0.0 0.0
; EXEC-NEXT: 0.0
; EXEC-NEXT: 0.0
