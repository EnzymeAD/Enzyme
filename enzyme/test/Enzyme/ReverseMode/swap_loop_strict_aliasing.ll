; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -enzyme-preopt=false -S | %lli -

@enzyme_const = external global i32
@enzyme_dupnoneed = external global i32
@.str = private unnamed_addr constant [32 x i8] c"%.1f %.1f %.1f %.1f %.1f\0A\00\00\00\00\00\00\00", align 1

declare void @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr nocapture readonly, ...)
declare void @abort()

define void @f_chained(i64 %n, ptr nocapture %d) {
entry:
  %cmp = icmp eq i64 %n, 0
  br i1 %cmp, label %exit, label %loop

loop:
  %i = phi i64 [ %next, %loop ], [ 0, %entry ]
  %idx = shl i64 %i, 2
  %base = getelementptr inbounds double, ptr %d, i64 %idx
  %a = getelementptr inbounds double, ptr %base, i64 2
  %b = getelementptr inbounds double, ptr %base, i64 4
  %va = load i64, ptr %a, align 8
  %vb = load i64, ptr %b, align 8
  store i64 %vb, ptr %a, align 8
  store i64 %va, ptr %b, align 8
  %next = add i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  %d2 = getelementptr inbounds double, ptr %d, i64 2
  %v2 = load double, ptr %d2, align 8
  %v0 = load double, ptr %d, align 8
  %sum = fadd double %v2, %v0
  store double %sum, ptr %d, align 8
  ret void
}

define void @f_unrolled(i64 %n, ptr nocapture %d) {
entry:
  %d4 = getelementptr double, ptr %d, i64 4
  %is_zero = icmp eq i64 %n, 0
  br i1 %is_zero, label %exit, label %check

check:
  %odd = and i64 %n, 1
  %is_one = icmp eq i64 %n, 1
  br i1 %is_one, label %epilogue_check, label %preheader

preheader:
  %even_n = and i64 %n, -2
  br label %unrolled_loop

unrolled_loop:
  %iv = phi i64 [ 0, %preheader ], [ %iv_next, %unrolled_loop ]
  %off0 = shl i64 %iv, 2
  %idx_a0 = or i64 %off0, 2
  %ptr_a0 = getelementptr inbounds double, ptr %d, i64 %idx_a0
  %val_a0 = load i64, ptr %ptr_a0, align 8
  %ptr_b0 = getelementptr double, ptr %d4, i64 %off0
  %val_b0 = load i64, ptr %ptr_b0, align 8
  store i64 %val_b0, ptr %ptr_a0, align 8
  store i64 %val_a0, ptr %ptr_b0, align 8
  %iv_next = add i64 %iv, 2
  %loop_done = icmp eq i64 %iv_next, %even_n
  br i1 %loop_done, label %loop_exit, label %unrolled_loop

loop_exit:
  %unrolled_end = shl i64 %iv_next, 2
  br label %epilogue_check

epilogue_check:
  %epilogue_off = phi i64 [ 0, %check ], [ %unrolled_end, %loop_exit ]
  %skip_epilogue = icmp eq i64 %odd, 0
  br i1 %skip_epilogue, label %exit, label %epilogue

epilogue:
  %epi_idx_a = or i64 %epilogue_off, 2
  %epi_ptr_a = getelementptr inbounds double, ptr %d, i64 %epi_idx_a
  %epi_val_a = load i64, ptr %epi_ptr_a, align 8
  %epi_ptr_b = getelementptr double, ptr %d4, i64 %epilogue_off
  %epi_val_b = load i64, ptr %epi_ptr_b, align 8
  store i64 %epi_val_b, ptr %epi_ptr_a, align 8
  store i64 %epi_val_a, ptr %epi_ptr_b, align 8
  br label %exit

exit:
  %d2 = getelementptr inbounds double, ptr %d, i64 2
  %v2 = load double, ptr %d2, align 8
  %v0 = load double, ptr %d, align 8
  %sum = fadd double %v2, %v0
  store double %sum, ptr %d, align 8
  ret void
}

define void @check_res(ptr %dx) {
entry:
  %p2 = getelementptr inbounds double, ptr %dx, i64 2
  %p4 = getelementptr inbounds double, ptr %dx, i64 4
  %v2 = load double, ptr %p2, align 8
  %v4 = load double, ptr %p4, align 8
  %c2 = fcmp une double %v2, 5.000000e+00
  %c4 = fcmp une double %v4, 4.000000e+00
  %bad = or i1 %c2, %c4
  br i1 %bad, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret void
}

define i32 @main() {
entry:
  %x = alloca [5 x double], align 16
  %dx = alloca [5 x double], align 16
  %dx0 = getelementptr inbounds [5 x double], ptr %dx, i64 0, i64 0
  %dx1 = getelementptr inbounds [5 x double], ptr %dx, i64 0, i64 1
  %dx2 = getelementptr inbounds [5 x double], ptr %dx, i64 0, i64 2
  %dx3 = getelementptr inbounds [5 x double], ptr %dx, i64 0, i64 3
  %dx4 = getelementptr inbounds [5 x double], ptr %dx, i64 0, i64 4

  store double 0.000000e+00, ptr %x, align 16
  store double 1.000000e+00, ptr %dx0, align 16
  store double 2.000000e+00, ptr %dx1, align 8
  store double 3.000000e+00, ptr %dx2, align 16
  store double 4.000000e+00, ptr %dx3, align 8
  store double 5.000000e+00, ptr %dx4, align 16
  call void (ptr, ...) @__enzyme_autodiff(ptr @f_chained, metadata !"enzyme_const", i64 1, metadata !"enzyme_dupnoneed", ptr %x, ptr %dx)
  call void @check_res(ptr %dx)

  store double 1.000000e+00, ptr %dx0, align 16
  store double 2.000000e+00, ptr %dx1, align 8
  store double 3.000000e+00, ptr %dx2, align 16
  store double 4.000000e+00, ptr %dx3, align 8
  store double 5.000000e+00, ptr %dx4, align 16
  call void (ptr, ...) @__enzyme_autodiff(ptr @f_unrolled, metadata !"enzyme_const", i64 1, metadata !"enzyme_dupnoneed", ptr %x, ptr %dx)
  call void @check_res(ptr %dx)
  ret i32 0
}

; CHECK-LABEL: define internal void @diffef_chained(
; CHECK: invertloop:
; CHECK: %[[CB:.+]] = load i64, ptr %"b'ipg_unwrap"
; CHECK-NEXT: store i64 0, ptr %"b'ipg_unwrap"
; CHECK: %[[CA:.+]] = load i64, ptr %"a'ipg_unwrap"
; CHECK-NEXT: store i64 0, ptr %"a'ipg_unwrap"

; CHECK-LABEL: define internal void @diffef_unrolled(
; CHECK: invertepilogue_phimerge:
; CHECK: %[[EB:.+]] = load i64, ptr %"epi_ptr_b'ipg_unwrap"
; CHECK-NEXT: store i64 0, ptr %"epi_ptr_b'ipg_unwrap"
; CHECK: %[[EA:.+]] = load i64, ptr %"epi_ptr_a'ipg_unwrap"
; CHECK-NEXT: store i64 0, ptr %"epi_ptr_a'ipg_unwrap"
; CHECK: store double %{{.+}}, ptr %"epi_ptr_b'ipg_unwrap"
; CHECK: store double %{{.+}}, ptr %"epi_ptr_a'ipg_unwrap"
