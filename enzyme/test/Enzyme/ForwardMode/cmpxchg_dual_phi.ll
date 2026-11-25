; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli -

@counter = global i64 0, align 8, !enzyme_inactive !0
@cached_slot = global ptr null, align 8, !enzyme_inactive !0
@sentinel = global i8 0, align 1, !enzyme_inactive !0

declare ptr @malloc(i64)
declare void @free(ptr)
declare void @abort()
declare double @__enzyme_fwddiff(ptr, ...)
declare [2 x double] @__enzyme_fwddiff_v2(ptr, ...)
declare void @__enzyme_fwddiff_void(ptr, ...)
declare void @__enzyme_autodiff(ptr, ...)
declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_fwdsplit(...)

define ptr @make_box(double %x) noinline {
entry:
  %box = call ptr @malloc(i64 8)
  store double %x, ptr %box, align 8
  %init1 = load atomic i64, ptr @counter monotonic, align 8, !enzyme_type !1
  br label %loop1

loop1:
  %old1 = phi i64 [ %init1, %entry ], [ %cur1, %loop1 ]
  %new1 = add i64 %old1, 1
  %cx1 = cmpxchg ptr @counter, i64 %old1, i64 %new1 seq_cst monotonic, align 8
  %cur1 = extractvalue { i64, i1 } %cx1, 0, !enzyme_type !1
  %ok1 = extractvalue { i64, i1 } %cx1, 1
  br i1 %ok1, label %after1, label %loop1

after1:
  %init2 = load atomic i64, ptr @counter monotonic, align 8, !enzyme_type !1
  br label %loop2

loop2:
  %old2 = phi i64 [ %init2, %after1 ], [ %cur2, %loop2 ]
  %new2 = add i64 %old2, 1
  %cx2 = cmpxchg ptr @counter, i64 %old2, i64 %new2 seq_cst monotonic, align 8
  %cur2 = extractvalue { i64, i1 } %cx2, 0, !enzyme_type !1
  %ok2 = extractvalue { i64, i1 } %cx2, 1
  br i1 %ok2, label %exit, label %loop2

exit:
  ret ptr %box
}

define double @square_boxed(double %x) {
entry:
  %box = call ptr @make_box(double %x)
  %val = load double, ptr %box, align 8
  call void @free(ptr %box)
  %sq = fmul double %val, %val
  ret double %sq
}

define void @scale_in_place(ptr %box) noinline {
entry:
  %val = load double, ptr %box, align 8
  %sq = fmul double %val, %val
  store double %sq, ptr %box, align 8
  %init1 = load atomic i64, ptr @counter monotonic, align 8, !enzyme_type !1
  br label %loop1

loop1:
  %old1 = phi i64 [ %init1, %entry ], [ %cur1, %loop1 ]
  %new1 = add i64 %old1, 1
  %cx1 = cmpxchg ptr @counter, i64 %old1, i64 %new1 seq_cst monotonic, align 8
  %cur1 = extractvalue { i64, i1 } %cx1, 0, !enzyme_type !1
  %ok1 = extractvalue { i64, i1 } %cx1, 1
  br i1 %ok1, label %after1, label %loop1

after1:
  %init2 = load atomic i64, ptr @counter monotonic, align 8, !enzyme_type !1
  br label %loop2

loop2:
  %old2 = phi i64 [ %init2, %after1 ], [ %cur2, %loop2 ]
  %new2 = add i64 %old2, 1
  %cx2 = cmpxchg ptr @counter, i64 %old2, i64 %new2 seq_cst monotonic, align 8
  %cur2 = extractvalue { i64, i1 } %cx2, 0, !enzyme_type !1
  %ok2 = extractvalue { i64, i1 } %cx2, 1
  br i1 %ok2, label %exit, label %loop2

exit:
  ret void
}

define void @scale_struct_with_cas(ptr %state) noinline {
entry:
  %val = load double, ptr %state, align 8
  %cube_part = fmul double %val, %val
  %cube = fmul double %cube_part, %val
  store double %cube, ptr %state, align 8
  %count_ptr = getelementptr inbounds { double, i64 }, ptr %state, i64 0, i32 1
  %init = load atomic i64, ptr %count_ptr monotonic, align 8
  br label %cas_loop

cas_loop:
  %old = phi i64 [ %init, %entry ], [ %cur, %cas_loop ]
  %inc = add i64 %old, 1
  %next = urem i64 %inc, 1024
  %cx = cmpxchg ptr %count_ptr, i64 %old, i64 %next seq_cst monotonic, align 8
  %cur = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %exit, label %cas_loop

exit:
  ret void
}

define double @pointer_cas_cube(double %x) noinline {
entry:
  %init_ptr = load atomic ptr, ptr @cached_slot monotonic, align 8
  br label %cas_loop

cas_loop:
  %old_ptr = phi ptr [ %init_ptr, %entry ], [ %cur_ptr, %cas_loop ]
  %cx = cmpxchg ptr @cached_slot, ptr %old_ptr, ptr @sentinel seq_cst monotonic, align 8
  %cur_ptr = extractvalue { ptr, i1 } %cx, 0
  %ok = extractvalue { ptr, i1 } %cx, 1
  br i1 %ok, label %exit, label %cas_loop

exit:
  %sq = fmul double %x, %x
  %cb = fmul double %sq, %x
  ret double %cb
}

define double @diff_scalar(double %x, double %dx) {
entry:
  %res = call double (ptr, ...) @__enzyme_fwddiff(ptr @square_boxed, double %x, double %dx)
  ret double %res
}

define [2 x double] @diff_batch(double %x, double %dx0, double %dx1) {
entry:
  %res = call [2 x double] (ptr, ...) @__enzyme_fwddiff_v2(ptr @square_boxed, metadata !"enzyme_width", i64 2, double %x, double %dx0, double %dx1)
  ret [2 x double] %res
}

define void @diff_split(ptr %box, ptr %dbox) {
entry:
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @scale_in_place, ptr %box, ptr %dbox)
  call void (...) @__enzyme_fwdsplit(ptr @scale_in_place, metadata !"enzyme_nofree", ptr %box, ptr %dbox, ptr %tape)
  ret void
}

define void @diff_struct_fwd(ptr %state, ptr %dstate) {
entry:
  call void (ptr, ...) @__enzyme_fwddiff_void(ptr @scale_struct_with_cas, ptr %state, ptr %dstate)
  ret void
}

define void @diff_struct_rev(ptr %state, ptr %dstate) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @scale_struct_with_cas, ptr %state, ptr %dstate)
  ret void
}

define double @diff_pointer_cas(double %x, double %dx) {
entry:
  %res = call double (ptr, ...) @__enzyme_fwddiff(ptr @pointer_cas_cube, double %x, double %dx)
  ret double %res
}

define i32 @main() {
entry:
  store i64 0, ptr @counter, align 8

  ; d/dx (x^2) at x = 3.0, dx = 2.0 is 12.0; counter increments by 2.
  %d1 = call double @diff_scalar(double 3.0, double 2.0)
  %err1 = fsub double %d1, 12.0
  %bad1_hi = fcmp ogt double %err1, 0.0009765625
  %bad1_lo = fcmp olt double %err1, -0.0009765625
  %bad1 = or i1 %bad1_hi, %bad1_lo
  br i1 %bad1, label %fail, label %check_batch

check_batch:
  ; Batched d/dx (x^2) at x = 4.0, dx = [1.5, -0.5] is [12.0, -4.0]; counter += 2.
  %db = call [2 x double] @diff_batch(double 4.0, double 1.5, double -0.5)
  %db0 = extractvalue [2 x double] %db, 0
  %db1 = extractvalue [2 x double] %db, 1
  %errb0 = fsub double %db0, 12.0
  %errb1 = fadd double %db1, 4.0
  %badb0_hi = fcmp ogt double %errb0, 0.0009765625
  %badb0_lo = fcmp olt double %errb0, -0.0009765625
  %badb1_hi = fcmp ogt double %errb1, 0.0009765625
  %badb1_lo = fcmp olt double %errb1, -0.0009765625
  %badb0 = or i1 %badb0_hi, %badb0_lo
  %badb1 = or i1 %badb1_hi, %badb1_lo
  %badb = or i1 %badb0, %badb1
  br i1 %badb, label %fail, label %check_split

check_split:
  ; Split forward mode at x = 5.0, dx = 3.0 gives sq = 25.0, dsq = 30.0; counter += 2.
  %box = alloca double, align 8
  %dbox = alloca double, align 8
  store double 5.0, ptr %box, align 8
  store double 3.0, ptr %dbox, align 8
  call void @diff_split(ptr %box, ptr %dbox)
  %primal_out = load double, ptr %box, align 8
  %shadow_out = load double, ptr %dbox, align 8
  %errs_p = fsub double %primal_out, 25.0
  %errs_d = fsub double %shadow_out, 30.0
  %bads_p_hi = fcmp ogt double %errs_p, 0.0009765625
  %bads_p_lo = fcmp olt double %errs_p, -0.0009765625
  %bads_d_hi = fcmp ogt double %errs_d, 0.0009765625
  %bads_d_lo = fcmp olt double %errs_d, -0.0009765625
  %bads_p = or i1 %bads_p_hi, %bads_p_lo
  %bads_d = or i1 %bads_d_hi, %bads_d_lo
  %bads = or i1 %bads_p, %bads_d
  br i1 %bads, label %fail, label %check_struct_fwd

check_struct_fwd:
  ; Forward mode on active { double, i64 } struct at x = 2.0, dx = 1.5 gives 3 * 4 * 1.5 = 18.0.
  %state_fwd = alloca { double, i64 }, align 8
  %dstate_fwd = alloca { double, i64 }, align 8
  store double 2.0, ptr %state_fwd, align 8
  %cnt_fwd = getelementptr inbounds { double, i64 }, ptr %state_fwd, i64 0, i32 1
  store i64 7, ptr %cnt_fwd, align 8
  store double 1.5, ptr %dstate_fwd, align 8
  %dcnt_fwd = getelementptr inbounds { double, i64 }, ptr %dstate_fwd, i64 0, i32 1
  store i64 0, ptr %dcnt_fwd, align 8
  call void @diff_struct_fwd(ptr %state_fwd, ptr %dstate_fwd)
  %sf_d = load double, ptr %dstate_fwd, align 8
  %sf_cnt = load i64, ptr %cnt_fwd, align 8
  %err_sf = fsub double %sf_d, 18.0
  %bad_sf_hi = fcmp ogt double %err_sf, 0.0009765625
  %bad_sf_lo = fcmp olt double %err_sf, -0.0009765625
  %bad_sf_cnt = icmp ne i64 %sf_cnt, 8
  %bad_sf0 = or i1 %bad_sf_hi, %bad_sf_lo
  %bad_sf = or i1 %bad_sf0, %bad_sf_cnt
  br i1 %bad_sf, label %fail, label %check_struct_rev

check_struct_rev:
  ; Reverse mode on active { double, i64 } struct at x = 2.0, dy = 2.5 gives 3 * 4 * 2.5 = 30.0.
  %state_rev = alloca { double, i64 }, align 8
  %dstate_rev = alloca { double, i64 }, align 8
  store double 2.0, ptr %state_rev, align 8
  %cnt_rev = getelementptr inbounds { double, i64 }, ptr %state_rev, i64 0, i32 1
  store i64 11, ptr %cnt_rev, align 8
  store double 2.5, ptr %dstate_rev, align 8
  %dcnt_rev = getelementptr inbounds { double, i64 }, ptr %dstate_rev, i64 0, i32 1
  store i64 0, ptr %dcnt_rev, align 8
  call void @diff_struct_rev(ptr %state_rev, ptr %dstate_rev)
  %sr_d = load double, ptr %dstate_rev, align 8
  %sr_cnt = load i64, ptr %cnt_rev, align 8
  %err_sr = fsub double %sr_d, 30.0
  %bad_sr_hi = fcmp ogt double %err_sr, 0.0009765625
  %bad_sr_lo = fcmp olt double %err_sr, -0.0009765625
  %bad_sr_cnt = icmp ne i64 %sr_cnt, 12
  %bad_sr0 = or i1 %bad_sr_hi, %bad_sr_lo
  %bad_sr = or i1 %bad_sr0, %bad_sr_cnt
  br i1 %bad_sr, label %fail, label %check_ptr_cas

check_ptr_cas:
  ; Pointer cmpxchg on inactive global slot: d/dx (x^3) at x = 2.0, dx = 0.5 is 6.0.
  %dp = call double @diff_pointer_cas(double 2.0, double 0.5)
  %err_dp = fsub double %dp, 6.0
  %bad_dp_hi = fcmp ogt double %err_dp, 0.0009765625
  %bad_dp_lo = fcmp olt double %err_dp, -0.0009765625
  %slot_after = load ptr, ptr @cached_slot, align 8
  %bad_slot = icmp ne ptr %slot_after, @sentinel
  %bad_dp0 = or i1 %bad_dp_hi, %bad_dp_lo
  %bad_dp = or i1 %bad_dp0, %bad_slot
  br i1 %bad_dp, label %fail, label %check_counter

check_counter:
  %final_count = load i64, ptr @counter, align 8
  %bad_count = icmp ne i64 %final_count, 6
  br i1 %bad_count, label %fail, label %done

fail:
  call void @abort()
  unreachable

done:
  ret i32 0
}

!0 = !{}
!1 = !{!"Unknown", i32 -1, !2}
!2 = !{!"Integer"}

; CHECK-LABEL: define internal { ptr, ptr } @fwddiffemake_box(
; CHECK-NOT: dual_phi
; CHECK: cmpxchg ptr @counter, i64 %old1, i64 %new1 seq_cst monotonic, align 8
; CHECK-NOT: dual_phi
; CHECK: cmpxchg ptr @counter, i64 %old2, i64 %new2 seq_cst monotonic, align 8
; CHECK-NOT: dual_phi
; CHECK: ret { ptr, ptr }

; CHECK-LABEL: define internal { ptr, [2 x ptr] } @fwddiffe2make_box(
; CHECK-NOT: dual_phi
; CHECK: cmpxchg ptr @counter, i64 %old1, i64 %new1 seq_cst monotonic, align 8
; CHECK-NOT: dual_phi
; CHECK: cmpxchg ptr @counter, i64 %old2, i64 %new2 seq_cst monotonic, align 8
; CHECK-NOT: dual_phi
; CHECK: ret { ptr, [2 x ptr] }

; CHECK-LABEL: define internal void @fwddiffescale_in_place(
; CHECK-NOT: dual_phi
; CHECK: ret void

; CHECK-LABEL: define internal void @fwddiffescale_struct_with_cas(
; CHECK-NOT: dual_phi
; CHECK: cmpxchg ptr %count_ptr, i64 %old, i64 %next seq_cst monotonic, align 8
; CHECK-NOT: dual_phi
; CHECK: ret void
