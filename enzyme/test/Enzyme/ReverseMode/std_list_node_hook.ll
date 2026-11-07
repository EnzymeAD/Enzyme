; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: const_len=1 const_val=7.000000
; EVAL: rev_dx=72.000000 rev_dy=100.000000
; EVAL: splice_swap_da=6.000000 splice_swap_db=20.000000
; EVAL: batch0=72.000000 batch1=200.000000
; EVAL: fwd=172.000000
; EVAL: split_primal=13.000000 split_da=6.000000 split_db=20.000000

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr nocapture readonly, ...)
declare void @abort()

@enzyme_const = external global i32
@enzyme_dup = external global i32
@enzyme_width = external global i32

declare void @__enzyme_autodiff_void(ptr, ...)
declare double @__enzyme_autodiff_1(ptr, ...)
declare { double, double } @__enzyme_autodiff_2(ptr, ...)
declare { [2 x double], [2 x double] } @__enzyme_autodiff_batch2(ptr, ...)
declare double @__enzyme_fwddiff(ptr, ...)
declare { ptr, double } @__enzyme_augmentfwd(ptr, ...)
declare { double, double } @__enzyme_reverse(ptr, ...)

; External legacy symbol declaration without a body to verify compile-time
; handling of unlinked libstdc++ declarations.
declare void @_ZNSt15_List_node_base7_M_hookEPS_(ptr, ptr)

@.str_const = private unnamed_addr constant [27 x i8] c"const_len=%d const_val=%f\0A\00", align 1
@.str_rev = private unnamed_addr constant [21 x i8] c"rev_dx=%f rev_dy=%f\0A\00", align 1
@.str_splice = private unnamed_addr constant [37 x i8] c"splice_swap_da=%f splice_swap_db=%f\0A\00", align 1
@.str_batch = private unnamed_addr constant [21 x i8] c"batch0=%f batch1=%f\0A\00", align 1
@.str_fwd = private unnamed_addr constant [8 x i8] c"fwd=%f\0A\00", align 1
@.str_split = private unnamed_addr constant [41 x i8] c"split_primal=%f split_da=%f split_db=%f\0A\00", align 1

define void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %this, ptr %pos) noinline {
entry:
  store ptr %pos, ptr %this, align 8
  %pos_prev_slot = getelementptr inbounds i8, ptr %pos, i64 8
  %prev = load ptr, ptr %pos_prev_slot, align 8
  %this_prev_slot = getelementptr inbounds i8, ptr %this, i64 8
  store ptr %prev, ptr %this_prev_slot, align 8
  store ptr %this, ptr %prev, align 8
  store ptr %this, ptr %pos_prev_slot, align 8
  ret void
}

define void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr %this) noinline {
entry:
  %next = load ptr, ptr %this, align 8
  %this_prev_slot = getelementptr inbounds i8, ptr %this, i64 8
  %prev = load ptr, ptr %this_prev_slot, align 8
  store ptr %next, ptr %prev, align 8
  %next_prev_slot = getelementptr inbounds i8, ptr %next, i64 8
  store ptr %prev, ptr %next_prev_slot, align 8
  ret void
}

define void @_ZNSt8__detail15_List_node_base10_M_reverseEv(ptr %this) noinline {
entry:
  br label %loop

loop:
  %cur = phi ptr [ %this, %entry ], [ %old_next, %loop ]
  %old_next = load ptr, ptr %cur, align 8
  %prev_slot = getelementptr inbounds i8, ptr %cur, i64 8
  %old_prev = load ptr, ptr %prev_slot, align 8
  store ptr %old_prev, ptr %cur, align 8
  store ptr %old_next, ptr %prev_slot, align 8
  %done = icmp eq ptr %old_next, %this
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr %this, ptr %first, ptr %last) noinline {
entry:
  %same = icmp eq ptr %this, %last
  br i1 %same, label %exit, label %do_transfer

do_transfer:
  %last_prev_slot = getelementptr inbounds i8, ptr %last, i64 8
  %last_prev = load ptr, ptr %last_prev_slot, align 8
  %first_prev_slot = getelementptr inbounds i8, ptr %first, i64 8
  %first_prev = load ptr, ptr %first_prev_slot, align 8
  %this_prev_slot = getelementptr inbounds i8, ptr %this, i64 8
  %this_prev = load ptr, ptr %this_prev_slot, align 8
  store ptr %this, ptr %last_prev, align 8
  store ptr %last, ptr %first_prev, align 8
  store ptr %first, ptr %this_prev, align 8
  store ptr %last_prev, ptr %this_prev_slot, align 8
  store ptr %first_prev, ptr %last_prev_slot, align 8
  store ptr %this_prev, ptr %first_prev_slot, align 8
  br label %exit

exit:
  ret void
}

define void @_ZNSt8__detail15_List_node_base4swapERS0_S1_(ptr %x, ptr %y) noinline {
entry:
  %x_next = load ptr, ptr %x, align 8
  %y_next = load ptr, ptr %y, align 8
  %x_prev_slot = getelementptr inbounds i8, ptr %x, i64 8
  %y_prev_slot = getelementptr inbounds i8, ptr %y, i64 8
  %x_prev = load ptr, ptr %x_prev_slot, align 8
  %y_prev = load ptr, ptr %y_prev_slot, align 8
  %x_nonempty = icmp ne ptr %x_next, %x
  %y_nonempty = icmp ne ptr %y_next, %y
  br i1 %x_nonempty, label %x_has, label %x_empty

x_has:
  br i1 %y_nonempty, label %both_nonempty, label %only_x

both_nonempty:
  store ptr %y_next, ptr %x, align 8
  store ptr %x_next, ptr %y, align 8
  store ptr %y_prev, ptr %x_prev_slot, align 8
  store ptr %x_prev, ptr %y_prev_slot, align 8
  %yn_prev_slot = getelementptr inbounds i8, ptr %y_next, i64 8
  store ptr %x, ptr %yn_prev_slot, align 8
  store ptr %x, ptr %y_prev, align 8
  %xn_prev_slot = getelementptr inbounds i8, ptr %x_next, i64 8
  store ptr %y, ptr %xn_prev_slot, align 8
  store ptr %y, ptr %x_prev, align 8
  ret void

only_x:
  store ptr %x_next, ptr %y, align 8
  store ptr %x_prev, ptr %y_prev_slot, align 8
  %xn_prev_slot2 = getelementptr inbounds i8, ptr %x_next, i64 8
  store ptr %y, ptr %xn_prev_slot2, align 8
  store ptr %y, ptr %x_prev, align 8
  store ptr %x, ptr %x, align 8
  store ptr %x, ptr %x_prev_slot, align 8
  ret void

x_empty:
  br i1 %y_nonempty, label %only_y, label %done

only_y:
  store ptr %y_next, ptr %x, align 8
  store ptr %y_prev, ptr %x_prev_slot, align 8
  %yn_prev_slot2 = getelementptr inbounds i8, ptr %y_next, i64 8
  store ptr %x, ptr %yn_prev_slot2, align 8
  store ptr %x, ptr %y_prev, align 8
  store ptr %y, ptr %y, align 8
  store ptr %y, ptr %y_prev_slot, align 8
  ret void

done:
  ret void
}

; External legacy _List_node_base::_M_hook test (unlinked body).
define double @legacy_external_hook(double %x, ptr %head) {
entry:
  %node = call noalias ptr @malloc(i64 24)
  %val_slot = getelementptr inbounds i8, ptr %node, i64 16
  store double %x, ptr %val_slot, align 8
  call void @_ZNSt15_List_node_base7_M_hookEPS_(ptr %node, ptr %head)
  %first = load ptr, ptr %head, align 8
  %first_val_slot = getelementptr inbounds i8, ptr %first, i64 16
  %res = load double, ptr %first_val_slot, align 8
  ret double %res
}

define double @test_legacy_external(double %x, ptr %head, ptr %dhead) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  %r = call double (ptr, ...) @__enzyme_autodiff_1(ptr @legacy_external_hook, double %x, i32 %dup, ptr %head, ptr %dhead)
  ret double %r
}

; Issue #2369 pattern: pushing an active double into an enzyme_const std::list.
define void @push_into_const_list(double %x, ptr %head, ptr %out_len, ptr %out_val) {
entry:
  %node = call noalias ptr @malloc(i64 24)
  %val_slot = getelementptr inbounds i8, ptr %node, i64 16
  store double %x, ptr %val_slot, align 8
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %node, ptr %head)
  %cfirst = load ptr, ptr %head, align 8
  %cval_ptr = getelementptr inbounds i8, ptr %cfirst, i64 16
  %cval = load double, ptr %cval_ptr, align 8
  %cnext = load ptr, ptr %cfirst, align 8
  %only_one = icmp eq ptr %cnext, %head
  %clen = select i1 %only_one, i32 1, i32 2
  store i32 %clen, ptr %out_len, align 4
  store double %cval, ptr %out_val, align 8
  ret void
}

; Active local C++11 list header ({ ptr _M_next, ptr _M_prev, i64 _M_size })
; exercising _M_hook (including a constant middle node), _M_reverse, weighted
; traversal, _M_unhook, and deallocation.
define double @list_hook_reverse_unhook(double %x, double %y) {
entry:
  %head = alloca { ptr, ptr, i64 }, align 8
  %head_prev = getelementptr inbounds i8, ptr %head, i64 8
  %head_size = getelementptr inbounds i8, ptr %head, i64 16
  store ptr %head, ptr %head, align 8
  store ptr %head, ptr %head_prev, align 8
  store i64 0, ptr %head_size, align 8

  %n1 = call noalias ptr @malloc(i64 24)
  %v1 = fmul double %x, 2.000000e+00
  %n1_val = getelementptr inbounds i8, ptr %n1, i64 16
  store double %v1, ptr %n1_val, align 8
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %n1, ptr %head)
  store i64 1, ptr %head_size, align 8

  %n2 = call noalias ptr @malloc(i64 24)
  %n2_val = getelementptr inbounds i8, ptr %n2, i64 16
  store double 3.000000e+00, ptr %n2_val, align 8
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %n2, ptr %head)

  %n3 = call noalias ptr @malloc(i64 24)
  %v3 = fmul double %y, 5.000000e+00
  %n3_val = getelementptr inbounds i8, ptr %n3, i64 16
  store double %v3, ptr %n3_val, align 8
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %n3, ptr %head)

  call void @_ZNSt8__detail15_List_node_base10_M_reverseEv(ptr %head)

  %first = load ptr, ptr %head, align 8
  br label %sum_loop

sum_loop:
  %it = phi ptr [ %first, %entry ], [ %next, %sum_loop ]
  %weight = phi double [ 1.000000e+00, %entry ], [ %next_weight, %sum_loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %next_acc, %sum_loop ]
  %val_ptr = getelementptr inbounds i8, ptr %it, i64 16
  %val = load double, ptr %val_ptr, align 8
  %sq = fmul double %val, %val
  %term = fmul double %weight, %sq
  %next_acc = fadd double %acc, %term
  %next_weight = fadd double %weight, 1.000000e+00
  %next = load ptr, ptr %it, align 8
  %done_sum = icmp eq ptr %next, %head
  br i1 %done_sum, label %after_sum, label %sum_loop

after_sum:
  %pop_node = load ptr, ptr %head, align 8
  call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(ptr %pop_node)
  call void @free(ptr %pop_node)
  %rem_first = load ptr, ptr %head, align 8
  br label %free_loop

free_loop:
  %fit = phi ptr [ %rem_first, %after_sum ], [ %fnext, %free_loop ]
  %fnext = load ptr, ptr %fit, align 8
  call void @free(ptr %fit)
  %done_free = icmp eq ptr %fnext, %head
  br i1 %done_free, label %exit, label %free_loop

exit:
  ret double %next_acc
}

; Active list exercising _M_transfer (splice) and swap.
define double @list_splice_and_swap(double %a, double %b) {
entry:
  %list1 = alloca [2 x ptr], align 8
  %list1_prev = getelementptr inbounds i8, ptr %list1, i64 8
  store ptr %list1, ptr %list1, align 8
  store ptr %list1, ptr %list1_prev, align 8

  %list2 = alloca [2 x ptr], align 8
  %list2_prev = getelementptr inbounds i8, ptr %list2, i64 8
  store ptr %list2, ptr %list2, align 8
  store ptr %list2, ptr %list2_prev, align 8

  %list3 = alloca [2 x ptr], align 8
  %list3_prev = getelementptr inbounds i8, ptr %list3, i64 8
  store ptr %list3, ptr %list3, align 8
  store ptr %list3, ptr %list3_prev, align 8

  %na = call noalias ptr @malloc(i64 24)
  %na_val = getelementptr inbounds i8, ptr %na, i64 16
  store double %a, ptr %na_val, align 8
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %na, ptr %list1)

  %nb = call noalias ptr @malloc(i64 24)
  %nb_val = getelementptr inbounds i8, ptr %nb, i64 16
  store double %b, ptr %nb_val, align 8
  call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %nb, ptr %list2)

  %l1_first = load ptr, ptr %list1, align 8
  call void @_ZNSt8__detail15_List_node_base11_M_transferEPS0_S1_(ptr %list2, ptr %l1_first, ptr %list1)
  call void @_ZNSt8__detail15_List_node_base4swapERS0_S1_(ptr %list2, ptr %list3)

  ; list3 now holds [b, a]; compute 10 * b^2 + 3 * a^2 = 10*v0^2 + 3*v1^2.
  %c0 = load ptr, ptr %list3, align 8
  %c0_val_ptr = getelementptr inbounds i8, ptr %c0, i64 16
  %c0_val = load double, ptr %c0_val_ptr, align 8
  %c1 = load ptr, ptr %c0, align 8
  %c1_val_ptr = getelementptr inbounds i8, ptr %c1, i64 16
  %c1_val = load double, ptr %c1_val_ptr, align 8

  %s0 = fmul double %c0_val, %c0_val
  %t0 = fmul double %s0, 1.000000e+01
  %s1 = fmul double %c1_val, %c1_val
  %t1 = fmul double %s1, 3.000000e+00
  %res = fadd double %t0, %t1

  call void @free(ptr %c0)
  call void @free(ptr %c1)
  ret double %res
}

define i32 @main() {
entry:
  ; 1. Constant list insertion (#2369): ensure no shadow node corrupts head.
  %chead = alloca [2 x ptr], align 8
  %out_len = alloca i32, align 4
  %out_val = alloca double, align 8
  %chead_prev = getelementptr inbounds i8, ptr %chead, i64 8
  store ptr %chead, ptr %chead, align 8
  store ptr %chead, ptr %chead_prev, align 8
  %cflag = load i32, ptr @enzyme_const, align 4
  call void (ptr, ...) @__enzyme_autodiff_void(ptr @push_into_const_list, double 7.000000e+00, i32 %cflag, ptr %chead, i32 %cflag, ptr %out_len, i32 %cflag, ptr %out_val)
  %clen = load i32, ptr %out_len, align 4
  %cval = load double, ptr %out_val, align 8
  %only_one = icmp eq i32 %clen, 1
  call i32 (ptr, ...) @printf(ptr @.str_const, i32 %clen, double %cval)
  br i1 %only_one, label %cont1, label %fail

cont1:
  ; 2. Reverse-mode gradient on _M_hook + _M_reverse + _M_unhook:
  ;    f(3, 2) = 1*(5*2)^2 + 2*(3)^2 + 3*(2*3)^2 = 100 + 18 + 108 = 226
  ;    df/dx = 24*x = 72, df/dy = 50*y = 100.
  %g = call { double, double } (ptr, ...) @__enzyme_autodiff_2(ptr @list_hook_reverse_unhook, double 3.000000e+00, double 2.000000e+00)
  %dx = extractvalue { double, double } %g, 0
  %dy = extractvalue { double, double } %g, 1
  call i32 (ptr, ...) @printf(ptr @.str_rev, double %dx, double %dy)

  ; 3. Reverse-mode gradient on _M_transfer + swap:
  ;    f(1, 1) -> f(a=1, b=1): df/da = 6*a = 6, df/db = 20*b = 20.
  %gs = call { double, double } (ptr, ...) @__enzyme_autodiff_2(ptr @list_splice_and_swap, double 1.000000e+00, double 1.000000e+00)
  %da = extractvalue { double, double } %gs, 0
  %db = extractvalue { double, double } %gs, 1
  call i32 (ptr, ...) @printf(ptr @.str_splice, double %da, double %db)

  ; 4. Batched reverse mode (width = 2) on list_hook_reverse_unhook:
  ;    lane 0 seeds = 1.0 -> dx = 72; lane 1 seeds = 2.0 -> dy = 2 * 100 = 200.
  %wflag = load i32, ptr @enzyme_width, align 4
  %gb = call { [2 x double], [2 x double] } (ptr, ...) @__enzyme_autodiff_batch2(ptr @list_hook_reverse_unhook, i32 %wflag, i32 2, double 3.000000e+00, double 2.000000e+00)
  %b0 = extractvalue { [2 x double], [2 x double] } %gb, 0, 0
  %b1 = extractvalue { [2 x double], [2 x double] } %gb, 1, 1
  %b1_scaled = fmul double %b1, 2.000000e+00
  call i32 (ptr, ...) @printf(ptr @.str_batch, double %b0, double %b1_scaled)

  ; 5. Forward mode tangent: dx=1, dy=1 at (3, 2) -> 72*1 + 100*1 = 172.
  %dup = load i32, ptr @enzyme_dup, align 4
  %fwd = call double (ptr, ...) @__enzyme_fwddiff(ptr @list_hook_reverse_unhook, i32 %dup, double 3.000000e+00, double 1.000000e+00, i32 %dup, double 2.000000e+00, double 1.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.str_fwd, double %fwd)

  ; 6. Split reverse mode (__enzyme_augmentfwd + __enzyme_reverse).
  %aug = call { ptr, double } (ptr, ...) @__enzyme_augmentfwd(ptr @list_splice_and_swap, double 1.000000e+00, double 1.000000e+00)
  %tape = extractvalue { ptr, double } %aug, 0
  %split_primal = extractvalue { ptr, double } %aug, 1
  %split_grad = call { double, double } (ptr, ...) @__enzyme_reverse(ptr @list_splice_and_swap, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, ptr %tape)
  %split_da = extractvalue { double, double } %split_grad, 0
  %split_db = extractvalue { double, double } %split_grad, 1
  call i32 (ptr, ...) @printf(ptr @.str_split, double %split_primal, double %split_da, double %split_db)
  ret i32 0

fail:
  call void @abort()
  unreachable
}

; CHECK-LABEL: define internal {{.*}}@diffelegacy_external_hook(
; CHECK: call void @_ZNSt15_List_node_base7_M_hookEPS_(ptr %[[DNODE:.+]], ptr %[[DHEAD:.+]])
; CHECK: call void @_ZNSt15_List_node_base7_M_hookEPS_(ptr %[[NODE:.+]], ptr %[[HEAD:.+]])

; CHECK-LABEL: define internal { double } @diffepush_into_const_list(
; CHECK: call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(ptr %[[CNODE:.+]], ptr %[[CHEAD:.+]])
; CHECK-NOT: call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(
; CHECK: ret { double }

; CHECK-LABEL: define internal { double, double } @diffelist_hook_reverse_unhook(
; CHECK: call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(
; CHECK: call void @_ZNSt8__detail15_List_node_base7_M_hookEPS0_(
; CHECK: call void @_ZNSt8__detail15_List_node_base10_M_reverseEv(
; CHECK: call void @_ZNSt8__detail15_List_node_base10_M_reverseEv(
; CHECK: call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(
; CHECK: call void @_ZNSt8__detail15_List_node_base9_M_unhookEv(
