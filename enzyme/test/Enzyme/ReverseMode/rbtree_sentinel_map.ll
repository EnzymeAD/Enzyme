; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 19.000000

@.fmt = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare double @__enzyme_autodiff(...)

; Map header layout (48 bytes):
;   [0..7]:   comparator byte / padding
;   [8..39]:  embedded sentinel _Rb_tree_node_base:
;               [8..11]  i32 color
;               [16..23] ptr parent (root)
;               [24..31] ptr leftmost
;               [32..39] ptr rightmost
;   [40..47]: i64 node_count
;
; Heap _Rb_tree_node layout (48 bytes):
;   [0..3]:   i32 color
;   [8..15]:  ptr parent
;   [16..23]: ptr left
;   [24..31]: ptr right
;   [32..39]: double key
;   [40..47]: double value

define internal void @map_init(ptr %map) {
entry:
  %header = getelementptr inbounds i8, ptr %map, i64 8
  store i32 0, ptr %header, align 8
  %parent = getelementptr inbounds i8, ptr %map, i64 16
  store ptr null, ptr %parent, align 8
  %left = getelementptr inbounds i8, ptr %map, i64 24
  store ptr %header, ptr %left, align 8
  %right = getelementptr inbounds i8, ptr %map, i64 32
  store ptr %header, ptr %right, align 8
  %count = getelementptr inbounds i8, ptr %map, i64 40
  store i64 0, ptr %count, align 8, !tbaa !0
  ret void
}

define internal { ptr, ptr } @hint_insert_pos(ptr %map, ptr %hint, double %key) {
entry:
  %end = getelementptr inbounds i8, ptr %map, i64 8
  %is_end = icmp eq ptr %end, %hint
  br i1 %is_end, label %at_end, label %check_hint

check_hint:
  %hint_key_ptr = getelementptr inbounds i8, ptr %hint, i64 32
  %hint_key = load double, ptr %hint_key_ptr, align 8
  %lt_hint = fcmp olt double %key, %hint_key
  %res_first_h = select i1 %lt_hint, ptr %hint, ptr null
  %res_second_h = select i1 %lt_hint, ptr %hint, ptr %end
  br label %ret_block

at_end:
  %root_slot = getelementptr inbounds i8, ptr %map, i64 16
  %root = load ptr, ptr %root_slot, align 8
  %empty = icmp eq ptr %root, null
  br i1 %empty, label %empty_tree, label %walk_right

empty_tree:
  br label %ret_block

walk_right:
  %rightmost_slot = getelementptr inbounds i8, ptr %map, i64 32
  %rightmost = load ptr, ptr %rightmost_slot, align 8
  br label %ret_block

ret_block:
  %first = phi ptr [ %res_first_h, %check_hint ], [ null, %empty_tree ], [ null, %walk_right ]
  %second = phi ptr [ %res_second_h, %check_hint ], [ %end, %empty_tree ], [ %rightmost, %walk_right ]
  %ins0 = insertvalue { ptr, ptr } undef, ptr %first, 0
  %ins1 = insertvalue { ptr, ptr } %ins0, ptr %second, 1
  ret { ptr, ptr } %ins1
}

define internal ptr @map_operator_index(ptr %map, double %key) {
entry:
  %root_slot = getelementptr inbounds i8, ptr %map, i64 16
  %root = load ptr, ptr %root_slot, align 8
  %end = getelementptr inbounds i8, ptr %map, i64 8
  %is_empty = icmp eq ptr %root, null
  br i1 %is_empty, label %insert_node, label %search_loop

search_loop:
  %cur = phi ptr [ %root, %entry ], [ %next, %search_loop ]
  %best = phi ptr [ %end, %entry ], [ %cand, %search_loop ]
  %cur_key_ptr = getelementptr inbounds i8, ptr %cur, i64 32
  %cur_key = load double, ptr %cur_key_ptr, align 8
  %cur_lt_key = fcmp olt double %cur_key, %key
  %right_ptr = getelementptr inbounds i8, ptr %cur, i64 24
  %left_ptr = getelementptr inbounds i8, ptr %cur, i64 16
  %cand = select i1 %cur_lt_key, ptr %best, ptr %cur
  %next_slot = select i1 %cur_lt_key, ptr %right_ptr, ptr %left_ptr
  %next = load ptr, ptr %next_slot, align 8
  %done = icmp eq ptr %next, null
  br i1 %done, label %lower_bound_exit, label %search_loop

lower_bound_exit:
  %cand_is_end = icmp eq ptr %cand, %end
  br i1 %cand_is_end, label %insert_node, label %check_equal

check_equal:
  %best_key_gep = getelementptr inbounds i8, ptr %best, i64 32
  %cur_key_gep = getelementptr inbounds i8, ptr %cur, i64 32
  %cand_key_ptr = select i1 %cur_lt_key, ptr %best_key_gep, ptr %cur_key_gep
  %cand_key = load double, ptr %cand_key_ptr, align 8
  %key_lt_cand = fcmp olt double %key, %cand_key
  br i1 %key_lt_cand, label %insert_node, label %exit

insert_node:
  %hint = phi ptr [ %end, %entry ], [ %end, %lower_bound_exit ], [ %cand, %check_equal ]
  %raw = call noalias ptr @malloc(i64 48)
  store i32 0, ptr %raw, align 8
  %n_parent = getelementptr inbounds i8, ptr %raw, i64 8
  store ptr null, ptr %n_parent, align 8
  %n_left = getelementptr inbounds i8, ptr %raw, i64 16
  store ptr null, ptr %n_left, align 8
  %n_right = getelementptr inbounds i8, ptr %raw, i64 24
  store ptr null, ptr %n_right, align 8
  %n_key = getelementptr inbounds i8, ptr %raw, i64 32
  store double %key, ptr %n_key, align 8
  %n_val = getelementptr inbounds i8, ptr %raw, i64 40
  store double 0.000000e+00, ptr %n_val, align 8
  %pos = call { ptr, ptr } @hint_insert_pos(ptr %map, ptr %hint, double %key)
  %pos0 = extractvalue { ptr, ptr } %pos, 0
  %pos1 = extractvalue { ptr, ptr } %pos, 1
  %has_pos0 = icmp ne ptr %pos0, null
  %pos1_is_end = icmp eq ptr %end, %pos1
  %or_cond = select i1 %has_pos0, i1 true, i1 %pos1_is_end
  br i1 %or_cond, label %link_root, label %link_child

link_root:
  store ptr %raw, ptr %root_slot, align 8
  %leftmost_slot = getelementptr inbounds i8, ptr %map, i64 24
  store ptr %raw, ptr %leftmost_slot, align 8
  %rightmost_slot = getelementptr inbounds i8, ptr %map, i64 32
  store ptr %raw, ptr %rightmost_slot, align 8
  br label %finish_insert

link_child:
  %pos1_key_ptr = getelementptr inbounds i8, ptr %pos1, i64 32
  %pos1_key = load double, ptr %pos1_key_ptr, align 8
  %insert_left = fcmp olt double %key, %pos1_key
  %p_left = getelementptr inbounds i8, ptr %pos1, i64 16
  %p_right = getelementptr inbounds i8, ptr %pos1, i64 24
  %child_slot = select i1 %insert_left, ptr %p_left, ptr %p_right
  store ptr %raw, ptr %child_slot, align 8
  store ptr %pos1, ptr %n_parent, align 8
  br i1 %insert_left, label %finish_insert, label %update_rightmost

update_rightmost:
  %rm_slot = getelementptr inbounds i8, ptr %map, i64 32
  store ptr %raw, ptr %rm_slot, align 8
  br label %finish_insert

finish_insert:
  %count_slot = getelementptr inbounds i8, ptr %map, i64 40
  %old_count = load i64, ptr %count_slot, align 8, !tbaa !0
  %new_count = add nuw i64 %old_count, 1
  store i64 %new_count, ptr %count_slot, align 8, !tbaa !0
  br label %exit

exit:
  %node = phi ptr [ %cand, %check_equal ], [ %raw, %finish_insert ]
  %val_ptr = getelementptr inbounds i8, ptr %node, i64 40
  ret ptr %val_ptr
}

define internal void @erase_subtree(ptr %node) {
entry:
  %is_null = icmp eq ptr %node, null
  br i1 %is_null, label %done, label %body

body:
  %left_slot = getelementptr inbounds i8, ptr %node, i64 16
  %left = load ptr, ptr %left_slot, align 8
  call void @erase_subtree(ptr %left)
  %right_slot = getelementptr inbounds i8, ptr %node, i64 24
  %right = load ptr, ptr %right_slot, align 8
  call void @erase_subtree(ptr %right)
  call void @free(ptr %node)
  br label %done

done:
  ret void
}

define double @compute_map(double %x) {
entry:
  %map = alloca [48 x i8], align 8
  call void @map_init(ptr %map)
  %v1_slot = call ptr @map_operator_index(ptr %map, double 1.000000e+00)
  %x_sq = fmul double %x, %x
  store double %x_sq, ptr %v1_slot, align 8
  %v2_slot = call ptr @map_operator_index(ptr %map, double 2.000000e+00)
  %three_x = fmul double 3.000000e+00, %x
  store double %three_x, ptr %v2_slot, align 8
  %v1_again = call ptr @map_operator_index(ptr %map, double 1.000000e+00)
  %cur_v1 = load double, ptr %v1_again, align 8
  %five_x = fmul double 5.000000e+00, %x
  %upd_v1 = fadd double %cur_v1, %five_x
  store double %upd_v1, ptr %v1_again, align 8
  %read1_slot = call ptr @map_operator_index(ptr %map, double 1.000000e+00)
  %read1 = load double, ptr %read1_slot, align 8
  %read2_slot = call ptr @map_operator_index(ptr %map, double 2.000000e+00)
  %read2 = load double, ptr %read2_slot, align 8
  %two_read2 = fmul double 2.000000e+00, %read2
  %sum = fadd double %read1, %two_read2
  %root_slot = getelementptr inbounds i8, ptr %map, i64 16
  %root = load ptr, ptr %root_slot, align 8
  call void @erase_subtree(ptr %root)
  ret double %sum
}

; CHECK-LABEL: define i32 @main(
; CHECK: call { double } @diffecompute_map(double 4.000000e+00, double 1.000000e+00)
define i32 @main() {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @compute_map, double 4.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %dx)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffecompute_map(double %x, double %differeturn)
; CHECK: call void @augmented_map_init(ptr %map, ptr %"map'ipa")
; CHECK: %[[M2:.+]] = fmul fast double %{{.+}}, 2.000000e+00
; CHECK: fadd fast double %{{.+}}, %[[M2]]
; CHECK: %[[M5:.+]] = fmul fast double %{{.+}}, 5.000000e+00
; CHECK: fadd fast double %{{.+}}, %[[M5]]
; CHECK: %[[M3:.+]] = fmul fast double %{{.+}}, 3.000000e+00
; CHECK: fadd fast double %{{.+}}, %[[M3]]
; CHECK: call void @diffemap_init(ptr %map, ptr %"map'ipa")
; CHECK: ret { double }

!0 = !{!1, !1, i64 0}
!1 = !{!"long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
