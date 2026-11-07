; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli -

; Forward-mode MPI_Recv with MPI_ANY_SOURCE / MPI_ANY_TAG must take the sender
; rank and tag for tangent receive(s) from the completed primal MPI_Status,
; allocating a scratch status when the primal call passed MPI_STATUS_IGNORE.

@q0_vals = global [3 x double] [double 10.0, double 110.0, double 210.0], align 8
@q0_tags = global [3 x i32] [i32 7, i32 7, i32 7], align 4
@q0_head = global i32 0, align 4

@q2_vals = global [3 x double] [double 20.0, double 120.0, double 220.0], align 8
@q2_tags = global [3 x i32] [i32 9, i32 9, i32 9], align 4
@q2_head = global i32 0, align 4

@wildcard_turn = global i32 0, align 4
@ompi_double = global i32 0, align 4
@ompi_world = global i32 0, align 4

define void @reset_queues() {
entry:
  store i32 0, ptr @q0_head, align 4
  store i32 0, ptr @q2_head, align 4
  store i32 0, ptr @wildcard_turn, align 4
  ret void
}

; Pop one double from sender 0 or sender 2. When source < 0 (MPI_ANY_SOURCE),
; alternate between sender 0 and sender 2 so that a second wildcard receive
; before draining sender 0's tangent(s) would pull sender 2's primal instead.
define i32 @pop_message(ptr %buf, i32 %source, i32 %tag, ptr %out_source, ptr %out_tag) {
entry:
  %is_any = icmp slt i32 %source, 0
  br i1 %is_any, label %pick_any, label %picked

pick_any:
  %turn = load i32, ptr @wildcard_turn, align 4
  %next_turn = xor i32 %turn, 1
  store i32 %next_turn, ptr @wildcard_turn, align 4
  %h0_check = load i32, ptr @q0_head, align 4
  %q0_avail = icmp slt i32 %h0_check, 3
  %h2_check = load i32, ptr @q2_head, align 4
  %q2_avail = icmp slt i32 %h2_check, 3
  %want_0 = icmp eq i32 %turn, 0
  %choose_0_first = and i1 %want_0, %q0_avail
  %fallback_0 = and i1 %q0_avail, true
  %no_2 = xor i1 %q2_avail, true
  %force_0 = and i1 %fallback_0, %no_2
  %use_0 = or i1 %choose_0_first, %force_0
  %any_src = select i1 %use_0, i32 0, i32 2
  br label %picked

picked:
  %actual_src = phi i32 [ %source, %entry ], [ %any_src, %pick_any ]
  %is_src0 = icmp eq i32 %actual_src, 0
  %vals_ptr = select i1 %is_src0, ptr @q0_vals, ptr @q2_vals
  %tags_ptr = select i1 %is_src0, ptr @q0_tags, ptr @q2_tags
  %head_ptr = select i1 %is_src0, ptr @q0_head, ptr @q2_head
  %head = load i32, ptr %head_ptr, align 4
  %in_bounds = icmp slt i32 %head, 3
  br i1 %in_bounds, label %check_tag, label %fail

check_tag:
  %idx = sext i32 %head to i64
  %tag_slot = getelementptr inbounds i32, ptr %tags_ptr, i64 %idx
  %msg_tag = load i32, ptr %tag_slot, align 4
  %tag_any = icmp slt i32 %tag, 0
  %tag_eq = icmp eq i32 %tag, %msg_tag
  %tag_ok = or i1 %tag_any, %tag_eq
  br i1 %tag_ok, label %deliver, label %fail

deliver:
  %val_slot = getelementptr inbounds double, ptr %vals_ptr, i64 %idx
  %val = load double, ptr %val_slot, align 8
  store double %val, ptr %buf, align 8
  %next_head = add i32 %head, 1
  store i32 %next_head, ptr %head_ptr, align 4
  store i32 %actual_src, ptr %out_source, align 4
  store i32 %msg_tag, ptr %out_tag, align 4
  ret i32 0

fail:
  store double -999.0, ptr %buf, align 8
  store i32 -99, ptr %out_source, align 4
  store i32 -99, ptr %out_tag, align 4
  ret i32 1
}

; Mock MPICH C ABI MPI_Recv: integer datatype/comm, MPI_STATUS_IGNORE == 1,
; MPI_SOURCE at i32 index 2, MPI_TAG at i32 index 3.
define i32 @MPI_Recv(ptr %buf, i32 %count, i32 %datatype, i32 %source, i32 %tag, i32 %comm, ptr %status) noinline {
entry:
  %src_box = alloca i32, align 4
  %tag_box = alloca i32, align 4
  %err = call i32 @pop_message(ptr %buf, i32 %source, i32 %tag, ptr %src_box, ptr %tag_box)
  %st_int = ptrtoint ptr %status to i64
  %ignore = icmp eq i64 %st_int, 1
  br i1 %ignore, label %done, label %write_status

write_status:
  store i32 8, ptr %status, align 4
  %c1 = getelementptr inbounds i32, ptr %status, i64 1
  store i32 0, ptr %c1, align 4
  %src_val = load i32, ptr %src_box, align 4
  %s_ptr = getelementptr inbounds i32, ptr %status, i64 2
  store i32 %src_val, ptr %s_ptr, align 4
  %tag_val = load i32, ptr %tag_box, align 4
  %t_ptr = getelementptr inbounds i32, ptr %status, i64 3
  store i32 %tag_val, ptr %t_ptr, align 4
  %e_ptr = getelementptr inbounds i32, ptr %status, i64 4
  store i32 %err, ptr %e_ptr, align 4
  br label %done

done:
  ret i32 %err
}

; Mock Open MPI C ABI PMPI_Recv: pointer datatype/comm, MPI_STATUS_IGNORE == 0,
; MPI_SOURCE at i32 index 0, MPI_TAG at i32 index 1.
define i32 @PMPI_Recv(ptr %buf, i32 %count, ptr %datatype, i32 %source, i32 %tag, ptr %comm, ptr %status) noinline {
entry:
  %src_box = alloca i32, align 4
  %tag_box = alloca i32, align 4
  %err = call i32 @pop_message(ptr %buf, i32 %source, i32 %tag, ptr %src_box, ptr %tag_box)
  %ignore = icmp eq ptr %status, null
  br i1 %ignore, label %done, label %write_status

write_status:
  %src_val = load i32, ptr %src_box, align 4
  store i32 %src_val, ptr %status, align 4
  %tag_val = load i32, ptr %tag_box, align 4
  %t_ptr = getelementptr inbounds i32, ptr %status, i64 1
  store i32 %tag_val, ptr %t_ptr, align 4
  %e_ptr = getelementptr inbounds i32, ptr %status, i64 2
  store i32 %err, ptr %e_ptr, align 4
  br label %done

done:
  ret i32 %err
}

; MPICH vector forward-mode receive with MPI_ANY_SOURCE (-2), MPI_ANY_TAG (-1),
; and MPI_STATUS_IGNORE ((ptr)1).
; CHECK-LABEL: define internal void @fwddiffe2recv_mpich_ignore(
; CHECK-SAME: ptr {{.*}}%y, [2 x ptr] %"y'")
; CHECK: %[[MPICH_STATUS:.+]] = alloca [8 x i32], align 8
; CHECK: call i32 @MPI_Recv(ptr %[[Y0:.+]], i32 1, i32 1275070475, i32 -2, i32 -1, i32 1140850688, ptr %[[MPICH_STATUS]])
; CHECK-NEXT: %[[MPICH_SRC_PTR:.+]] = getelementptr inbounds i32, ptr %[[MPICH_STATUS]], i32 2
; CHECK-NEXT: %[[MPICH_TAG_PTR:.+]] = getelementptr inbounds i32, ptr %[[MPICH_STATUS]], i32 3
; CHECK-NEXT: %[[MPICH_SRC:.+]] = load i32, ptr %[[MPICH_SRC_PTR]], align 4
; CHECK-NEXT: %[[MPICH_TAG:.+]] = load i32, ptr %[[MPICH_TAG_PTR]], align 4
; CHECK: call i32 @MPI_Recv(ptr {{.+}}, i32 1, i32 1275070475, i32 %[[MPICH_SRC]], i32 %[[MPICH_TAG]], i32 1140850688, ptr inttoptr (i64 1 to ptr))
; CHECK: call i32 @MPI_Recv(ptr {{.+}}, i32 1, i32 1275070475, i32 %[[MPICH_SRC]], i32 %[[MPICH_TAG]], i32 1140850688, ptr inttoptr (i64 1 to ptr))
define void @recv_mpich_ignore(ptr %y) {
entry:
  %c0 = call i32 @MPI_Recv(ptr %y, i32 1, i32 1275070475, i32 -2, i32 -1, i32 1140850688, ptr inttoptr (i64 1 to ptr))
  %y1 = getelementptr inbounds double, ptr %y, i64 1
  %c1 = call i32 @MPI_Recv(ptr %y1, i32 1, i32 1275070475, i32 -2, i32 -1, i32 1140850688, ptr inttoptr (i64 1 to ptr))
  ret void
}

; Open MPI scalar forward-mode receive with runtime source/tag/status.
; CHECK-LABEL: define internal void @fwddifferecv_ompi_runtime_status(
; CHECK-SAME: ptr {{.*}}%y, ptr {{.*}}%"y'", i32 %src, i32 %tag, ptr %st)
; CHECK-DAG: %[[OMPI_PRI_STATUS:.+]] = alloca [8 x i32], align 8
; CHECK-DAG: %[[OMPI_TAN_STATUS:.+]] = alloca [8 x i32], align 8
; CHECK: %[[ST_INT:.+]] = ptrtoint ptr %st to i64
; CHECK-NEXT: %[[IS_IGN:.+]] = icmp ule i64 %[[ST_INT]], 1
; CHECK-NEXT: %[[EFF_ST:.+]] = select i1 %[[IS_IGN]], ptr %[[OMPI_PRI_STATUS]], ptr %st
; CHECK-NEXT: call i32 @PMPI_Recv(ptr %y, i32 1, ptr @ompi_double, i32 %src, i32 %tag, ptr @ompi_world, ptr %[[EFF_ST]])
; CHECK-NEXT: %[[OMPI_TAG_PTR:.+]] = getelementptr inbounds i32, ptr %[[EFF_ST]], i32 1
; CHECK-NEXT: %[[OMPI_SRC:.+]] = load i32, ptr %[[EFF_ST]], align 4
; CHECK-NEXT: %[[OMPI_TAG:.+]] = load i32, ptr %[[OMPI_TAG_PTR]], align 4
; CHECK: call i32 @PMPI_Recv(ptr %"y'", i32 1, ptr @ompi_double, i32 %[[OMPI_SRC]], i32 %[[OMPI_TAG]], ptr @ompi_world, ptr %[[OMPI_TAN_STATUS]])
define void @recv_ompi_runtime_status(ptr %y, i32 %src, i32 %tag, ptr %st) {
entry:
  %c0 = call i32 @PMPI_Recv(ptr %y, i32 1, ptr @ompi_double, i32 %src, i32 %tag, ptr @ompi_world, ptr %st)
  ret void
}

declare void @__enzyme_fwddiff(...)

define i32 @main() {
entry:
  %y = alloca [2 x double], align 8
  %dy1 = alloca [2 x double], align 8
  %dy2 = alloca [2 x double], align 8
  %user_st = alloca [8 x i32], align 8

  ; Test 1: MPICH vector forward mode (width 2) with MPI_ANY_SOURCE / MPI_ANY_TAG
  ; and MPI_STATUS_IGNORE.
  call void @reset_queues()
  call void (...) @__enzyme_fwddiff(ptr @recv_mpich_ignore, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %y, ptr %dy1, ptr %dy2)

  %y_0 = load double, ptr %y, align 8
  %dy1_0 = load double, ptr %dy1, align 8
  %dy2_0 = load double, ptr %dy2, align 8
  %y_1_ptr = getelementptr inbounds double, ptr %y, i64 1
  %dy1_1_ptr = getelementptr inbounds double, ptr %dy1, i64 1
  %dy2_1_ptr = getelementptr inbounds double, ptr %dy2, i64 1
  %y_1 = load double, ptr %y_1_ptr, align 8
  %dy1_1 = load double, ptr %dy1_1_ptr, align 8
  %dy2_1 = load double, ptr %dy2_1_ptr, align 8

  %exp_dy1_0 = fadd double %y_0, 100.0
  %exp_dy2_0 = fadd double %y_0, 200.0
  %exp_dy1_1 = fadd double %y_1, 100.0
  %exp_dy2_1 = fadd double %y_1, 200.0

  %ok0 = fcmp oeq double %dy1_0, %exp_dy1_0
  %ok1 = fcmp oeq double %dy2_0, %exp_dy2_0
  %ok2 = fcmp oeq double %dy1_1, %exp_dy1_1
  %ok3 = fcmp oeq double %dy2_1, %exp_dy2_1
  %ok01 = and i1 %ok0, %ok1
  %ok23 = and i1 %ok2, %ok3
  %test1_ok = and i1 %ok01, %ok23
  br i1 %test1_ok, label %test2, label %bad

test2:
  ; Test 2: Open MPI runtime status parameter, first with caller-supplied status,
  ; then with MPI_STATUS_IGNORE (null). Each sender here sends 1 primal + 1 tangent.
  store i32 0, ptr @q0_head, align 4
  store i32 0, ptr @q2_head, align 4
  store i32 0, ptr @wildcard_turn, align 4
  store i32 -77, ptr %user_st, align 4

  call void (...) @__enzyme_fwddiff(ptr @recv_ompi_runtime_status, metadata !"enzyme_dup", ptr %y, ptr %dy1, metadata !"enzyme_const", i32 -1, metadata !"enzyme_const", i32 -1, metadata !"enzyme_const", ptr %user_st)
  call void (...) @__enzyme_fwddiff(ptr @recv_ompi_runtime_status, metadata !"enzyme_dup", ptr %y_1_ptr, ptr %dy1_1_ptr, metadata !"enzyme_const", i32 -1, metadata !"enzyme_const", i32 -1, metadata !"enzyme_const", ptr null)

  %oy_0 = load double, ptr %y, align 8
  %ody_0 = load double, ptr %dy1, align 8
  %oy_1 = load double, ptr %y_1_ptr, align 8
  %ody_1 = load double, ptr %dy1_1_ptr, align 8
  %st_src = load i32, ptr %user_st, align 4
  %st_tag_ptr = getelementptr inbounds i32, ptr %user_st, i64 1
  %st_tag = load i32, ptr %st_tag_ptr, align 4

  %oexp_0 = fadd double %oy_0, 100.0
  %oexp_1 = fadd double %oy_1, 100.0
  %ook0 = fcmp oeq double %ody_0, %oexp_0
  %ook1 = fcmp oeq double %ody_1, %oexp_1
  %ook_src = icmp eq i32 %st_src, 0
  %ook_tag = icmp eq i32 %st_tag, 7
  %ook01 = and i1 %ook0, %ook1
  %ook_st = and i1 %ook_src, %ook_tag
  %test2_ok = and i1 %ook01, %ook_st
  br i1 %test2_ok, label %good, label %bad

good:
  ret i32 0

bad:
  ret i32 1
}
