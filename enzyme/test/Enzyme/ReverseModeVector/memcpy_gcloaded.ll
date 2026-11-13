; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli -

; Batched reverse-mode differentiation of a memcpy whose source is an
; uncacheable julia.gc_loaded call whose shadow is still a placeholder PHINode
; when the memcpy is visited. The shadows must be looked up in the reverse pass
; before the chain rule splits them into lanes: looking up lane 0 materializes
; the real shadow and erases the placeholder, so a per-lane lookup would extract
; lane 1 from the erased placeholder.

define internal double @energy(ptr nocapture readonly align 8 dereferenceable(16) "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}" %pair) noinline nounwind readonly {
entry:
  %w_ptr = getelementptr inbounds i8, ptr %pair, i64 8
  %w = load double, ptr %w_ptr, align 8
  %w_sq = fmul double %w, %w
  %x = load double, ptr %pair, align 8
  %prod = fmul double %x, %w_sq
  ret double %prod
}

define ptr @julia.gc_loaded(ptr nocapture readnone %mem, ptr readnone %data) nofree norecurse nosync nounwind willreturn readnone "enzyme_nocache" "enzyme_shouldrecompute" {
entry:
  ret ptr %data
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @__enzyme_autodiff(ptr, ...)

; CHECK-LABEL: define internal void @diffe2accumulate_energy(
; CHECK: invertpeel:
; CHECK-DAG: %[[GC0:.+]] = call ptr @julia.gc_loaded(ptr %"mem'il_phi_unwrap", ptr %"data'il_phi_unwrap")
; CHECK-DAG: %[[GC1:.+]] = call ptr @julia.gc_loaded(ptr %"mem'il_phi_unwrap{{[0-9]+}}", ptr %"data'il_phi_unwrap{{[0-9]+}}")
; CHECK: getelementptr inbounds{{.*}} double, ptr %[[GC0]], i64
; CHECK: getelementptr inbounds{{.*}} double, ptr %[[GC1]], i64
define void @accumulate_energy(ptr nocapture readonly align 8 dereferenceable(24) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@double, [-1,8]:Pointer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer}" %vec, ptr nocapture writeonly align 8 "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %out) {
entry:
  %slot = alloca [2 x double], align 8
  %len_ptr = getelementptr inbounds i8, ptr %vec, i64 16
  %len = load i64, ptr %len_ptr, align 8
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %exit, label %peel

peel:
  %mem_ptr = getelementptr inbounds i8, ptr %vec, i64 8
  %mem = load ptr, ptr %mem_ptr, align 8
  %data = load ptr, ptr %vec, align 8
  %first_ptr = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" ptr @julia.gc_loaded(ptr %mem, ptr %data)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %slot, ptr align 8 %first_ptr, i64 16, i1 false)
  %first_val = call double @energy(ptr %slot)
  %more = icmp ugt i64 %len, 1
  br i1 %more, label %loop, label %exit

loop:
  %idx = phi i64 [ 1, %peel ], [ %idx_next, %loop ]
  %acc = phi double [ %first_val, %peel ], [ %acc_next, %loop ]
  %loop_data = load ptr, ptr %vec, align 8
  %loop_mem = load ptr, ptr %mem_ptr, align 8
  %loop_base = call "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" ptr @julia.gc_loaded(ptr %loop_mem, ptr %loop_data)
  %byte_off = shl i64 %idx, 4
  %elem_ptr = getelementptr inbounds i8, ptr %loop_base, i64 %byte_off
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %slot, ptr align 8 %elem_ptr, i64 16, i1 false)
  %elem_val = call double @energy(ptr %slot)
  %acc_next = fadd double %acc, %elem_val
  %idx_next = add i64 %idx, 1
  %cond = icmp ult i64 %idx_next, %len
  br i1 %cond, label %loop, label %exit

exit:
  %res = phi double [ 0.0, %entry ], [ %first_val, %peel ], [ %acc_next, %loop ]
  store double %res, ptr %out, align 8
  ret void
}

define i32 @main() {
entry:
  %buf = alloca [8 x double], align 8
  %dbuf1 = alloca [8 x double], align 8
  %dbuf2 = alloca [8 x double], align 8
  %dummy_mem = alloca i64, align 8

  %vec = alloca { ptr, ptr, i64 }, align 8
  %dvec1 = alloca { ptr, ptr, i64 }, align 8
  %dvec2 = alloca { ptr, ptr, i64 }, align 8

  %out = alloca double, align 8
  %dout1 = alloca double, align 8
  %dout2 = alloca double, align 8

  store [8 x double] [double 1.0, double 0.5, double 2.0, double 1.0, double 3.0, double 1.5, double 4.0, double 2.0], ptr %buf, align 8
  store [8 x double] zeroinitializer, ptr %dbuf1, align 8
  store [8 x double] zeroinitializer, ptr %dbuf2, align 8
  store i64 0, ptr %dummy_mem, align 8

  store ptr %buf, ptr %vec, align 8
  %v_mem = getelementptr inbounds i8, ptr %vec, i64 8
  store ptr %dummy_mem, ptr %v_mem, align 8
  %v_len = getelementptr inbounds i8, ptr %vec, i64 16
  store i64 4, ptr %v_len, align 8

  store ptr %dbuf1, ptr %dvec1, align 8
  %dv1_mem = getelementptr inbounds i8, ptr %dvec1, i64 8
  store ptr %dummy_mem, ptr %dv1_mem, align 8
  %dv1_len = getelementptr inbounds i8, ptr %dvec1, i64 16
  store i64 4, ptr %dv1_len, align 8

  store ptr %dbuf2, ptr %dvec2, align 8
  %dv2_mem = getelementptr inbounds i8, ptr %dvec2, i64 8
  store ptr %dummy_mem, ptr %dv2_mem, align 8
  %dv2_len = getelementptr inbounds i8, ptr %dvec2, i64 16
  store i64 4, ptr %dv2_len, align 8

  store double 0.0, ptr %out, align 8
  store double 1.0, ptr %dout1, align 8
  store double 3.0, ptr %dout2, align 8

  call void (ptr, ...) @__enzyme_autodiff(ptr @accumulate_energy, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %vec, ptr %dvec1, ptr %dvec2, metadata !"enzyme_dup", ptr %out, ptr %dout1, ptr %dout2)

  ; Lane 0 (seed 1.0): d/dx_0 = 0.25, d/dw_0 = 1.0, d/dx_3 = 4.0, d/dw_3 = 16.0
  ; Lane 1 (seed 3.0): d/dx_0 = 0.75, d/dw_0 = 3.0, d/dx_3 = 12.0, d/dw_3 = 48.0
  %d1_0 = load double, ptr %dbuf1, align 8
  %p1_1 = getelementptr inbounds double, ptr %dbuf1, i64 1
  %d1_1 = load double, ptr %p1_1, align 8
  %p1_6 = getelementptr inbounds double, ptr %dbuf1, i64 6
  %d1_6 = load double, ptr %p1_6, align 8
  %p1_7 = getelementptr inbounds double, ptr %dbuf1, i64 7
  %d1_7 = load double, ptr %p1_7, align 8

  %d2_0 = load double, ptr %dbuf2, align 8
  %p2_1 = getelementptr inbounds double, ptr %dbuf2, i64 1
  %d2_1 = load double, ptr %p2_1, align 8
  %p2_6 = getelementptr inbounds double, ptr %dbuf2, i64 6
  %d2_6 = load double, ptr %p2_6, align 8
  %p2_7 = getelementptr inbounds double, ptr %dbuf2, i64 7
  %d2_7 = load double, ptr %p2_7, align 8

  %ok0 = fcmp oeq double %d1_0, 0.25
  %ok1 = fcmp oeq double %d1_1, 1.0
  %ok2 = fcmp oeq double %d1_6, 4.0
  %ok3 = fcmp oeq double %d1_7, 16.0
  %ok4 = fcmp oeq double %d2_0, 0.75
  %ok5 = fcmp oeq double %d2_1, 3.0
  %ok6 = fcmp oeq double %d2_6, 12.0
  %ok7 = fcmp oeq double %d2_7, 48.0

  %ok01 = and i1 %ok0, %ok1
  %ok23 = and i1 %ok2, %ok3
  %ok45 = and i1 %ok4, %ok5
  %ok67 = and i1 %ok6, %ok7
  %ok03 = and i1 %ok01, %ok23
  %ok47 = and i1 %ok45, %ok67
  %all_ok = and i1 %ok03, %ok47
  %ret = select i1 %all_ok, i32 0, i32 1
  ret i32 %ret
}
