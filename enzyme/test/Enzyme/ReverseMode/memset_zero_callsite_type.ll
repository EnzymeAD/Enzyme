; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O3>" -S | %lli -

@enzyme_dup = external global i32, align 4
@enzyme_width = external global i32, align 4

@grad_init = private unnamed_addr constant [27 x float] [
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00,
  float 1.000000e+00, float 1.000000e+00, float 1.000000e+00
], align 16

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @__enzyme_autodiff(...)
declare void @__enzyme_fwddiff(...)
declare void @abort()

define void @irreps_to_out(ptr nocapture readnone %x, ptr nocapture readnone %y, ptr nocapture readnone %z, ptr nocapture writeonly %out) {
entry:
  call void @llvm.memset.p0.i64(ptr align 4 %out, i8 0, i64 108, i1 false)
  ret void
}

define void @zero_doubles(ptr nocapture writeonly %out) {
entry:
  call void @llvm.memset.p0.i64(ptr align 8 %out, i8 0, i64 32, i1 false)
  ret void
}

define void @zero_vec4(ptr nocapture writeonly %out) {
entry:
  call void @llvm.memset.p0.i64(ptr align 4 %out, i8 0, i64 16, i1 false)
  ret void
}

define void @zero_ints(ptr nocapture writeonly %out) {
entry:
  call void @llvm.memset.p0.i64(ptr align 4 %out, i8 0, i64 16, i1 false)
  ret void
}

define void @run_alloca_only_memset() {
entry:
  %dbl_out = alloca [4 x double], align 16
  %dbl_dout = alloca [4 x double], align 16
  call void @llvm.memset.p0.i64(ptr align 16 %dbl_out, i8 0, i64 32, i1 false)
  call void @llvm.memset.p0.i64(ptr align 16 %dbl_dout, i8 0, i64 32, i1 false)
  %dup = load i32, ptr @enzyme_dup, align 4
  call void (...) @__enzyme_fwddiff(ptr @zero_doubles, i32 %dup, ptr %dbl_out, ptr %dbl_dout)
  ret void
}

define i32 @main() {
entry:
  %out = alloca [27 x float], align 16
  %d_out = alloca [27 x float], align 16
  %x = alloca float, align 4
  %y = alloca float, align 4
  %z = alloca float, align 4
  %dx = alloca float, align 4
  %dy = alloca float, align 4
  %dz = alloca float, align 4

  call void @llvm.memset.p0.i64(ptr align 16 %out, i8 0, i64 108, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %d_out, ptr align 16 @grad_init, i64 108, i1 false)
  store float 1.000000e+00, ptr %x, align 4
  store float 2.000000e+00, ptr %y, align 4
  store float 3.000000e+00, ptr %z, align 4
  store float 0.000000e+00, ptr %dx, align 4
  store float 0.000000e+00, ptr %dy, align 4
  store float 0.000000e+00, ptr %dz, align 4

  %dup = load i32, ptr @enzyme_dup, align 4
  call void (...) @__enzyme_autodiff(ptr @irreps_to_out, i32 %dup, ptr %x, ptr %dx, i32 %dup, ptr %y, ptr %dy, i32 %dup, ptr %z, ptr %dz, i32 %dup, ptr %out, ptr %d_out)

  %d0 = load float, ptr %d_out, align 16
  %cmp0 = fcmp une float %d0, 0.000000e+00
  br i1 %cmp0, label %fail, label %check_last

check_last:
  %d26_ptr = getelementptr inbounds [27 x float], ptr %d_out, i64 0, i64 26
  %d26 = load float, ptr %d26_ptr, align 8
  %cmp26 = fcmp une float %d26, 0.000000e+00
  br i1 %cmp26, label %fail, label %test_alloca_fallback

test_alloca_fallback:
  call void @run_alloca_only_memset()
  br label %test_width2

test_width2:
  %v_out = alloca [4 x float], align 16
  %v_d1 = alloca [4 x float], align 16
  %v_d2 = alloca [4 x float], align 16
  call void @llvm.memset.p0.i64(ptr align 16 %v_out, i8 0, i64 16, i1 false)
  store float 5.000000e+00, ptr %v_d1, align 16
  %v_d2_3 = getelementptr inbounds [4 x float], ptr %v_d2, i64 0, i64 3
  store float 7.000000e+00, ptr %v_d2_3, align 4
  %width_tag = load i32, ptr @enzyme_width, align 4
  call void (...) @__enzyme_autodiff(ptr @zero_vec4, i32 %width_tag, i64 2, i32 %dup, ptr %v_out, ptr %v_d1, ptr %v_d2)
  %vd1_0 = load float, ptr %v_d1, align 16
  %vd2_3 = load float, ptr %v_d2_3, align 4
  %v_bad1 = fcmp une float %vd1_0, 0.000000e+00
  %v_bad2 = fcmp une float %vd2_3, 0.000000e+00
  %v_bad = or i1 %v_bad1, %v_bad2
  br i1 %v_bad, label %fail, label %test_int

test_int:
  %i_out = alloca [4 x i32], align 16
  %i_dout = alloca [4 x i32], align 16
  store i32 9, ptr %i_out, align 16
  store i32 42, ptr %i_dout, align 16
  call void (...) @__enzyme_autodiff(ptr @zero_ints, i32 %dup, ptr %i_out, ptr %i_dout)
  %id0 = load i32, ptr %i_dout, align 16
  %i_bad = icmp ne i32 %id0, 0
  br i1 %i_bad, label %fail, label %done

fail:
  call void @abort()
  unreachable

done:
  ret i32 0
}

; CHECK-LABEL: define internal void @fwddiffezero_doubles(
; CHECK-SAME: ptr nocapture writeonly %out, ptr nocapture %[[FDOUT:.+]])
; CHECK: call void @llvm.memset.p0.i64(ptr align 8 %out, i8 0, i64 32, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 8 %[[FDOUT]], i8 0, i64 32, i1 false)
; CHECK-NEXT: ret void

; CHECK-LABEL: define internal void @diffeirreps_to_out(
; CHECK-SAME: ptr nocapture readnone %x, ptr nocapture readnone %[[DX:.+]], ptr nocapture readnone %y, ptr nocapture readnone %[[DY:.+]], ptr nocapture readnone %z, ptr nocapture readnone %[[DZ:.+]], ptr nocapture writeonly %out, ptr nocapture %[[DOUT:.+]])
; CHECK: call void @llvm.memset.p0.i64(ptr align 4 %out, i8 0, i64 108, i1 false)
; CHECK: invertentry:
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 4 %[[DOUT]], i8 0, i64 108, i1 false)
; CHECK-NEXT: ret void

; CHECK-LABEL: define internal void @diffe2zero_vec4(
; CHECK-SAME: ptr nocapture writeonly %out, [2 x ptr] %[[VDOUT:.+]])
; CHECK: invertentry:
; CHECK-NEXT: %[[L0:.+]] = extractvalue [2 x ptr] %[[VDOUT]], 0
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 4 %[[L0]], i8 0, i64 16, i1 false)
; CHECK-NEXT: %[[L1:.+]] = extractvalue [2 x ptr] %[[VDOUT]], 1
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 4 %[[L1]], i8 0, i64 16, i1 false)
; CHECK-NEXT: ret void

; CHECK-LABEL: define internal void @diffezero_ints(
; CHECK-SAME: ptr nocapture writeonly %out, ptr nocapture %[[IDOUT:.+]])
; CHECK: call void @llvm.memset.p0.i64(ptr align 4 %out, i8 0, i64 16, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 4 %[[IDOUT]], i8 0, i64 16, i1 false)
; CHECK: invertentry:
; CHECK-NEXT: ret void
