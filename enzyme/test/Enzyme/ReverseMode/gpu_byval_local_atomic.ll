; RUN: if [ %llvmver -eq 15 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s; fi

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.OperationScalarPopulator = type { ptr, i64, double }
%struct.ScalarBox = type { double }

@enzyme_const = external global i32
@enzyme_dup = external global i32

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @__enzyme_autodiff(ptr, ...)

define void @device_populator(i32 %size, ptr %op_void) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp = icmp ult i32 %tid, %size
  br i1 %cmp, label %body, label %exit

body:
  %idx = zext i32 %tid to i64
  %arr = load ptr, ptr %op_void, align 8
  %scale = load double, ptr %arr, align 8
  %val_ptr = getelementptr inbounds i8, ptr %op_void, i64 16
  %val = load double, ptr %val_ptr, align 8
  %prod = fmul double %val, %scale
  %slot = getelementptr inbounds double, ptr %arr, i64 %idx
  store double %prod, ptr %slot, align 8
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define ptx_kernel void @kernel_byval(
; CHECK: call void @diffedevice_populator(i32 %size, ptr %operation, ptr %d_operation)
define ptx_kernel void @kernel_byval(i32 %size, ptr byval(%struct.OperationScalarPopulator) align 8 %operation, ptr byval(%struct.OperationScalarPopulator) align 8 %d_operation) {
entry:
  %ec = load i32, ptr @enzyme_const, align 4
  %ed = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @device_populator, i32 %ec, i32 %size, i32 %ed, ptr %operation, ptr %d_operation)
  ret void
}

; CHECK-LABEL: define ptx_kernel void @kernel_global(
; CHECK: call void @diffedevice_populator{{[0-9]+}}(i32 %size, ptr %global_op, ptr %d_global_op)
define ptx_kernel void @kernel_global(i32 %size, ptr %global_op, ptr %d_global_op) {
entry:
  %ec = load i32, ptr @enzyme_const, align 4
  %ed = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @device_populator, i32 %ec, i32 %size, i32 %ed, ptr %global_op, ptr %d_global_op)
  ret void
}

define void @inner_load(ptr %x_ptr, ptr %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idx = zext i32 %tid to i64
  %v = load double, ptr %x_ptr, align 8
  %dst = getelementptr inbounds double, ptr %out, i64 %idx
  store double %v, ptr %dst, align 8
  ret void
}

define void @outer_forward(ptr %x_ptr, ptr %out) {
entry:
  call void @inner_load(ptr %x_ptr, ptr %out)
  ret void
}

; CHECK-LABEL: define ptx_kernel void @kernel_nested_alloca(
; CHECK: call void @diffeouter_forward(ptr %x, ptr %dx, ptr %out, ptr %d_out)
define ptx_kernel void @kernel_nested_alloca(double %x_init, ptr %out, ptr %d_out, ptr %dx_sink) {
entry:
  %x = alloca double, align 8
  %dx = alloca double, align 8
  store double %x_init, ptr %x, align 8
  store double 0.000000e+00, ptr %dx, align 8
  %ed = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @outer_forward, i32 %ed, ptr %x, ptr %dx, i32 %ed, ptr %out, ptr %d_out)
  %res = load double, ptr %dx, align 8
  store double %res, ptr %dx_sink, align 8
  ret void
}

define void @byval_callee(ptr byval(%struct.ScalarBox) align 8 %box, ptr %out) {
entry:
  %v = load double, ptr %box, align 8
  store double %v, ptr %out, align 8
  ret void
}

; CHECK-LABEL: define ptx_kernel void @kernel_direct_byval(
; CHECK: call void @diffebyval_callee(ptr %box, ptr %d_box, ptr %out, ptr %d_out)
define ptx_kernel void @kernel_direct_byval(ptr byval(%struct.ScalarBox) align 8 %box, ptr byval(%struct.ScalarBox) align 8 %d_box, ptr %out, ptr %d_out) {
entry:
  %ed = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @byval_callee, i32 %ed, ptr %box, ptr %d_box, i32 %ed, ptr %out, ptr %d_out)
  ret void
}

; CHECK-LABEL: define ptx_kernel void @kernel_direct_byval_global_shadow(
; CHECK: call void @diffebyval_callee{{[0-9]+}}(ptr %box, ptr %global_d_box, ptr %out, ptr %d_out)
define ptx_kernel void @kernel_direct_byval_global_shadow(ptr byval(%struct.ScalarBox) align 8 %box, ptr %global_d_box, ptr %out, ptr %d_out) {
entry:
  %ed = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_autodiff(ptr @byval_callee, i32 %ed, ptr %box, ptr %global_d_box, i32 %ed, ptr %out, ptr %d_out)
  ret void
}

; CHECK-LABEL: define internal void @diffedevice_populator(
; CHECK: invertbody:
; CHECK:   %[[ARR:.+]] = load ptr, ptr %"op_void'", align 8
; CHECK:   %[[VAL_PTR:.+]] = getelementptr inbounds i8, ptr %"op_void'", i64 16
; CHECK-NEXT:   %[[OLD_VAL:.+]] = load double, ptr %[[VAL_PTR]], align 8
; CHECK-NEXT:   %[[NEW_VAL:.+]] = fadd fast double %[[OLD_VAL]], %{{.+}}
; CHECK-NEXT:   store double %[[NEW_VAL]], ptr %[[VAL_PTR]], align 8
; CHECK-NEXT:   atomicrmw fadd ptr %[[ARR]], double %{{.+}} monotonic, align 8
; CHECK-NEXT:   br label %invertentry

; CHECK-LABEL: define internal void @diffedevice_populator{{[0-9]+}}(
; CHECK: invertbody:
; CHECK:   %[[GARR:.+]] = load ptr, ptr %"op_void'", align 8
; CHECK:   %[[GVAL_PTR:.+]] = getelementptr inbounds i8, ptr %"op_void'", i64 16
; CHECK-NEXT:   atomicrmw fadd ptr %[[GVAL_PTR]], double %{{.+}} monotonic, align 8
; CHECK-NEXT:   atomicrmw fadd ptr %[[GARR]], double %{{.+}} monotonic, align 8
; CHECK-NEXT:   br label %invertentry

; CHECK-LABEL: define internal void @diffeinner_load(
; CHECK-NOT: atomicrmw
; CHECK:   %[[OLD_X:.+]] = load double, ptr %"x_ptr'", align 8
; CHECK-NEXT:   %[[NEW_X:.+]] = fadd fast double %[[OLD_X]], %{{.+}}
; CHECK-NEXT:   store double %[[NEW_X]], ptr %"x_ptr'", align 8
; CHECK-NEXT:   ret void

; CHECK-LABEL: define internal void @diffebyval_callee(
; CHECK-NOT: atomicrmw
; CHECK:   %[[OLD_BOX:.+]] = load double, ptr %"box'", align 8
; CHECK-NEXT:   %[[NEW_BOX:.+]] = fadd fast double %[[OLD_BOX]], %{{.+}}
; CHECK-NEXT:   store double %[[NEW_BOX]], ptr %"box'", align 8
; CHECK-NEXT:   ret void

; CHECK-LABEL: define internal void @diffebyval_callee{{[0-9]+}}(
; CHECK:   atomicrmw fadd ptr %"box'", double %{{.+}} monotonic, align 8
; CHECK-NEXT:   ret void
