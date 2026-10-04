; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; The forward derivative of a cuBLAS v2 dot writes its result through a pointer in host
; memory, which is only correct with the handle in CUBLAS_POINTER_MODE_HOST: it checks the
; pointer mode of the handle first.

declare i32 @cublasDdot_v2(ptr, i32, ptr, i32, ptr, i32, ptr)

define void @f(ptr %handle, i32 %n, ptr %x, ptr %y, ptr %res) {
entry:
  %r = call i32 @cublasDdot_v2(ptr %handle, i32 %n, ptr %x, i32 1, ptr %y, i32 1, ptr %res)
  ret void
}

declare void @__enzyme_fwddiff(...)

define void @active(ptr %handle, i32 %n, ptr %x, ptr %dx, ptr %y, ptr %dy, ptr %res, ptr %dres) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_const", ptr %handle, metadata !"enzyme_const", i32 %n, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_dup", ptr %res, ptr %dres)
  ret void
}

; CHECK: @[[msg:.+]] = private unnamed_addr constant {{.*}} c"Enzyme: the derivative of cublasDdot_v2 is only supported for a cuBLAS handle in CUBLAS_POINTER_MODE_HOST, but the handle is in another pointer mode\0A
; CHECK: define internal void @fwddiffef(ptr %handle,
; CHECK: call void @__enzyme_cublas_pointer_mode_check[[n:[0-9]+]](ptr %handle)
; CHECK: call {{.*}}@cublasDdot_v2(ptr %handle,
; CHECK: define internal void @__enzyme_cublas_pointer_mode_check[[n]](ptr %handle)
; CHECK: call i32 @cublasGetPointerMode_v2(ptr %handle, ptr %mode)
; CHECK: bad:
; CHECK-NEXT: call i32 @puts(ptr @[[msg]])
