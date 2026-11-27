; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s

declare void @__enzyme_fwddiff(...)

@enzyme_const = external global i32
@enzyme_dup = external global i32

%struct.Product = type { ptr, ptr }
%struct.Kernel = type { ptr, ptr }

define internal void @scale_and_add(ptr %y, ptr %A, ptr %x) {
entry:
  %a0 = load double, ptr %A, align 8
  %x0 = load double, ptr %x, align 8
  %mul = fmul double %a0, %x0
  store double %mul, ptr %y, align 8
  ret void
}

define internal void @set_noalias(ptr %y, ptr %prod) {
entry:
  %A = load ptr, ptr %prod, align 8
  %x_slot = getelementptr inbounds i8, ptr %prod, i64 8
  %x = load ptr, ptr %x_slot, align 8
  call void @scale_and_add(ptr %y, ptr %A, ptr %x)
  ret void
}

define internal void @call_assignment(ptr %y, ptr %prod) {
entry:
  call void @set_noalias(ptr %y, ptr %prod)
  ret void
}

define void @matmul_wrapped(ptr %x, ptr %y, ptr %A) {
entry:
  %prod = alloca %struct.Product, align 8
  store ptr %A, ptr %prod, align 8
  %x_slot = getelementptr inbounds i8, ptr %prod, i64 8
  store ptr %x, ptr %x_slot, align 8
  call void @call_assignment(ptr %y, ptr %prod)
  ret void
}

define internal void @run_kernel(ptr %k) {
entry:
  %y = load ptr, ptr %k, align 8
  %eval_slot = getelementptr inbounds i8, ptr %k, i64 8
  %eval = load ptr, ptr %eval_slot, align 8
  %A = load ptr, ptr %eval, align 8
  %x_slot = getelementptr inbounds i8, ptr %eval, i64 8
  %x = load ptr, ptr %x_slot, align 8
  %a0 = load double, ptr %A, align 8
  %x0 = load double, ptr %x, align 8
  %mul = fmul double %a0, %x0
  store double %mul, ptr %y, align 8
  ret void
}

define void @matmul_nested_kernel(ptr %x, ptr %y, ptr %A) {
entry:
  %eval = alloca %struct.Product, align 8
  %kernel = alloca %struct.Kernel, align 8
  store ptr %A, ptr %eval, align 8
  %x_slot = getelementptr inbounds i8, ptr %eval, i64 8
  store ptr %x, ptr %x_slot, align 8
  store ptr %y, ptr %kernel, align 8
  %eval_slot = getelementptr inbounds i8, ptr %kernel, i64 8
  store ptr %eval, ptr %eval_slot, align 8
  call void @run_kernel(ptr %kernel)
  ret void
}

define void @caller(ptr %x, ptr %dx, ptr %y, ptr %dy, ptr %A) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @matmul_wrapped, ptr @enzyme_dup, ptr %x, ptr %dx, ptr @enzyme_dup, ptr %y, ptr %dy, ptr @enzyme_const, ptr %A)
  call void (...) @__enzyme_fwddiff(ptr @matmul_nested_kernel, ptr @enzyme_dup, ptr %x, ptr %dx, ptr @enzyme_dup, ptr %y, ptr %dy, ptr @enzyme_const, ptr %A)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffematmul_wrapped(
; CHECK-SAME: ptr %x, ptr %"x'", ptr {{.*}}%y, ptr {{.*}}%"y'", ptr %A)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @fwddiffescale_and_add(ptr %y, ptr %"y'", ptr %A, ptr %x, ptr %"x'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define internal void @fwddiffescale_and_add(
; CHECK-SAME: ptr {{.*}}%y, ptr {{.*}}%"y'", ptr {{.*}}%A, ptr {{.*}}%x, ptr {{.*}}%"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a0 = load double, ptr %A, align 8
; CHECK-NEXT:   %"x0'ipl" = load double, ptr %"x'", align 8
; CHECK-NEXT:   %x0 = load double, ptr %x, align 8
; CHECK-NEXT:   %mul = fmul double %a0, %x0
; CHECK-NEXT:   %0 = fmul fast double %"x0'ipl", %a0
; CHECK-NEXT:   store double %0, ptr %"y'", align 8
; CHECK-NEXT:   store double %mul, ptr %y, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define internal void @fwddiffematmul_nested_kernel(
; CHECK-SAME: ptr %x, ptr %"x'", ptr %y, ptr %"y'", ptr %A)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a0.i = load double, ptr %A, align 8
; CHECK-NEXT:   %"x0.i'ipl" = load double, ptr %"x'", align 8
; CHECK-NEXT:   %x0.i = load double, ptr %x, align 8
; CHECK-NEXT:   %mul.i = fmul double %a0.i, %x0.i
; CHECK-NEXT:   %0 = fmul fast double %"x0.i'ipl", %a0.i
; CHECK-NEXT:   store double %0, ptr %"y'", align 8
; CHECK-NEXT:   store double %mul.i, ptr %y, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
