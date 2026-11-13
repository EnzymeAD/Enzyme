; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s --check-prefix=NOOPT; fi

declare double @modf(double, ptr)
declare double @__enzyme_autodiff(...)
declare void @__enzyme_autodiff_v(...)

; Overwriting an active alloca with modf must zero its shadow (derivative = 0).
define double @f_overwrite(double %x) {
entry:
  %p = alloca double, align 8
  store double %x, ptr %p, align 8
  %frac = call double @modf(double 3.5, ptr %p)
  %res = load double, ptr %p, align 8
  ret double %res
}

define double @df_overwrite(double %x) {
entry:
  %d = call double (...) @__enzyme_autodiff(ptr @f_overwrite, double %x)
  ret double %d
}

; Active arg0 + active overwritten arg1 + used frac: derivative is 2.0.
define double @f_overwrite_sum(double %x) {
entry:
  %p = alloca double, align 8
  store double %x, ptr %p, align 8
  %x2 = fmul double %x, 2.0
  %frac = call double @modf(double %x2, ptr %p)
  %res = load double, ptr %p, align 8
  %sum = fadd double %frac, %res
  ret double %sum
}

define double @df_overwrite_sum(double %x) {
entry:
  %d = call double (...) @__enzyme_autodiff(ptr @f_overwrite_sum, double %x)
  ret double %d
}

; Active arg0 + active overwritten arg1 with unused fractional return.
define double @f_unused_frac(double %x) {
entry:
  %p = alloca double, align 8
  store double %x, ptr %p, align 8
  %x2 = fmul double %x, 2.0
  %frac = call double @modf(double %x2, ptr %p)
  %res = load double, ptr %p, align 8
  ret double %res
}

define double @df_unused_frac(double %x) {
entry:
  %d = call double (...) @__enzyme_autodiff(ptr @f_unused_frac, double %x)
  ret double %d
}

; Indirect write through a loaded pointer must still zero the shadow pointee.
define void @f_holder(ptr %h) {
entry:
  %yp = load ptr, ptr %h, align 8
  %frac = call double @modf(double 3.5, ptr %yp)
  ret void
}

define void @df_holder(ptr %h, ptr %dh) {
entry:
  call void (...) @__enzyme_autodiff_v(ptr @f_holder, metadata !"enzyme_dup", ptr %h, ptr %dh)
  ret void
}

; Runtime activity must guard shadow zeroing so shadow == primal is not clobbered.
define void @f_rt(ptr %p) {
entry:
  %frac = call double @modf(double 3.5, ptr %p)
  ret void
}

define void @df_rt(ptr %p, ptr %dp) {
entry:
  call void (...) @__enzyme_autodiff_v(ptr @f_rt, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", ptr %p, ptr %dp)
  ret void
}

; CHECK-LABEL: define {{.*}}double @df_overwrite(
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }

; CHECK-LABEL: define {{.*}}double @df_overwrite_sum(
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 2.000000e+00
; CHECK-NEXT: }

; CHECK-LABEL: define {{.*}}double @df_unused_frac(
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret double 0.000000e+00
; CHECK-NEXT: }

; CHECK-LABEL: define void @df_holder(
; CHECK: load ptr, ptr %dh
; CHECK: call double @modf(double 3.500000e+00, ptr {{.*}}writeonly{{.*}})
; CHECK: store double 0.000000e+00, ptr
; CHECK: ret void

; NOOPT-LABEL: define internal void @diffef_rt(
; NOOPT: call double @modf(double 3.500000e+00, ptr {{.*}}writeonly{{.*}} %p)
; NOOPT: icmp ne ptr %p, %"p'"
; NOOPT: br i1 {{.*}}, label %invertentry_active, label %invertentry_amerge
; NOOPT: invertentry_active:
; NOOPT: store double 0.000000e+00, ptr %"p'"
