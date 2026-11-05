; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Check that DetectPointerArgOfFn traces pointer arguments stored into
; non-escaping -O0 parameter stack slots (`alloca ptr`), deducing `readonly`
; and `nocapture` on `%src` even when `@copy_scaled` itself writes to `%dst`.

declare double @__enzyme_autodiff(...)

define void @copy_scaled(ptr %src, ptr %dst, double %scale) noinline {
entry:
  %src.addr = alloca ptr, align 8
  %dst.addr = alloca ptr, align 8
  %scale.addr = alloca double, align 8
  store ptr %src, ptr %src.addr, align 8
  store ptr %dst, ptr %dst.addr, align 8
  store double %scale, ptr %scale.addr, align 8
  %s_ptr = load ptr, ptr %src.addr, align 8
  %w = load double, ptr %s_ptr, align 8
  %sc = load double, ptr %scale.addr, align 8
  %prod = fmul double %w, %sc
  %d_ptr = load ptr, ptr %dst.addr, align 8
  store double %prod, ptr %d_ptr, align 8
  ret void
}

define void @caller(ptr %out, double %scale) {
entry:
  %weight = alloca double, align 8
  store double 3.000000e+00, ptr %weight, align 8
  call void @copy_scaled(ptr %weight, ptr %out, double %scale)
  ret void
}

define double @grad_caller(ptr %out, ptr %dout, double %scale) {
entry:
  %res = call double (...) @__enzyme_autodiff(ptr @caller, metadata !"enzyme_dupnoneed", ptr %out, ptr %dout, double %scale)
  ret double %res
}

; CHECK: define void @copy_scaled(ptr {{.*}}readonly {{.*}}%src, ptr {{.*}}writeonly {{.*}}%dst, double %scale)
; CHECK-LABEL: define internal { double } @diffecaller(
; CHECK-NOT: %"weight'ipa"
; CHECK: call {{(fast )?}}{ double } @diffecopy_scaled(ptr %weight, ptr {{%out|undef}}, ptr %"out'", double %scale)
