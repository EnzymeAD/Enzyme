; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EXEC

; make_env writes only its sret parameter, so Enzyme marks it as writing no
; memory the caller can see. It still hands %cslot back to the caller inside
; that sret, from where the closure later loads through it as a double. The
; slot therefore needs a (zero) shadow; treating it as inactive aliased its
; shadow with the primal and leaked the primal c into the tangent.

declare double @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)
@enzyme_dup = external global i8
@enzyme_const = external global i8
@fmt = private constant [7 x i8] c"%f %f\0A\00"

define internal double @closure(ptr %env, ptr %xp) {
  %cp = load ptr, ptr %env, align 8
  %c = load double, ptr %cp, align 8
  %x = load double, ptr %xp, align 8
  %cx = fmul double %c, %x
  %cxx = fmul double %cx, %x
  ret double %cxx
}

define internal void @make_env(ptr sret([16 x i8]) %out, ptr %x, ptr %cp) {
  store ptr %x, ptr %out, align 8
  %slot = getelementptr inbounds i8, ptr %out, i64 8
  store ptr %cp, ptr %slot, align 8
  ret void
}

define internal double @sum(ptr %env) {
  %x = load ptr, ptr %env, align 8
  %envc = getelementptr inbounds i8, ptr %env, i64 8
  %r = call double @closure(ptr %envc, ptr %x)
  ret double %r
}

define double @f(ptr %x, double %c) {
  %env = alloca [16 x i8], align 8
  %cslot = alloca double, align 8
  store double %c, ptr %cslot, align 8
  call void @make_env(ptr sret([16 x i8]) %env, ptr %x, ptr %cslot)
  %r = call double @sum(ptr %env)
  ret double %r
}

define double @df(ptr %x, ptr %dx, double %c) {
  %r = call double (...) @__enzyme_fwddiff(ptr @f, ptr @enzyme_dup, ptr %x, ptr %dx, ptr @enzyme_const, double %c)
  ret double %r
}

define i32 @main() {
  %x = alloca double, align 8
  %dx = alloca double, align 8
  store double 3.0, ptr %x
  store double 1.0, ptr %dx
  %p = call double @f(ptr %x, double 2.0)
  %r = call double @df(ptr %x, ptr %dx, double 2.0)
  call i32 (ptr, ...) @printf(ptr @fmt, double %p, double %r)
  ret i32 0
}

; EXEC: 18.000000 12.000000

; CHECK-LABEL: define internal double @fwddiffef(ptr %x, ptr %"x'", double %c)
; CHECK: %"cslot'ipa" = alloca double
; CHECK: store double 0.000000e+00, ptr %"cslot'ipa"
; CHECK: call void @fwddiffemake_env(ptr {{.*}}%env, ptr {{.*}}%"env'ipa", ptr %x, ptr %"x'", ptr %cslot, ptr %"cslot'ipa")
