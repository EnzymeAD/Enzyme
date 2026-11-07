; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -preserve-nvvm -globalopt -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,globalopt,enzyme,preserve-nvvm-end,globalopt" -S | FileCheck %s --check-prefixes=CHECK,CLEANUP
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,globalopt,enzyme,preserve-nvvm-end,globalopt" -S | %lli -

; Internalized zero-initialized marker globals (such as `int enzyme_dup;` under
; LTO without `-rdynamic`) must not be folded to `i32 0` by GlobalOpt before
; Enzyme parses `__enzyme_autodiff`, and should be cleaned up once
; `preserve-nvvm-end` restores their initializer visibility.

@enzyme_dup = internal global i32 0, align 4
@enzyme_dupnoneed = internal global i32 0, align 4
@enzyme_out = internal global i32 0, align 4
@enzyme_const = internal global i32 0, align 4

declare double @llvm.log.f64(double)
declare void @abort()
declare void @_Z17__enzyme_autodiff(ptr, i32, ptr, ptr, i32, ptr, ptr, i32, ptr, ptr, i32, ptr)

define internal double @logpdf(ptr %x, ptr %mu, ptr %sigma, ptr %scale) {
entry:
  %xv = load double, ptr %x, align 8
  %muv = load double, ptr %mu, align 8
  %sv = load double, ptr %sigma, align 8
  %sc = load double, ptr %scale, align 8
  %diff = fsub double %xv, %muv
  %z = fdiv double %diff, %sv
  %z2 = fmul double %z, %z
  %half_z2 = fmul double -5.000000e-01, %z2
  %log_s = call double @llvm.log.f64(double %sv)
  %base = fsub double %half_z2, %log_s
  %res = fmul double %base, %sc
  ret double %res
}

define void @grad_logpdf(double %x, double %mu, double %sigma, double %scale, ptr %out) {
entry:
  %x.addr = alloca double, align 8
  %mu.addr = alloca double, align 8
  %sigma.addr = alloca double, align 8
  %scale.addr = alloca double, align 8
  store double %x, ptr %x.addr, align 8
  store double %mu, ptr %mu.addr, align 8
  store double %sigma, ptr %sigma.addr, align 8
  store double %scale, ptr %scale.addr, align 8
  %d_x = getelementptr inbounds double, ptr %out, i64 0
  %d_mu = getelementptr inbounds double, ptr %out, i64 1
  %d_sigma = getelementptr inbounds double, ptr %out, i64 2
  %m_dup0 = load i32, ptr @enzyme_dup, align 4
  %m_dup1 = load i32, ptr @enzyme_dup, align 4
  %m_dup2 = load i32, ptr @enzyme_dupnoneed, align 4
  %m_const = load i32, ptr @enzyme_const, align 4
  call void @_Z17__enzyme_autodiff(ptr @logpdf, i32 %m_dup0, ptr %x.addr, ptr %d_x, i32 %m_dup1, ptr %mu.addr, ptr %d_mu, i32 %m_dup2, ptr %sigma.addr, ptr %d_sigma, i32 %m_const, ptr %scale.addr)
  ret void
}

define i32 @main() {
entry:
  %grads = alloca [3 x double], align 16
  %g0 = getelementptr inbounds [3 x double], ptr %grads, i64 0, i64 0
  %g1 = getelementptr inbounds [3 x double], ptr %grads, i64 0, i64 1
  %g2 = getelementptr inbounds [3 x double], ptr %grads, i64 0, i64 2
  store double 0.000000e+00, ptr %g0, align 8
  store double 0.000000e+00, ptr %g1, align 8
  store double 0.000000e+00, ptr %g2, align 8
  call void @grad_logpdf(double 5.000000e-01, double 0.000000e+00, double 1.000000e+00, double 2.000000e+00, ptr %g0)
  %v0 = load double, ptr %g0, align 8
  %v1 = load double, ptr %g1, align 8
  %v2 = load double, ptr %g2, align 8
  ; At x=0.5, mu=0.0, sigma=1.0, scale=2.0:
  ; d/dx = -2 * 0.5 = -1.0, d/dmu = +1.0, d/dsigma = 2 * (0.25 - 1.0) = -1.5
  %e0 = fsub double %v0, -1.000000e+00
  %ok0a = fcmp olt double %e0, 1.000000e-12
  %ok0b = fcmp ogt double %e0, -1.000000e-12
  %ok0 = and i1 %ok0a, %ok0b
  %e1 = fsub double %v1, 1.000000e+00
  %ok1a = fcmp olt double %e1, 1.000000e-12
  %ok1b = fcmp ogt double %e1, -1.000000e-12
  %ok1 = and i1 %ok1a, %ok1b
  %e2 = fsub double %v2, -1.500000e+00
  %ok2a = fcmp olt double %e2, 1.000000e-12
  %ok2b = fcmp ogt double %e2, -1.000000e-12
  %ok2 = and i1 %ok2a, %ok2b
  %ok01 = and i1 %ok0, %ok1
  %ok = and i1 %ok01, %ok2
  br i1 %ok, label %done, label %fail

fail:
  call void @abort()
  unreachable

done:
  ret i32 0
}

; CLEANUP-NOT: @enzyme_
; CHECK-LABEL: define void @grad_logpdf(
; CHECK-NOT: _Z17__enzyme_autodiff
; CHECK: call {{.*}}void @diffe{{.*}}logpdf(
