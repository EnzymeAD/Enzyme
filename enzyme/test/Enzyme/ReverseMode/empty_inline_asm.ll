; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 6.000000 10.000000 8.000000

@.fmt = private unnamed_addr constant [13 x i8] c"%f %f %f\0A\00\00\00\00", align 1

declare i32 @printf(ptr, ...)
declare void @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)

define i32 @main() {
entry:
  %vec = alloca [2 x double], align 16
  %dvec = alloca [2 x double], align 16
  %fslot = alloca double, align 8
  %dfslot = alloca double, align 8
  store double 3.000000e+00, ptr %vec, align 16
  %vec1 = getelementptr inbounds double, ptr %vec, i64 1
  store double 5.000000e+00, ptr %vec1, align 8
  store double 0.000000e+00, ptr %dvec, align 16
  %dvec1 = getelementptr inbounds double, ptr %dvec, i64 1
  store double 0.000000e+00, ptr %dvec1, align 8
  store double 4.000000e+00, ptr %fslot, align 8
  store double 1.000000e+00, ptr %dfslot, align 8
  call void (...) @__enzyme_autodiff(ptr @barrier_kernels, metadata !"enzyme_dup", ptr %vec, ptr %dvec)
  %g0 = load double, ptr %dvec, align 16
  %g1 = load double, ptr %dvec1, align 8
  %f0 = call double (...) @__enzyme_fwddiff(ptr @ptr_barrier_sq, metadata !"enzyme_dup", ptr %fslot, ptr %dfslot)
  %printed = call i32 (ptr, ...) @printf(ptr nonnull @.fmt, double %g0, double %g1, double %f0)
  ret i32 0
}

define double @x86_barrier_callers(ptr %x, ptr %dx, double %val) {
entry:
  call void (...) @__enzyme_autodiff(ptr @eigen_xmm_barrier, metadata !"enzyme_dup", ptr %x, ptr %dx)
  %f = call double (...) @__enzyme_fwddiff(ptr @reg_barrier_sq, double %val, double 1.000000e+00)
  ret double %f
}

; CHECK-LABEL: define internal void @diffebarrier_kernels(
define double @barrier_kernels(ptr nocapture readonly %input) {
entry:
  %slot = alloca <2 x double>, align 16
  call void asm sideeffect "", ""()
  %loaded = load <2 x double>, ptr %input, align 16
  %sq = fmul <2 x double> %loaded, %loaded
  store <2 x double> %sq, ptr %slot, align 16
  call void asm "", "=*m,0"(ptr nonnull elementtype(<2 x double>) %slot, <2 x double> %sq)
  %slot_kept = call ptr asm "", "=r,0"(ptr nonnull %slot)
  %after = load <2 x double>, ptr %slot_kept, align 16
  %lane0 = extractelement <2 x double> %after, i64 0
  %lane1 = extractelement <2 x double> %after, i64 1
  ; CHECK: fadd fast <2 x double>
  %sum = fadd double %lane0, %lane1
  ret double %sum
}

; CHECK-LABEL: define internal double @fwddiffeptr_barrier_sq(
define double @ptr_barrier_sq(ptr nocapture readonly %p) {
entry:
  %p_kept = call ptr asm "", "=r,0"(ptr nonnull %p)
  call void asm "", "=*m,*m"(ptr nonnull elementtype(double) %p_kept, ptr nonnull elementtype(double) %p_kept)
  %x = load double, ptr %p_kept, align 8
  ; CHECK: fmul fast double
  ; CHECK: fadd fast double
  %sq = fmul double %x, %x
  ret double %sq
}

; CHECK-LABEL: define internal void @diffeeigen_xmm_barrier(
define double @eigen_xmm_barrier(ptr nocapture readonly %input) {
entry:
  %slot = alloca <2 x double>, align 16
  call void asm sideeffect "#EIGEN PRODUCT PACK LHS", "~{dirflag},~{fpsr},~{flags}"()
  %loaded = load <2 x double>, ptr %input, align 16
  %scaled = fmul <2 x double> %loaded, <double 2.000000e+00, double 3.000000e+00>
  store <2 x double> %scaled, ptr %slot, align 16
  call void asm "", "=*imr|x,0,~{dirflag},~{fpsr},~{flags}"(ptr nonnull elementtype(<2 x double>) %slot, <2 x double> %scaled)
  %after = load <2 x double>, ptr %slot, align 16
  %lane0 = extractelement <2 x double> %after, i64 0
  %lane1 = extractelement <2 x double> %after, i64 1
  ; CHECK: fmul fast <2 x double> %{{.+}}, <double 2.000000e+00, double 3.000000e+00>
  %sum = fadd double %lane0, %lane1
  ret double %sum
}

; CHECK-LABEL: define internal double @fwddiffereg_barrier_sq(
define double @reg_barrier_sq(double %x) {
entry:
  %kept = call double asm "", "=r,0"(double %x)
  ; CHECK: %[[P0:.+]] = fmul fast double %"x'", %kept
  ; CHECK: %[[P1:.+]] = fmul fast double %"x'", %kept
  ; CHECK: %[[SUM:.+]] = fadd fast double %[[P0]], %[[P1]]
  ; CHECK: ret double %[[SUM]]
  %sq = fmul double %kept, %kept
  ret double %sq
}
