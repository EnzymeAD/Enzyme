; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EXEC; fi

; Kokkos::atomic_add / atomic_sub on complex<T> call Desul's fetch_oper with
; the two components ({double, double} is passed as two scalars, <2 x float>
; as one vector). With the result unused (as for atomic_add) the call is
; lowered to one atomicrmw per component before differentiation.

%complexd = type { double, double }

; Stand-ins for the Desul templates: old = *dest; *dest += val; return old.
define %complexd @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_add_fetch_operatorIN6Kokkos7complexIdEEKS5_EES5_NS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %op, ptr %dest, double %re, double %im, i8 %order, i8 %scope) noinline {
  %old = load %complexd, ptr %dest, align 8
  %ore = extractvalue %complexd %old, 0
  %oim = extractvalue %complexd %old, 1
  %nre = fadd double %ore, %re
  %nim = fadd double %oim, %im
  %n0 = insertvalue %complexd undef, double %nre, 0
  %n1 = insertvalue %complexd %n0, double %nim, 1
  store %complexd %n1, ptr %dest, align 8
  ret %complexd %old
}

define <2 x float> @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_sub_fetch_operatorIN6Kokkos7complexIfEEKS5_EES5_NS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %op, ptr %dest, <2 x float> %val, i8 %order, i8 %scope) noinline {
  %old = load <2 x float>, ptr %dest, align 4
  %new = fsub <2 x float> %old, %val
  store <2 x float> %new, ptr %dest, align 4
  ret <2 x float> %old
}

declare double @__enzyme_autodiff(...)
declare <2 x float> @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)
@fmt = private constant [10 x i8] c"%f %f %f\0A\00"

define double @kernel(ptr %buf, double %x) {
entry:
  %op = alloca i8, align 1
  %sq = fmul double %x, %x
  %tw = fmul double %x, 2.0
  %old = call %complexd @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_add_fetch_operatorIN6Kokkos7complexIdEEKS5_EES5_NS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %op, ptr %buf, double %sq, double %tw, i8 0, i8 0)
  %new = load %complexd, ptr %buf, align 8
  %re = extractvalue %complexd %new, 0
  %im = extractvalue %complexd %new, 1
  %res = fmul double %re, %im
  ret double %res
}

define double @dkernel(ptr %buf, ptr %dbuf, double %x) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kernel, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_out", double %x)
  ret double %dx
}

define <2 x float> @kernelf(ptr %buf, <2 x float> %v) {
entry:
  %op = alloca i8, align 1
  %old = call <2 x float> @_ZN5desul4Impl22host_atomic_fetch_operINS0_19_sub_fetch_operatorIN6Kokkos7complexIfEEKS5_EES5_NS_18MemoryOrderRelaxedENS_17MemoryScopeDeviceEEET0_RKT_PS6_S6_T1_T2_(ptr %op, ptr %buf, <2 x float> %v, i8 0, i8 0)
  %new = load <2 x float>, ptr %buf, align 4
  ret <2 x float> %new
}

define <2 x float> @dkernelf(ptr %buf, ptr %dbuf, <2 x float> %v, <2 x float> %dv) {
entry:
  %r = call <2 x float> (...) @__enzyme_fwddiff(ptr @kernelf, metadata !"enzyme_dup", ptr %buf, ptr %dbuf, metadata !"enzyme_dup", <2 x float> %v, <2 x float> %dv)
  ret <2 x float> %r
}

define i32 @main() {
  %buf = alloca %complexd, align 8
  %dbuf = alloca %complexd, align 8
  store %complexd { double 1.0, double 2.0 }, ptr %buf
  store %complexd zeroinitializer, ptr %dbuf
  %dx = call double @dkernel(ptr %buf, ptr %dbuf, double 3.0)
  %db = load %complexd, ptr %dbuf
  %dbre = extractvalue %complexd %db, 0
  %dbim = extractvalue %complexd %db, 1
  call i32 (ptr, ...) @printf(ptr @fmt, double %dx, double %dbre, double %dbim)
  ret i32 0
}

; buf = (1 + x^2, 2 + 2x) = (10, 8); res = re * im.
; d res/dx = im * 2x + re * 2 = 48 + 20 = 68; d(buf) = (im, re) = (8, 10).
; EXEC: 68.000000 8.000000 10.000000

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK-NOT: call %complexd
; CHECK: atomicrmw fadd ptr %buf, double %sq monotonic, align 8
; CHECK-NEXT: %[[DIMP:.+]] = getelementptr inbounds double, ptr %"buf'", i64 1
; CHECK-NEXT: %[[IMP:.+]] = getelementptr inbounds double, ptr %buf, i64 1
; CHECK-NEXT: atomicrmw fadd ptr %[[IMP]], double %tw monotonic, align 8
; CHECK-NOT: call %complexd
; CHECK: load atomic double, ptr %[[DIMP]] monotonic, align 8
; CHECK: load atomic double, ptr %"buf'" monotonic, align 8

; CHECK-LABEL: define internal <2 x float> @fwddiffekernelf(
; CHECK-NOT: call <2 x float>
; CHECK: %[[DV0:.+]] = extractelement <2 x float> %"v'", i64 0
; CHECK-NEXT: %[[V0:.+]] = extractelement <2 x float> %v, i64 0
; CHECK-NEXT: %[[DV1:.+]] = extractelement <2 x float> %"v'", i64 1
; CHECK-NEXT: %[[V1:.+]] = extractelement <2 x float> %v, i64 1
; CHECK-NEXT: atomicrmw fsub ptr %buf, float %[[V0]] monotonic, align 4
; CHECK-NEXT: atomicrmw fsub ptr %"buf'", float %[[DV0]] monotonic, align 4
; CHECK-NEXT: %[[DP1:.+]] = getelementptr inbounds float, ptr %"buf'", i64 1
; CHECK-NEXT: %[[P1:.+]] = getelementptr inbounds float, ptr %buf, i64 1
; CHECK-NEXT: atomicrmw fsub ptr %[[P1]], float %[[V1]] monotonic, align 4
; CHECK-NEXT: atomicrmw fsub ptr %[[DP1]], float %[[DV1]] monotonic, align 4
; CHECK-NOT: call <2 x float>
