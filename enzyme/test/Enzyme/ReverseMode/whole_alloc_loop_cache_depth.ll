; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; The stack allocation %phi is defined in the entry block, but its lifetime
; (and therefore the scope in which its contents are known to be valid) is one
; iteration of the loop. Caching the whole allocation would therefore require
; caching its contents for every iteration of the loop, and is more expensive
; than caching the single value stored into it within the loop (and
; rematerializing the allocation). When choosing what to cache, the allocation
; should thus be considered to be at the loop depth of its lifetime, with the
; size of its contents, rather than a pointer outside of the loop.

@enzyme_const = external global i32

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
declare void @__enzyme_autodiff(i8*, ...)

define double @energy(double* %D, double* %x, i64 %np, i64 %k, i64 %k2) {
entry:
  %phi = alloca [3 x double], align 16
  %phi_i8 = bitcast [3 x double]* %phi to i8*
  br label %loop

loop:
  %p = phi i64 [ 0, %entry ], [ %p_next, %loop ]
  %s = phi double [ 0.0, %entry ], [ %s_next, %loop ]
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %phi_i8)
  %x_gep = getelementptr inbounds double, double* %x, i64 %p
  %xv = load double, double* %x_gep, align 8
  ; Overwrite the input so that it cannot be reloaded in the reverse pass.
  store double 0.0, double* %x_gep, align 8
  %v = fadd double %xv, 1.0
  %phi_k = getelementptr inbounds [3 x double], [3 x double]* %phi, i64 0, i64 %k
  store double %v, double* %phi_k, align 8
  %phi_k2 = getelementptr inbounds [3 x double], [3 x double]* %phi, i64 0, i64 %k2
  %l = load double, double* %phi_k2, align 8
  %D_gep = getelementptr inbounds double, double* %D, i64 %p
  %Dv = load double, double* %D_gep, align 8
  %r = fmul double %l, %Dv
  %s_next = fadd double %s, %r
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %phi_i8)
  %p_next = add nuw nsw i64 %p, 1
  %cmp_p = icmp eq i64 %p_next, %np
  br i1 %cmp_p, label %exit, label %loop

exit:
  ret double %s_next
}

define void @caller(double* %D, double* %dD, double* %x, i64 %np, i64 %k, i64 %k2) {
entry:
  %c = load i32, i32* @enzyme_const
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, double*, i64, i64, i64)* @energy to i8*), double* %D, double* %dD, i32 %c, double* %x, i64 %np, i64 %k, i64 %k2)
  ret void
}

; The value stored is cached, and the allocation is rematerialized, rather than
; caching the contents of the whole allocation for each iteration.
; CHECK: define internal void @diffeenergy(
; CHECK-NOT: @llvm.memcpy
; CHECK: remat_loop_loop:
; CHECK-NOT: @llvm.memcpy
; CHECK: store double %{{.*}}, {{double\*|ptr}} %phi_k_unwrap
