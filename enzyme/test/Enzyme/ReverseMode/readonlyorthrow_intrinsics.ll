; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.assume(i1 noundef)
declare float @llvm.maxnum.f32(float, float)
declare float @expf(float)

define float @helper_with_intrinsics(ptr nocapture %x, i64 %n) {
entry:
  %tmp = alloca float, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %tmp)
  %cmp = icmp sgt i64 %n, 0
  call void @llvm.assume(i1 %cmp)
  %v = load float, ptr %x, align 4
  %m = call float @llvm.maxnum.f32(float %v, float 0.000000e+00)
  %e = call float @expf(float %m)
  store float %e, ptr %tmp, align 4
  %res = load float, ptr %tmp, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %tmp)
  ret float %res
}

; CHECK: define float @helper_with_intrinsics(ptr {{.*}}readonly{{.*}} %x, i64 %n) #[[RO:[0-9]+]]
; CHECK: attributes #[[RO]] = { {{.*}}"enzyme_ReadOnlyOrThrow" }
