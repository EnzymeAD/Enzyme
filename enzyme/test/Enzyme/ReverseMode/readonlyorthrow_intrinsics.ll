; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
declare void @llvm.enzyme.lifetime_start(...)
declare void @llvm.enzyme.lifetime_end(...)
declare void @llvm.assume(i1 noundef)
declare float @llvm.maxnum.f32(float, float)
declare float @expf(float)
declare noalias ptr @malloc(i64)

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

define ptr @helper_with_enzyme_lifetime(ptr nocapture %x) {
entry:
  %buf = call noalias ptr @malloc(i64 8)
  call void (...) @llvm.enzyme.lifetime_start(i64 8, ptr nonnull %buf)
  %v = load i64, ptr %x, align 8
  store i64 %v, ptr %buf, align 8
  call void (...) @llvm.enzyme.lifetime_end(i64 8, ptr nonnull %buf)
  ret ptr null
}

; CHECK: define float @helper_with_intrinsics(ptr {{.*}}readonly{{.*}} %x, i64 %n) #[[RO:[0-9]+]]
; CHECK: define ptr @helper_with_enzyme_lifetime(ptr {{.*}}readonly{{.*}} %x) #[[RO]]
; CHECK: attributes #[[RO]] = { {{.*}}"enzyme_ReadOnlyOrThrow" }
