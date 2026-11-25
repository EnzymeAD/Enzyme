// REQUIRES: enzyme, nvptx-device-profiler
// DEFINE: %{clang} = %clang -x cuda -nocudainc -nocudalib -O1 -S -emit-llvm \
// DEFINE:   -I%poseidon_profile_inc %s
// DEFINE: %{host} = %{clang} --cuda-host-only -mllvm -poseidon-compilation-id=u
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} --cuda-device-only --cuda-gpu-arch=%gpu_arch %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-compilation-id=u -mllvm -poseidon-cache=%t/c -o %t/dev.ll 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: %{host} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -o %t/gen.ll 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: FileCheck --check-prefixes=CALLS,STUB --implicit-check-not=__poseidon_fp_optimize --implicit-check-not=__enzyme_fp_optimize --implicit-check-not=__enzyme_autodiff %s < %t/gen.ll
// RUN: %{host} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -DNO_MARKERS -o %t/nomarkers.ll
// RUN: FileCheck --check-prefix=STUB %s < %t/nomarkers.ll
// RUN: sed -n '/^define .*@_Z19__device_stub__kickPdPKdd(/,/^}/p' %t/gen.ll > %t/stub.gen
// RUN: sed -n '/^define .*@_Z19__device_stub__kickPdPKdd(/,/^}/p' %t/nomarkers.ll > %t/stub.nomarkers
// RUN: diff %t/stub.nomarkers %t/stub.gen
// RUN: %{host} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t/u -o %t/use.ll 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: FileCheck --check-prefix=CALLS --implicit-check-not=__poseidon_fp_optimize --implicit-check-not=__enzyme_fp_optimize %s < %t/use.ll
// RUN: %{host} %loadClangPoseidon -o %t/plain.ll 2>&1 | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: FileCheck --check-prefix=CALLS --implicit-check-not=__poseidon_fp_optimize --implicit-check-not=__enzyme_fp_optimize %s < %t/plain.ll

// The host half of a CUDA or HIP compilation holds no site, so there a marker
// call, under either name and with either list, is a call of the function
// with the primal values of its arguments, silently, in profile generation,
// profile use and a build without a profile; the shadow dy reaches no call.
// The launch stub of an annotated kernel is rewritten for profiling as without
// the marker calls.

#include "poseidon/poseidon.h"

#define __global__ __attribute__((global))

struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

int enzyme_dup, enzyme_const, poseidon_tau;
template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
template <typename R, typename... T> R __enzyme_fp_optimize(void *, T...);

// STUB-LABEL: define {{.*}} @_Z19__device_stub__kickPdPKdd(
// STUB: [[GRID:%[0-9]+]] = alloca %struct.dim3
// STUB-NEXT: [[BLOCK:%[0-9]+]] = alloca %struct.dim3
// STUB-NEXT: [[SHMEM:%[0-9]+]] = alloca i64
// STUB-NEXT: [[STREAM:%[0-9]+]] = alloca ptr
// STUB: [[ARGS:%[0-9]+]] = alloca [3 x ptr]
// STUB: call i32 @__cudaPopCallConfiguration(ptr {{.*}}[[GRID]], ptr {{.*}}[[BLOCK]], ptr {{.*}}[[SHMEM]], ptr {{.*}}[[STREAM]])
// STUB-NEXT: [[SM:%[0-9]+]] = load i64, ptr [[SHMEM]]
// STUB-NEXT: [[ST:%[0-9]+]] = load ptr, ptr [[STREAM]]
// STUB-NEXT: call void @__poseidon_launch_profiled(ptr {{.*}}@_Z19__device_stub__kickPdPKdd, i32 0, ptr {{.*}}[[GRID]], ptr {{.*}}[[BLOCK]], i64 [[SM]], ptr [[ST]], i32 3, ptr {{.*}}[[ARGS]], i32 2, ptr {{.*}}@poseidon.ptr._Z4kickPdPKdd, ptr {{.*}}@poseidon.off._Z4kickPdPKdd, ptr {{.*}}@poseidon.seed._Z4kickPdPKdd)
// STUB-NEXT: ret void
POSEIDON_OPTIMIZE __global__ void kick(double *v, const double *a, double dt) {
  v[0] += a[0] * dt;
}

__attribute__((noinline)) double sq(double x, double y) {
  return x * x + y * y;
}

__attribute__((noinline)) void axpy(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] += 2.0 * x[i];
}

// CALLS-LABEL: define {{.*}} @_Z4hostPKdPdS1_i(
// CALLS-SAME: ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, i32 {{.*}}[[N:%[0-9]+]])
// CALLS-NEXT: [[X0:%[0-9]+]] = load double, ptr [[X]]
// CALLS-NEXT: [[X1P:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[X]], i64 8
// CALLS-NEXT: [[X1:%[0-9]+]] = load double, ptr [[X1P]]
double host(const double *x, double *y, double *dy, int n) {
  double s = 0.0;
#ifndef NO_MARKERS
  // CALLS-NEXT: [[S:%[0-9]+]] = {{(tail )?}}call {{.*}}double @_Z2sqdd(double [[X0]], double [[X1]])
  s = __poseidon_fp_optimize<double>((void *)sq, x[0], x[1]);
  // CALLS-NEXT: {{(tail )?}}call void @_Z4axpyPKdPdi(ptr {{(nonnull )?}}[[X]], ptr {{(nonnull )?}}[[Y]], i32 [[N]])
  __enzyme_fp_optimize<void>((void *)axpy, enzyme_const, x, enzyme_dup, y, dy,
                             n);
  // CALLS-NEXT: {{(tail )?}}call void @_Z4axpyPKdPdi(ptr {{(nonnull )?}}[[X]], ptr {{(nonnull )?}}[[Y]], i32 [[N]])
  __poseidon_fp_optimize<void>((void *)axpy, poseidon_tau, 1e-6, enzyme_const,
                               x, enzyme_dup, y, dy, n);
  // CALLS-NEXT: {{(tail )?}}call void @_Z4axpyPKdPdi(ptr {{(nonnull )?}}[[X]], ptr {{(nonnull )?}}[[Y]], i32 [[N]])
  __enzyme_fp_optimize<void>((void *)axpy, x, y, n);
#endif
  // CALLS-NEXT: ret double [[S]]
  return s;
}

// QUIET-NOT: {{.}}
