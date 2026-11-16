// REQUIRES: enzyme, amdgpu-registered-target, rocm
// DEFINE: %{clang} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -o %t/gen.ll
// RUN: FileCheck --check-prefix=GEN --implicit-check-not=__poseidon_site_explicit %s < %t/gen.ll
// RUN: %{clang} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE %s
// RUN: FileCheck --check-prefix=IR --implicit-check-not=__poseidon_site_explicit %s < %t/use.ll
// RUN: %{clang} -o - | FileCheck --check-prefix=PLAIN %s

// per_call_explicit_nvptx.cu for HIP: in AMDGPU device code an explicit list
// through __poseidon_optimize makes the call a site over the function itself,
// with the shadows the caller gives. Without the plugin the call goes through
// an identity nothing defines.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#define HD __host__ __device__

__device__ int enzyme_dup, enzyme_const, poseidon_tau;

__device__ double sum(const double *x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  return s;
}

__device__ void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i] * x[i] + 1.0;
}

HD __attribute__((noinline)) double norm2(const double *x) {
  return x[0] * x[0] + x[1] * x[1];
}

HD double twice(const double *x, double *dx) {
  return 2.0 * __poseidon_optimize(norm2, enzyme_dup, x, dx);
}

__global__ void k(const double *x, double *dx, double *y, double *dy, int n) {
  y[0] = __poseidon_optimize(sum, enzyme_dup, x, dx, n);
  y[3] = twice(x, dx);
  __poseidon_optimize_tau(1e-6, scale, enzyme_const, x, enzyme_dup, y + 1, dy,
                          n);
  y[2] = __poseidon_optimize(sum, poseidon_tau, 1e-9, enzyme_const, x, n);
}

// GEN-DAG: @poseidon_site_preprocess__Z3sumPKdi =
// GEN-DAG: @poseidon_site_preprocess__Z5scalePKdPdi =
// GEN-DAG: @poseidon_site_preprocess__Z5norm2PKd =

// USE-DAG: [poseidon] _Z3sumPKdi: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z3sumPKdi.fpprofile); left unchanged
// USE-DAG: [poseidon] _Z5scalePKdPdi: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z5scalePKdPdi.fpprofile); left unchanged
// USE-DAG: [poseidon] _Z5norm2PKd: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z5norm2PKd.fpprofile); left unchanged
// IR-LABEL: define {{.*}} @_Z1kPKdPdS1_S1_i(
// IR: call double @_Z3sumPKdi(
// IR: call void @_Z5scalePKdPdi(
// IR: call double @_Z3sumPKdi(

// PLAIN-DAG: call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z3sumPKdi)
// PLAIN-DAG: call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z5scalePKdPdi, double
// PLAIN-DAG: declare {{.*}}ptr @__poseidon_site_explicit(ptr
