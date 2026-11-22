// REQUIRES: enzyme, nvptx-device-profiler
// DEFINE: %{clang} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -o %t/gen.ll
// RUN: FileCheck --check-prefix=GEN --implicit-check-not=__poseidon_site_explicit %s < %t/gen.ll
// RUN: %{clang} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE %s
// RUN: FileCheck --check-prefix=IR --implicit-check-not=__poseidon_site_explicit %s < %t/use.ll
// RUN: %{clang} -o - | FileCheck --check-prefix=PLAIN %s
// RUN: %clang -x cuda --cuda-host-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/host.ll 2>&1 | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: FileCheck --check-prefix=HOST --implicit-check-not=__poseidon_site_explicit --implicit-check-not=__poseidon_fp_optimize %s < %t/host.ll

// In device code an explicit list through __poseidon_optimize makes the call
// a site over the function itself, with the shadows the caller gives: here
// sum reaches no constant range of memory through x, which the pass could not
// make the shadow of. poseidon_tau followed by the function's own arguments
// is no explicit list: that call is a per-call site, as the call
// __poseidon_optimize_tau writes. Without the plugin the call goes through an
// identity nothing defines. The host half of a host-device function holds no
// site: there the call runs the function with the primal values, silently.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

#ifdef __CUDA_ARCH__
__attribute__((device)) int enzyme_dup, enzyme_const, poseidon_tau;
#else
int enzyme_dup, enzyme_const, poseidon_tau;
#endif

__attribute__((device)) double sum(const double *x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  return s;
}

__attribute__((device)) void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i] * x[i] + 1.0;
}

HD __attribute__((noinline)) double norm2(const double *x) {
  return x[0] * x[0] + x[1] * x[1];
}

HD double twice(const double *x, double *dx) {
  return 2.0 * __poseidon_optimize(norm2, enzyme_dup, x, dx);
}

HD __attribute__((noinline)) double spread(const double *x) {
  return x[0] * x[1] - x[1] * x[1];
}

HD double third(const double *x) {
  return __poseidon_optimize(spread, poseidon_tau, 1e-7, x);
}

__attribute__((global)) void k(const double *x, double *dx, double *y,
                               double *dy, int n) {
  y[0] = __poseidon_optimize(sum, enzyme_dup, x, dx, n);
  y[3] = twice(x, dx);
  __poseidon_optimize_tau(1e-6, scale, enzyme_const, x, enzyme_dup, y + 1, dy,
                          n);
  y[2] = __poseidon_optimize(sum, poseidon_tau, 1e-9, enzyme_const, x, n);
  y[4] = third(x);
}

// GEN-DAG: @poseidon_site_preprocess__Z3sumPKdi =
// GEN-DAG: @poseidon_site_preprocess__Z5scalePKdPdi =
// GEN-DAG: @poseidon_site_preprocess__Z5norm2PKd =
// GEN-DAG: @poseidon_site_preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0 =

// USE-DAG: [poseidon] _Z3sumPKdi: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z3sumPKdi.fpprofile); left unchanged
// USE-DAG: [poseidon] _Z5scalePKdPdi: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z5scalePKdPdi.fpprofile); left unchanged
// USE-DAG: [poseidon] _Z5norm2PKd: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z5norm2PKd.fpprofile); left unchanged
// USE-DAG: [poseidon] _Z6spreadPKd.poseidon._Z5thirdPKd.0: the profiled program did not contain this site (no profile at {{.*}}/preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0.fpprofile); left unchanged
// IR-LABEL: define {{.*}} @_Z1kPKdPdS1_S1_i(
// IR: call double @_Z3sumPKdi(
// IR: call void @_Z5scalePKdPdi(
// IR: call double @_Z3sumPKdi(
// IR: call {{.*}}double @_Z6spreadPKd(

// PLAIN-DAG: call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z3sumPKdi)
// PLAIN-DAG: call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z6spreadPKd)
// PLAIN-DAG: call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z5scalePKdPdi, double
// PLAIN-DAG: declare {{.*}}ptr @__poseidon_site_explicit(ptr

// QUIET-NOT: {{.}}
// HOST-LABEL: define {{.*}} @_Z5twicePKdPd(
// HOST: call {{.*}}double @_Z5norm2PKd(ptr
// HOST-LABEL: define {{.*}} @_Z5thirdPKd(
// HOST: call {{.*}}double @_Z6spreadPKd(ptr
