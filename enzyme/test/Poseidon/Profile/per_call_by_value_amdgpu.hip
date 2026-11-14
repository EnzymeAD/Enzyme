// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN --implicit-check-not=warning: %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s

// per_call_by_value_nvptx.cu for HIP. A kernel takes an aggregate by
// reference into its constant argument segment, which the frontend copies
// into a local object once the aggregate's address is taken; a pointer into a
// member of a kernel's closure or of a struct parameter is a pointer into that
// object, and its shadow is a local object of the aggregate's type in the
// stack's address space.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

struct qfunc {
  double c;
  __device__ void operator()(const double &x, double &y) const {
    y = c * x * x + 1.0;
  }
};

template <typename F>
__device__ void body(const F *f, const double *x, double *y) { (*f)(*x, *y); }

template <typename F>
__device__ void call_site(const F &f, const double *x, double *y) {
  __poseidon_optimize(body<F>, &f, x, y);
}

struct closure {
  qfunc qf;
  double *u;
  __device__ void operator()(int i) const {
    double a = u[i], b;
    call_site(qf, &a, &b);
    u[i] = 2.0 * b;
  }
};

template <typename L> __global__ void forall_kernel(L body, int n) {
  int i = threadIdx.x;
  if (i < n)
    body(i);
}

template __global__ void forall_kernel<closure>(closure, int);

struct data_t {
  double *u;
  qfunc qf;
};

__global__ void k_data(const data_t data, int n) {
  int i = blockIdx.x;
  if (i < n) {
    double a = data.u[i], b;
    __poseidon_optimize(body<qfunc>, &data.qf, &a, &b);
    data.u[i] = 2.0 * b;
  }
}

// GEN-DAG: [poseidon-profgen] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in void call_site<qfunc>(qfunc const&, double const*, double*): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon-profgen] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in k_data(data_t, int): 3 shadow object(s), output(s) 2

// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z13forall_kernelI7closureEvT_i(ptr addrspace(4) {{.*}}byref(%struct.closure) {{.*}}%0, i32
// IR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[CS:[0-9]+]], i8 0, i64 16
// IR: %[[CSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[CS]] to ptr
// IR: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0(ptr {{.*}}, ptr {{.*}}%[[CSG]],
// IR: ret void

// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z6k_data6data_ti(ptr addrspace(4) {{.*}}byref(%struct.data_t) {{.*}}%0, i32
// IR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[DS:[0-9]+]], i8 0, i64 16
// IR: %[[DSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DS]] to ptr
// IR: %[[QFS:[0-9]+]] = getelementptr inbounds nuw i8, ptr %[[DSG]], i64 8
// IR: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0(ptr {{.*}}, ptr {{.*}}%[[QFS]],
// IR: ret void

// USE-NOT: is not a Poseidon site
// USE-DAG: [poseidon] _Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-NOT: is not a Poseidon site
