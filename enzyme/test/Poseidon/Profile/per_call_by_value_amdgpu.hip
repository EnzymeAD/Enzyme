// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN --implicit-check-not=warning: %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: FileCheck --check-prefix=BODY %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s

// A kernel takes an aggregate by reference into its constant argument segment,
// which the frontend copies into a local object once the aggregate's address is
// taken, and a device function takes a small one in registers, which the
// frontend stores to a local object. A pointer into a member of a kernel's
// closure, of a struct parameter or of a struct a device function takes by
// value is a pointer into that object: the primal pointer is passed as it is,
// with nothing stored through it, and its shadow is a local object of the
// aggregate's type in the stack's address space, whose seed follows the rule
// of a local object.

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

struct state {
  double s[2];
};

__device__ void advance(state *st) {
  st->s[1] = st->s[0] * st->s[0] + st->s[1];
}

__attribute__((noinline)) __device__ double run(state st) {
  __poseidon_optimize(advance, &st);
  return st.s[1];
}

__global__ void k_state(double *u) { u[0] = run(state{{u[0], u[1]}}); }

// GEN-DAG: [poseidon-profgen] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in void call_site<qfunc>(qfunc const&, double const*, double*): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon-profgen] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in k_data(data_t, int): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon-profgen] advance(state*) at call 0 in run(state): 1 shadow object(s), output(s) 0

// The functor's primal is the closure's local copy, its shadow a zero closure.
// a holds u[i], its shadow is zero; b is read after the call, its shadow is
// seeded.
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z13forall_kernelI7closureEvT_i(
// IR-SAME: ptr addrspace(4) {{.*}}byref(%struct.closure) {{.*}}%[[CLOSURE:[0-9]+]], i32
// IR: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[COPY:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// IR: %[[CS:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// IR: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) {{.*}}%[[COPY]], ptr addrspace(4) {{.*}}%[[CLOSURE]], i64 16, i1 false)
// IR-DAG: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// IR-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR: store double %[[AV]], ptr addrspace(5) %[[A]],
// IR-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[CS]], i8 0, i64 16, i1 false)
// IR: store i64 0, ptr addrspace(5) %[[DA]],
// IR: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// IR: %[[CSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[CS]] to ptr
// IR: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// IR: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// IR-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0(ptr %[[COPYG]], ptr %[[CSG]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// IR: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// IR: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR: store double %[[R]], ptr %[[UI]],
// IR: ret void
// BODY-LABEL: define {{.*}}amdgpu_kernel void @_Z13forall_kernelI7closureEvT_i(
// BODY: %[[COPY:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// BODY-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// BODY-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY-NOT: store {{.*}}, ptr %[[COPYG]],
// BODY-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[COPYG]],
// BODY: {{^}}}

// The member eight bytes into the parameter's local copy, and as far into its
// shadow; a and b as above.
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z6k_data6data_ti(
// IR-SAME: ptr addrspace(4) {{.*}}byref(%struct.data_t) {{.*}}%[[DATA:[0-9]+]], i32
// IR: %[[COPY:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// IR: %[[DS:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// IR: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) {{.*}}%[[COPY]], ptr addrspace(4) {{.*}}%[[DATA]], i64 16, i1 false)
// IR-DAG: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// IR-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR: store double %[[AV]], ptr addrspace(5) %[[A]],
// IR-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR: %[[QF:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[COPYG]], i64 8
// IR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[DS]], i8 0, i64 16, i1 false)
// IR: store i64 0, ptr addrspace(5) %[[DA]],
// IR: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// IR: %[[DSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DS]] to ptr
// IR: %[[QFS:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DSG]], i64 8
// IR: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// IR: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// IR-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0(ptr {{(nonnull )?}}%[[QF]], ptr {{(nonnull )?}}%[[QFS]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// IR: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// IR: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR: store double %[[R]], ptr %[[UI]],
// IR: ret void
// BODY-LABEL: define {{.*}}amdgpu_kernel void @_Z6k_data6data_ti(
// BODY: %[[COPY:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// BODY-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// BODY-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY-NOT: store {{.*}}, ptr %[[COPYG]],
// BODY-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[COPYG]],
// BODY: {{^}}}

// The struct arrives in registers and is stored to a local object; the call
// writes into it and it is read after the call, so its shadow is seeded.
// IR-LABEL: define {{.*}}double @_Z3run5state(
// IR-SAME: [2 x double] %[[V:[0-9]+]])
// IR: %[[ST:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// IR: %[[SS:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// IR: %[[V0:[0-9]+]] = extractvalue [2 x double] %[[V]], 0
// IR: store double %[[V0]], ptr addrspace(5) %[[ST]],
// IR: %[[V1:[0-9]+]] = extractvalue [2 x double] %[[V]], 1
// IR: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[ST]], i32 8
// IR: store double %[[V1]], ptr addrspace(5) %[[ST1]],
// IR: %[[STG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[ST]] to ptr
// IR: store double 1.000000e+00, ptr addrspace(5) %[[SS]],
// IR: %[[SS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[SS]], i32 8
// IR: store double 1.000000e+00, ptr addrspace(5) %[[SS1]],
// IR-NOT: store
// IR: %[[SSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[SS]] to ptr
// IR-NOT: store
// IR: call {{.*}} @diffe{{.*}}_Z7advanceP5state.poseidon._Z3run5state.0(ptr %[[STG]], ptr %[[SSG]])
// IR: %[[R:[0-9]+]] = load double, ptr addrspace(5) %[[ST1]],
// IR: ret double %[[R]]

// USE-NOT: cannot be a Poseidon site
// USE-DAG: per_call_by_value_amdgpu.hip:[[@LINE-145]]:39: warning: Poseidon: no profile at {{.*}}Inputs/other_sites/preprocess__Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0.fpprofile:
// USE-DAG: per_call_by_value_amdgpu.hip:[[@LINE-133]]:17: warning: Poseidon: no profile at {{.*}}Inputs/other_sites/preprocess__Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0.fpprofile:
// USE-DAG: per_call_by_value_amdgpu.hip:[[@LINE-117]]:45: warning: Poseidon: no profile at {{.*}}Inputs/other_sites/preprocess__Z7advanceP5state.poseidon._Z3run5state.0.fpprofile:
// USE-NOT: cannot be a Poseidon site
// USE: {{^}}3 warnings generated when compiling for {{.+}}.{{$}}
