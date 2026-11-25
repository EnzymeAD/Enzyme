// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{nvptx} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ %{nvptx} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/nvptx-c -mllvm -poseidon-print %s -o %t/nvptx.ll 2>&1 | FileCheck --check-prefix=GEN --implicit-check-not=warning: %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=IR-NVPTX %s < %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=BODY-NVPTX %s < %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t/nvptx-u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/amdgpu-c -mllvm -poseidon-print %s -o %t/amdgpu.ll 2>&1 | FileCheck --check-prefix=GEN --implicit-check-not=warning: %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=IR-AMDGPU %s < %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=BODY-AMDGPU %s < %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t/amdgpu-u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s %}

// A pointer into an aggregate the caller takes by value is a pointer into an
// object whose extent the pass knows, as one into a local object is: a
// kernel's closure, a struct parameter of a kernel, a struct a device function
// takes by value. The shadow is a local object of the aggregate's type, and
// the primal pointer is passed as it is: nothing is stored through the
// parameter. Its seed follows the rule of a local object.

#include "poseidon/poseidon.h"

#define DEVICE __attribute__((device))
#define GLOBAL __attribute__((global))

#ifdef __NVPTX__
#define THREAD_X __nvvm_read_ptx_sreg_tid_x()
#define BLOCK_ID_X __nvvm_read_ptx_sreg_ctaid_x()
#define GRID_CONSTANT __attribute__((grid_constant))
#else
#define THREAD_X __builtin_amdgcn_workitem_id_x()
#define BLOCK_ID_X __builtin_amdgcn_workgroup_id_x()
#define GRID_CONSTANT
#endif

struct qfunc {
  double c;
  DEVICE void operator()(const double &x, double &y) const {
    y = c * x * x + 1.0;
  }
};

template <typename F>
DEVICE void body(const F *f, const double *x, double *y) { (*f)(*x, *y); }

// The closure of a lambda captured by value, the functor its first member.
template <typename F>
DEVICE void call_site(const F &f, const double *x, double *y) {
  __poseidon_optimize(body<F>, &f, x, y);
}

struct closure {
  qfunc qf;
  double *u;
  DEVICE void operator()(int i) const {
    double a = u[i], b;
    call_site(qf, &a, &b);
    u[i] = 2.0 * b;
  }
};

template <typename L> GLOBAL void forall_kernel(L body, int n) {
  int i = THREAD_X;
  if (i < n)
    body(i);
}

template GLOBAL void forall_kernel<closure>(closure, int);

// A member at a non-zero offset of a struct parameter, __grid_constant__ on
// NVPTX.
struct data_t {
  double *u;
  qfunc qf;
};

GLOBAL void k_data(GRID_CONSTANT const data_t data, int n) {
  int i = BLOCK_ID_X;
  if (i < n) {
    double a = data.u[i], b;
    __poseidon_optimize(body<qfunc>, &data.qf, &a, &b);
    data.u[i] = 2.0 * b;
  }
}

// An aggregate a device function takes by value and the call writes into,
// read after the call: an output, seeded.
struct state {
  double s[2];
};

DEVICE void advance(state *st) {
  st->s[1] = st->s[0] * st->s[0] + st->s[1];
}

__attribute__((noinline)) DEVICE double run(state st) {
  __poseidon_optimize(advance, &st);
  return st.s[1];
}

GLOBAL void k_state(double *u) {
  u[0] = run(state{{u[0], u[1]}});
}

// GEN-DAG: [poseidon] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in void call_site<qfunc>(qfunc const&, double const*, double*): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in k_data(data_t, int): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon] advance(state*) at call 0 in run(state): 1 shadow object(s), output(s) 0

// The functor's shadow is a zero closure; the primal is the parameter itself.
// a holds u[i], its shadow is zero; b is read after the call, its shadow is
// seeded.
// IR-NVPTX-LABEL: define {{.*}} @_Z13forall_kernelI7closureEvT_i(
// IR-NVPTX-SAME: ptr {{.*}}byval(%struct.closure) {{.*}}[[CLOSURE:%[0-9]+]], i32
// IR-NVPTX: %[[CS:[0-9]+]] = alloca %struct.closure, align 8
// IR-NVPTX: %[[A:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[DA:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[B:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[DB:[0-9]+]] = alloca double, align 8
// IR-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// IR-NVPTX: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR-NVPTX: store double %[[AV]], ptr %[[A]],
// IR-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// IR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[CS]], i8 0, i64 16, i1 false)
// IR-NVPTX: store i64 0, ptr %[[DA]],
// IR-NVPTX: store double 1.000000e+00, ptr %[[DB]],
// IR-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// IR-NVPTX: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0(ptr {{(nonnull )?}}[[CLOSURE]], ptr %[[CS]], ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// IR-NVPTX: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// IR-NVPTX: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR-NVPTX: store double %[[R]], ptr %[[UI]],
// IR-NVPTX: ret void
// BODY-NVPTX-LABEL: define {{.*}} @_Z13forall_kernelI7closureEvT_i(
// BODY-NVPTX-SAME: byval(%struct.closure) {{.*}}[[CLOSURE:%[0-9]+]], i32
// BODY-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// BODY-NVPTX-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[CLOSURE]],
// BODY-NVPTX: {{^}}}

// The member eight bytes into the parameter, and as far into its shadow; a and
// b as above.
// IR-NVPTX-LABEL: define {{.*}} @_Z6k_data6data_ti(
// IR-NVPTX-SAME: ptr {{.*}}byval(%struct.data_t) {{.*}}"nvvm.grid_constant" [[DATA:%[0-9]+]], i32
// IR-NVPTX: %[[DS:[0-9]+]] = alloca %struct.data_t, align 8
// IR-NVPTX: %[[A:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[DA:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[B:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[DB:[0-9]+]] = alloca double, align 8
// IR-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// IR-NVPTX: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR-NVPTX: store double %[[AV]], ptr %[[A]],
// IR-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// IR-NVPTX: %[[QF:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr [[DATA]], i64 8
// IR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DS]], i8 0, i64 16, i1 false)
// IR-NVPTX: store i64 0, ptr %[[DA]],
// IR-NVPTX: store double 1.000000e+00, ptr %[[DB]],
// IR-NVPTX: %[[QFS:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DS]], i64 8
// IR-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// IR-NVPTX: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0(ptr {{(nonnull )?}}%[[QF]], ptr %[[QFS]], ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// IR-NVPTX: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// IR-NVPTX: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR-NVPTX: store double %[[R]], ptr %[[UI]],
// IR-NVPTX: ret void
// BODY-NVPTX-LABEL: define {{.*}} @_Z6k_data6data_ti(
// BODY-NVPTX-SAME: byval(%struct.data_t) {{.*}}[[DATA:%[0-9]+]], i32
// BODY-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// BODY-NVPTX-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[DATA]],
// BODY-NVPTX: {{^}}}

// The call writes the parameter's second element, read after the call: both
// elements of the shadow are seeded, and nothing is stored through the
// parameter.
// IR-NVPTX-LABEL: define {{.*}} @_Z3run5state(
// IR-NVPTX-SAME: ptr {{.*}}byval(%struct.state) {{.*}}[[STATE:%[0-9]+]])
// IR-NVPTX: %[[SS:[0-9]+]] = alloca %struct.state
// IR-NVPTX-NOT: store
// IR-NVPTX: store double 1.000000e+00, ptr %[[SS]],
// IR-NVPTX: %[[SS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[SS]], i64 8
// IR-NVPTX: store double 1.000000e+00, ptr %[[SS1]],
// IR-NVPTX-NOT: store
// IR-NVPTX: call {{.*}} @diffe{{.*}}_Z7advanceP5state.poseidon._Z3run5state.0(ptr {{(nonnull )?}}[[STATE]], ptr %[[SS]])
// IR-NVPTX-NOT: store
// IR-NVPTX: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr [[STATE]], i64 8
// IR-NVPTX: %[[R:[0-9]+]] = load double, ptr %[[ST1]],
// IR-NVPTX: ret double %[[R]]
// BODY-NVPTX-LABEL: define {{.*}} @_Z3run5state(
// BODY-NVPTX-SAME: byval(%struct.state) {{.*}}[[STATE:%[0-9]+]])
// BODY-NVPTX-NOT: store {{.*}}, ptr [[STATE]],
// BODY-NVPTX-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[STATE]],
// BODY-NVPTX: {{^}}}

// On AMDGPU a kernel takes an aggregate by reference into its constant
// argument segment, which the frontend copies into a local object once the
// aggregate's address is taken, and a device function takes a small one in
// registers, which the frontend stores to a local object. The primal pointer
// is then a pointer into that local copy, and the locals and their shadows are
// in private memory, reaching the derivative through casts to generic
// pointers.
// The functor's primal is the closure's local copy, its shadow a zero closure.
// a holds u[i], its shadow is zero; b is read after the call, its shadow is
// seeded.
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z13forall_kernelI7closureEvT_i(
// IR-AMDGPU-SAME: ptr addrspace(4) {{.*}}byref(%struct.closure) {{.*}}%[[CLOSURE:[0-9]+]], i32
// IR-AMDGPU: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// IR-AMDGPU: %[[CS:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// IR-AMDGPU: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) {{.*}}%[[COPY]], ptr addrspace(4) {{.*}}%[[CLOSURE]], i64 16, i1 false)
// IR-AMDGPU-DAG: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// IR-AMDGPU-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-AMDGPU-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR-AMDGPU: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR-AMDGPU: store double %[[AV]], ptr addrspace(5) %[[A]],
// IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[CS]], i8 0, i64 16, i1 false)
// IR-AMDGPU: store i64 0, ptr addrspace(5) %[[DA]],
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// IR-AMDGPU: %[[CSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[CS]] to ptr
// IR-AMDGPU: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// IR-AMDGPU: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0(ptr %[[COPYG]], ptr %[[CSG]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// IR-AMDGPU: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// IR-AMDGPU: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR-AMDGPU: store double %[[R]], ptr %[[UI]],
// IR-AMDGPU: ret void
// BODY-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z13forall_kernelI7closureEvT_i(
// BODY-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY-AMDGPU: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY-AMDGPU-NOT: store {{.*}}, ptr %[[COPYG]],
// BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[COPYG]],
// BODY-AMDGPU: {{^}}}

// The member eight bytes into the parameter's local copy, and as far into its
// shadow; a and b as above.
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z6k_data6data_ti(
// IR-AMDGPU-SAME: ptr addrspace(4) {{.*}}byref(%struct.data_t) {{.*}}%[[DATA:[0-9]+]], i32
// IR-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// IR-AMDGPU: %[[DS:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// IR-AMDGPU: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) {{.*}}%[[COPY]], ptr addrspace(4) {{.*}}%[[DATA]], i64 16, i1 false)
// IR-AMDGPU-DAG: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// IR-AMDGPU-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-AMDGPU-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR-AMDGPU: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR-AMDGPU: store double %[[AV]], ptr addrspace(5) %[[A]],
// IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR-AMDGPU: %[[QF:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[COPYG]], i64 8
// IR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[DS]], i8 0, i64 16, i1 false)
// IR-AMDGPU: store i64 0, ptr addrspace(5) %[[DA]],
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// IR-AMDGPU: %[[DSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DS]] to ptr
// IR-AMDGPU: %[[QFS:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DSG]], i64 8
// IR-AMDGPU: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// IR-AMDGPU: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0(ptr {{(nonnull )?}}%[[QF]], ptr {{(nonnull )?}}%[[QFS]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// IR-AMDGPU: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// IR-AMDGPU: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR-AMDGPU: store double %[[R]], ptr %[[UI]],
// IR-AMDGPU: ret void
// BODY-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z6k_data6data_ti(
// BODY-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY-AMDGPU: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// BODY-AMDGPU-NOT: store {{.*}}, ptr %[[COPYG]],
// BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[COPYG]],
// BODY-AMDGPU: {{^}}}

// The struct arrives in registers and is stored to a local object; the call
// writes into it and it is read after the call, so its shadow is seeded.
// IR-AMDGPU-LABEL: define {{.*}}double @_Z3run5state(
// IR-AMDGPU-SAME: [2 x double] %[[V:[0-9]+]])
// IR-AMDGPU: %[[ST:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// IR-AMDGPU: %[[SS:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// IR-AMDGPU: %[[V0:[0-9]+]] = extractvalue [2 x double] %[[V]], 0
// IR-AMDGPU: store double %[[V0]], ptr addrspace(5) %[[ST]],
// IR-AMDGPU: %[[V1:[0-9]+]] = extractvalue [2 x double] %[[V]], 1
// IR-AMDGPU: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[ST]], i32 8
// IR-AMDGPU: store double %[[V1]], ptr addrspace(5) %[[ST1]],
// IR-AMDGPU: %[[STG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[ST]] to ptr
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[SS]],
// IR-AMDGPU: %[[SS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[SS]], i32 8
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[SS1]],
// IR-AMDGPU-NOT: store
// IR-AMDGPU: %[[SSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[SS]] to ptr
// IR-AMDGPU-NOT: store
// IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z7advanceP5state.poseidon._Z3run5state.0(ptr %[[STG]], ptr %[[SSG]])
// IR-AMDGPU-NOT: store
// IR-AMDGPU: %[[R:[0-9]+]] = load double, ptr addrspace(5) %[[ST1]],
// IR-AMDGPU: ret double %[[R]]
// BODY-AMDGPU-LABEL: define {{.*}}double @_Z3run5state(
// BODY-AMDGPU: %[[ST:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// BODY-AMDGPU: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[ST]], i32 8
// BODY-AMDGPU: store double %{{[0-9]+}}, ptr addrspace(5) %[[ST1]],
// BODY-AMDGPU: %[[STG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[ST]] to ptr
// BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[ST]],
// BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[ST1]],
// BODY-AMDGPU-NOT: store {{.*}}, ptr %[[STG]],
// BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[STG]],
// BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr addrspace(5) {{.*}}%[[ST]],
// BODY-AMDGPU: {{^}}}

// USE-NOT: cannot be a Poseidon site
// USE-DAG: per_call_by_value.cu:[[@LINE-251]]:35: warning: Poseidon: no profile at {{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0.fpprofile:
// USE-DAG: per_call_by_value.cu:[[@LINE-237]]:13: warning: Poseidon: no profile at {{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0.fpprofile:
// USE-DAG: per_call_by_value.cu:[[@LINE-219]]:41: warning: Poseidon: no profile at {{.*}}_Z7advanceP5state.poseidon._Z3run5state.0.fpprofile:
// USE-NOT: cannot be a Poseidon site
// USE: {{^}}3 warnings generated when compiling for {{.+}}.{{$}}
