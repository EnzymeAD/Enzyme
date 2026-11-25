// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN --implicit-check-not=warning: %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: FileCheck --check-prefix=BODY %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s

// A pointer into an aggregate the caller takes by value is a pointer into an
// object whose extent the pass knows, as one into a local object is: a
// kernel's closure, a __grid_constant__ parameter, a struct a device function
// takes by value. The shadow is a local object of the aggregate's type, and
// the primal pointer is passed as it is: nothing is stored through the
// parameter. Its seed follows the rule of a local object.

#include "poseidon/poseidon.h"

#define DEVICE __attribute__((device))
#define GLOBAL __attribute__((global))

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
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    body(i);
}

template GLOBAL void forall_kernel<closure>(closure, int);

// A member at a non-zero offset of a __grid_constant__ parameter.
struct data_t {
  double *u;
  qfunc qf;
};

GLOBAL void k_grid(__attribute__((grid_constant)) const data_t data, int n) {
  int i = __nvvm_read_ptx_sreg_ctaid_x();
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

// GEN-DAG: [poseidon-profgen] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in void call_site<qfunc>(qfunc const&, double const*, double*): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon-profgen] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in k_grid(data_t, int): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon-profgen] advance(state*) at call 0 in run(state): 1 shadow object(s), output(s) 0

// The functor's shadow is a zero closure; the primal is the parameter itself.
// a holds u[i], its shadow is zero; b is read after the call, its shadow is
// seeded.
// IR-LABEL: define {{.*}} @_Z13forall_kernelI7closureEvT_i(
// IR-SAME: ptr {{.*}}byval(%struct.closure) {{.*}}[[CLOSURE:%[0-9]+]], i32
// IR: %[[CS:[0-9]+]] = alloca %struct.closure, align 8
// IR: %[[A:[0-9]+]] = alloca double, align 8
// IR: %[[DA:[0-9]+]] = alloca double, align 8
// IR: %[[B:[0-9]+]] = alloca double, align 8
// IR: %[[DB:[0-9]+]] = alloca double, align 8
// IR-NOT: store {{.*}}, ptr [[CLOSURE]],
// IR: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR: store double %[[AV]], ptr %[[A]],
// IR-NOT: store {{.*}}, ptr [[CLOSURE]],
// IR: call void @llvm.memset.p0.i64(ptr {{.*}}%[[CS]], i8 0, i64 16, i1 false)
// IR: store i64 0, ptr %[[DA]],
// IR: store double 1.000000e+00, ptr %[[DB]],
// IR-NOT: store {{.*}}, ptr [[CLOSURE]],
// IR: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0(ptr {{(nonnull )?}}[[CLOSURE]], ptr %[[CS]], ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// IR: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// IR: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR: store double %[[R]], ptr %[[UI]],
// IR: ret void
// BODY-LABEL: define {{.*}} @_Z13forall_kernelI7closureEvT_i(
// BODY-SAME: byval(%struct.closure) {{.*}}[[CLOSURE:%[0-9]+]], i32
// BODY-NOT: store {{.*}}, ptr [[CLOSURE]],
// BODY-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[CLOSURE]],
// BODY: {{^}}}

// The member eight bytes into the parameter, and as far into its shadow; a and
// b as above.
// IR-LABEL: define {{.*}} @_Z6k_grid6data_ti(
// IR-SAME: ptr {{.*}}byval(%struct.data_t) {{.*}}"nvvm.grid_constant" [[DATA:%[0-9]+]], i32
// IR: %[[DS:[0-9]+]] = alloca %struct.data_t, align 8
// IR: %[[A:[0-9]+]] = alloca double, align 8
// IR: %[[DA:[0-9]+]] = alloca double, align 8
// IR: %[[B:[0-9]+]] = alloca double, align 8
// IR: %[[DB:[0-9]+]] = alloca double, align 8
// IR-NOT: store {{.*}}, ptr [[DATA]],
// IR: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// IR: store double %[[AV]], ptr %[[A]],
// IR-NOT: store {{.*}}, ptr [[DATA]],
// IR: %[[QF:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr [[DATA]], i64 8
// IR: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DS]], i8 0, i64 16, i1 false)
// IR: store i64 0, ptr %[[DA]],
// IR: store double 1.000000e+00, ptr %[[DB]],
// IR: %[[QFS:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DS]], i64 8
// IR-NOT: store {{.*}}, ptr [[DATA]],
// IR: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_grid6data_ti.0(ptr {{(nonnull )?}}%[[QF]], ptr %[[QFS]], ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// IR: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// IR: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR: store double %[[R]], ptr %[[UI]],
// IR: ret void
// BODY-LABEL: define {{.*}} @_Z6k_grid6data_ti(
// BODY-SAME: byval(%struct.data_t) {{.*}}[[DATA:%[0-9]+]], i32
// BODY-NOT: store {{.*}}, ptr [[DATA]],
// BODY-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[DATA]],
// BODY: {{^}}}

// The call writes the parameter's second element, read after the call: both
// elements of the shadow are seeded, and nothing is stored through the
// parameter.
// IR-LABEL: define {{.*}} @_Z3run5state(
// IR-SAME: ptr {{.*}}byval(%struct.state) {{.*}}[[STATE:%[0-9]+]])
// IR: %[[SS:[0-9]+]] = alloca %struct.state
// IR-NOT: store
// IR: store double 1.000000e+00, ptr %[[SS]],
// IR: %[[SS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[SS]], i64 8
// IR: store double 1.000000e+00, ptr %[[SS1]],
// IR-NOT: store
// IR: call {{.*}} @diffe{{.*}}_Z7advanceP5state.poseidon._Z3run5state.0(ptr {{(nonnull )?}}[[STATE]], ptr %[[SS]])
// IR-NOT: store
// IR: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr [[STATE]], i64 8
// IR: %[[R:[0-9]+]] = load double, ptr %[[ST1]],
// IR: ret double %[[R]]
// BODY-LABEL: define {{.*}} @_Z3run5state(
// BODY-SAME: byval(%struct.state) {{.*}}[[STATE:%[0-9]+]])
// BODY-NOT: store {{.*}}, ptr [[STATE]],
// BODY-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[STATE]],
// BODY: {{^}}}

// USE-NOT: cannot be a Poseidon site
// USE-DAG: per_call_by_value_nvptx.cu:[[@LINE-128]]:35: warning: Poseidon: no profile at {{.*}}Inputs/other_sites/preprocess__Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0.fpprofile:
// USE-DAG: per_call_by_value_nvptx.cu:[[@LINE-115]]:13: warning: Poseidon: no profile at {{.*}}Inputs/other_sites/preprocess__Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_grid6data_ti.0.fpprofile:
// USE-DAG: per_call_by_value_nvptx.cu:[[@LINE-97]]:41: warning: Poseidon: no profile at {{.*}}Inputs/other_sites/preprocess__Z7advanceP5state.poseidon._Z3run5state.0.fpprofile:
// USE-NOT: cannot be a Poseidon site
// USE: {{^}}3 warnings generated when compiling for {{.+}}.{{$}}
