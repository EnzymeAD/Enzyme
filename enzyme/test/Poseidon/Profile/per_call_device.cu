// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{nvptx} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc
// DEFINE: %{gen} = %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate
// DEFINE: %{use} = %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ %{nvptx} %{gen} -mllvm -poseidon-cache=%t/nvptx-c -mllvm -poseidon-print %s -o %t/nvptx.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=IR-NVPTX %s < %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=BODY-NVPTX %s < %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} %{use} -mllvm -poseidon-cache=%t/nvptx-u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s %}
// RUN: %if nvptx-device-profiler %{ not %{nvptx} %{gen} -mllvm -poseidon-cache=%t/nvptx-rc -DREFUSED %s -o /dev/null 2>&1 | FileCheck --check-prefixes=REFUSED,REFUSED-GEN %s %}
// RUN: %if nvptx-device-profiler %{ not %{nvptx} %{use} -mllvm -poseidon-cache=%t/nvptx-ru -DREFUSED %s -o /dev/null 2>&1 | FileCheck --check-prefixes=REFUSED,REFUSED-USE %s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %{gen} -mllvm -poseidon-cache=%t/amdgpu-c -mllvm -poseidon-print %s -o %t/amdgpu.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=IR-AMDGPU %s < %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=BODY-AMDGPU %s < %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %{use} -mllvm -poseidon-cache=%t/amdgpu-u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %{amdgpu} %{gen} -mllvm -poseidon-cache=%t/amdgpu-rc -DREFUSED %s -o /dev/null 2>&1 | FileCheck --check-prefixes=REFUSED,REFUSED-GEN %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %{amdgpu} %{use} -mllvm -poseidon-cache=%t/amdgpu-ru -DREFUSED %s -o /dev/null 2>&1 | FileCheck --check-prefixes=REFUSED,REFUSED-USE %s %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} %{gen} -DBY_VALUE -mllvm -poseidon-cache=%t/nvptx-vc -mllvm -poseidon-print %s -o %t/nvptx-v.ll 2>&1 | FileCheck --check-prefix=VAL-GEN --implicit-check-not=warning: %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=VAL-IR-NVPTX %s < %t/nvptx-v.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=VAL-BODY-NVPTX %s < %t/nvptx-v.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} %{use} -DBY_VALUE -mllvm -poseidon-cache=%t/nvptx-vu %s -o /dev/null 2>&1 | FileCheck --check-prefix=VAL-USE %s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %{gen} -DBY_VALUE -mllvm -poseidon-cache=%t/amdgpu-vc -mllvm -poseidon-print %s -o %t/amdgpu-v.ll 2>&1 | FileCheck --check-prefix=VAL-GEN --implicit-check-not=warning: %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=VAL-IR-AMDGPU %s < %t/amdgpu-v.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=VAL-BODY-AMDGPU %s < %t/amdgpu-v.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %{use} -DBY_VALUE -mllvm -poseidon-cache=%t/amdgpu-vu %s -o /dev/null 2>&1 | FileCheck --check-prefix=VAL-USE %s %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} -O1 %{use} -DLOCALS %s -o %t/nvptx-l.ll 2>&1 | FileCheck --check-prefix=LOC %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=LOC-IR,LOC-IR-NVPTX %s < %t/nvptx-l.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} -O1 %{gen} -DLOCALS -mllvm -poseidon-cache=%t/nvptx-lc -mllvm -poseidon-print %s -o %t/nvptx-lgen.ll 2>&1 | FileCheck --check-prefix=LOC-GEN %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=LOC-GENIR-NVPTX %s < %t/nvptx-lgen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} -O1 %{use} -DLOCALS %s -o %t/amdgpu-l.ll 2>&1 | FileCheck --check-prefix=LOC %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=LOC-IR,LOC-IR-AMDGPU %s < %t/amdgpu-l.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} -O1 %{gen} -DLOCALS -mllvm -poseidon-cache=%t/amdgpu-lc -mllvm -poseidon-print %s -o %t/amdgpu-lgen.ll 2>&1 | FileCheck --check-prefix=LOC-GEN %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=LOC-GENIR-AMDGPU %s < %t/amdgpu-lgen.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} -O1 %{gen} -DSEEDS -mllvm -poseidon-print -mllvm -poseidon-cache=%t/seeds %s -o %t/seeds.ll 2>&1 | FileCheck --check-prefix=SEED %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=SEED-IR %s < %t/seeds.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=SEED-SHADOW,SEED-BOTH %s < %t/seeds.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=SEED-SHADOW,SEED-STEP %s < %t/seeds.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} -O1 %{gen} -DFUNCTOR -mllvm -poseidon-print -mllvm -poseidon-cache=%t/functor %s -o %t/functor.ll 2>&1 | FileCheck --check-prefix=FUNCTOR %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=FUNCTOR-IR %s < %t/functor.ll %}

// A pointer into an object of unknown extent gets a local shadow (private on
// AMDGPU) over the constant range the call reaches, zero with 1.0 where it
// writes; pointers at a constant distance share one; the run stops where such
// memory overlaps a written argument. A pointer indexed by a value or passed
// where the pass cannot follow makes the call no site, an error naming it.
// A pointer into an aggregate taken by value (a kernel's closure or struct
// parameter, a device function's struct; BY_VALUE) has a known extent: its
// shadow is a local of its type, and nothing is stored through the parameter.
// At -O1 (LOCALS): a local's shadow is zero with the seed where the call
// writes, two device pointers a constant distance apart share one shadow, and
// an object holding pointers is passed as a constant.
// On NVPTX, an object the call writes is seeded only if something after the
// call (or the call again) may read it (SEEDS); the address of an empty
// functor, poison when lowered, is passed as a constant (FUNCTOR).

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

#ifdef SEEDS
__attribute__((device)) void both(const double *x, double *p, double *t) {
  p[0] = x[0] * x[0];
  t[0] = 2.0 * x[0] * x[1];
}

__attribute__((device)) void step(double *s, double *y) {
  y[0] = s[0] * s[0];
  s[0] = s[0] + 0.5 * y[0];
}

__attribute__((global)) void k(double *x, int n) {
  for (int i = 0; i < n; ++i) {
    double in[2] = {x[2 * i], x[2 * i + 1]}, p, t;
    __poseidon_optimize(both, in, &p, &t);
    x[2 * i] = 3.0 * t;
  }
  double s = x[0];
  for (int i = 0; i < n; ++i) {
    double y;
    __poseidon_optimize(step, &s, &y);
    x[i + 1] = 3.0 * y;
  }
}
#elif defined(FUNCTOR)
struct square {
  __attribute__((device)) void operator()(const double &x, double &y) const {
    y = x * x + 1.0;
  }
};

template <typename f_t>
__attribute__((device)) void apply(const f_t *f, const double *x, double *y) {
  (*f)(*x, *y);
}

__attribute__((global)) void k(double *x) {
  const square f{};
  double a = x[0], b;
  __poseidon_optimize(apply<square>, &f, &a, &b);
  x[1] = 2.0 * b;
}
#elif defined(LOCALS)
struct holder {
  const double *c;
};

__attribute__((device)) double sq(double x) { return x * x + 1.0; }

__attribute__((device)) void scale(const double *x, double *y) {
  y[0] = x[0] * x[0] + x[1];
  y[1] = x[1] * x[1];
}

__attribute__((device)) void weigh(const holder &h, const double *x,
                                   double *y) {
  y[0] = h.c[0] * x[0] * x[0];
}

__attribute__((global)) void k(double *x) {
  x[0] = __poseidon_optimize(sq, x[1]);
  double a[2] = {x[2], x[3]}, b[2];
  __poseidon_optimize(scale, a, b);
  x[4] = b[0] + b[1];
  __poseidon_optimize(scale, x + 5, x + 7);
  holder h{x + 9};
  double c, d = x[10];
  __poseidon_optimize(weigh, h, &d, &c);
  x[11] = c;
}
#elif defined(BY_VALUE)
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

// A member at a non-zero offset of a struct parameter (__grid_constant__).
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

// A by-value aggregate the call writes and is read after: seeded.
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
#else
DEVICE void qf(const double *x, double *y) {
  y[0] = x[0] * x[0] + x[1];
  y[1] = x[1] * x[1];
}

// Input and output in device memory: distinct bases.
GLOBAL void k_inout(const double *x, double *y, int n) {
  int i = THREAD_X;
  if (i < n)
    __poseidon_optimize(qf, x + 2 * i, y + 2 * i);
}

// A local input, the output in device memory.
GLOBAL void k_out(const double *x, double *y, int n) {
  int i = THREAD_X;
  if (i >= n)
    return;
  double a[2] = {x[2 * i], x[2 * i + 1]};
  __poseidon_optimize(qf, a, y + 2 * i);
}

// In place, a constant distance apart: one shadow, nothing checked.
GLOBAL void k_inplace(double *x, int n) {
  int i = THREAD_X;
  int j = 2 * i + 1;
  if (i < n)
    __poseidon_optimize(qf, x + 2 * i, x + j);
}

// A pointer into a local object through a select: checked for overlap.
GLOBAL void k_select(const double *x, double *y, int n) {
  double a[2] = {x[0], x[1]}, b[2] = {x[2], x[3]};
  const double *in = n > 0 ? a : b;
  __poseidon_optimize(qf, in, a);
  y[0] = a[0] + b[1];
}

// A read-only reference parameter of a device function that is not inlined.
struct qfunc {
  double c;
};

DEVICE void body(const qfunc *f, const double *x, double *y) {
  *y = f->c * *x * *x + 1.0;
}

__attribute__((noinline)) DEVICE void run_ref(const qfunc &f, double *u,
                                              int i) {
  double a = u[i], b;
  __poseidon_optimize(body, &f, &a, &b);
  u[i] = 2.0 * b;
}

GLOBAL void k_ref(double *u, double c, int n) {
  int i = BLOCK_ID_X;
  const qfunc f{c};
  if (i < n)
    run_ref(f, u, i);
}

// 16-byte aligned at the pointer, accessed 8 bytes before it: the shadow is
// as aligned as the pointer.
struct alignas(16) pair {
  double a, b;
};

DEVICE void lean(const pair *p, double *y) {
  *y = p->a * reinterpret_cast<const double *>(p)[-1] + p->b;
}

GLOBAL void k_align(const double *x, double *y) {
  int i = THREAD_X;
  __poseidon_optimize(lean, reinterpret_cast<const pair *>(x + 4 * i + 2),
                      y + i);
}

#ifdef REFUSED
// Refused: a loop over a given length, an opaque callee, and both.
DEVICE void sum(const double *x, double *y, int n) {
  double s = 0.0;
  for (int k = 0; k < n; ++k)
    s += x[k] * x[k];
  *y = s;
}

DEVICE void sink(const double *x);

DEVICE void leak(const double *x, double *y) {
  sink(x);
  *y = x[0] * x[0];
}

DEVICE void scan(const double *x, double *y, int n) {
  for (int k = 0; k < n; ++k)
    y[k] = x[k] * x[k];
}

// A helper that passes the pointer on to itself one element further.
DEVICE __attribute__((noinline)) double chain(const double *x, int n) {
  return n > 0 ? x[0] + chain(x + 1, n - 1) : 0.0;
}
DEVICE void climb(const double *x, double *y) { *y = 2.0 * chain(x, 3); }

GLOBAL void k_refused(const double *x, double *y, int n) {
  int i = THREAD_X;
  double s;
  __poseidon_optimize(sum, x + 8 * i, &s, n);
  y[i] = s;
  __poseidon_optimize(leak, x + 8 * i, &s);
  y[i + 1] = s;
  __poseidon_optimize(scan, x + 8 * i, y + 8 * i, n);
  __poseidon_optimize(climb, x + 8 * i, &s);
  y[i + 2] = s;
}
#endif
#endif

// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_inout(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_out(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_inplace(double*, int): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_select(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon] lean(pair const*, double*) at call 0 in k_align(double const*, double*): 2 shadow object(s), output(s) 1
// REFUSED-DAG: per_call_device.cu:[[@LINE-20]]:13: error: Poseidon: sum(double const*, double*, int) at call 0 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-DAG: per_call_device.cu:[[@LINE-21]]:13: error: Poseidon: leak(double const*, double*) at call 1 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call passes to sink(double const*), whose body the pass cannot see; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-DAG: per_call_device.cu:[[@LINE-22]]:13: error: Poseidon: scan(double const*, double*, int) at call 2 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant; argument 2 is a pointer the call indexes with a value that is not a constant; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-DAG: per_call_device.cu:[[@LINE-23]]:13: error: Poseidon: climb(double const*, double*) at call 3 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call passes to chain(double const*, int), which passes it on to itself at another offset; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-GEN: {{^}}4 errors generated when compiling for {{.+}}.{{$}}
// REFUSED-USE: {{^}}4 errors generated when compiling for {{.+}}.{{$}}
// USE-NOT: {{qf|body|lean}}({{.*}}) at call 0 in {{.*}} cannot be a Poseidon site
// USE-DAG: per_call_device.cu:[[@LINE-124]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0.fpprofile:
// USE-DAG: per_call_device.cu:[[@LINE-118]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0.fpprofile:
// USE-DAG: per_call_device.cu:[[@LINE-110]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0.fpprofile:
// USE-DAG: per_call_device.cu:[[@LINE-103]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0.fpprofile:
// USE-DAG: per_call_device.cu:[[@LINE-88]]:39: warning: Poseidon: no profile at {{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0.fpprofile:
// USE-DAG: per_call_device.cu:[[@LINE-65]]:13: warning: Poseidon: no profile at {{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0.fpprofile:
// USE: {{^}}6 warnings generated when compiling for {{.+}}.{{$}}

// Each overlap check names its call.
// IR-NVPTX: @[[INOUT:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [77 x i8] c"qf(double const*, double*) at call 0 in k_inout(double const*, double*, int)\00"
// IR-NVPTX: @[[OUT:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [75 x i8] c"qf(double const*, double*) at call 0 in k_out(double const*, double*, int)\00"
// IR-NVPTX: @[[SELECT:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [78 x i8] c"qf(double const*, double*) at call 0 in k_select(double const*, double*, int)\00"
// IR-NVPTX: @[[REFSTR:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [92 x i8] c"body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int)\00"
// IR-NVPTX: @[[ALIGN:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [72 x i8] c"lean(pair const*, double*) at call 0 in k_align(double const*, double*)\00"

// 16-byte shadows, zero for x + 2i and seeded for y + 2i; the run stops if the
// two ranges meet; nothing else is stored through the parameters.
// IR-NVPTX-LABEL: define {{.*}} @_Z7k_inoutPKdPdi(
// IR-NVPTX-SAME: ptr {{.*}}[[XARG:%[0-9]+]], ptr {{.*}}[[YARG:%[0-9]+]], i32
// IR-NVPTX-DAG: %[[XS:[0-9]+]] = alloca [16 x i8], align 8
// IR-NVPTX-DAG: %[[YS:[0-9]+]] = alloca [16 x i8], align 8
// IR-NVPTX-DAG: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[YARG]], i64 %[[I2:[0-9]+]]
// IR-NVPTX-DAG: %[[X:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[XARG]], i64 %[[I2]]
// IR-NVPTX-DAG: call void @llvm.memset.p0.i64(ptr {{.*}}%[[XS]], i8 0, i64 16, i1 false)
// IR-NVPTX-DAG: store double 1.000000e+00, ptr %[[YS]], align 8
// IR-NVPTX-DAG: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[YS]], i64 8
// IR-NVPTX-DAG: store double 1.000000e+00, ptr %[[YS1]], align 8
// IR-NVPTX-NOT: store
// IR-NVPTX: %[[XA:[0-9]+]] = ptrtoint ptr %[[X]] to i64
// IR-NVPTX-NEXT: %[[XE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[XA]], 16
// IR-NVPTX-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-NVPTX-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 16
// IR-NVPTX-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-NVPTX-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[YE]], %[[XA]]
// IR-NVPTX-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[XE]], %[[YA]]
// IR-NVPTX: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-NVPTX: {{^}}[[STOP]]:
// IR-NVPTX-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[INOUT]] to ptr), i32 1, i32 2)
// IR-NVPTX-NEXT: unreachable
// IR-NVPTX: {{^}}[[GO]]:
// IR-NVPTX-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0(ptr %[[X]], ptr %[[XS]], ptr %[[Y]], ptr %[[YS]])
// IR-NVPTX-NOT: store
// IR-NVPTX: ret void
// The only stores in the kernel are the two seeds.
// BODY-NVPTX-LABEL: define {{.*}} @_Z7k_inoutPKdPdi(
// BODY-NVPTX-NOT: store
// BODY-NVPTX: store double 1.000000e+00, ptr %[[YS:[0-9]+]], align 8
// BODY-NVPTX-NEXT: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[YS]], i64 8
// BODY-NVPTX-NEXT: store double 1.000000e+00, ptr %[[YS1]], align 8
// BODY-NVPTX-NOT: store
// BODY-NVPTX: {{^}}}

// A local object against a pointer of unknown extent: checked as well.
// IR-NVPTX-LABEL: define {{.*}} @_Z5k_outPKdPdi(
// IR-NVPTX-SAME: ptr {{.*}}[[XARG:%[0-9]+]], ptr {{.*}}[[YARG:%[0-9]+]], i32
// IR-NVPTX: %[[YS:[0-9]+]] = alloca [16 x i8], align 8
// IR-NVPTX: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// IR-NVPTX: %[[AS:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// IR-NVPTX: %[[X0:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[XARG]], i64 %[[I2:[0-9]+]]
// IR-NVPTX: store double %{{[0-9]+}}, ptr %[[A]],
// IR-NVPTX: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[YARG]], i64 %[[I2]]
// IR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[AS]], i8 0, i64 16, i1 false)
// IR-NVPTX: store double 1.000000e+00, ptr %[[YS]],
// IR-NVPTX: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[YS]], i64 8
// IR-NVPTX: store double 1.000000e+00, ptr %[[YS1]],
// IR-NVPTX: %[[AA:[0-9]+]] = ptrtoint ptr %[[A]] to i64
// IR-NVPTX-NEXT: %[[AE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[AA]], 16
// IR-NVPTX-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-NVPTX-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 16
// IR-NVPTX-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-NVPTX-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[YE]], %[[AA]]
// IR-NVPTX-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[AE]], %[[YA]]
// IR-NVPTX: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-NVPTX: {{^}}[[STOP]]:
// IR-NVPTX-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[OUT]] to ptr), i32 1, i32 2)
// IR-NVPTX-NEXT: unreachable
// IR-NVPTX: {{^}}[[GO]]:
// IR-NVPTX-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0(ptr %[[A]], ptr %[[AS]], ptr %[[Y]], ptr %[[YS]])
// IR-NVPTX: ret void

// One 24-byte shadow, the output's 8 bytes into it.
// IR-NVPTX-LABEL: define {{.*}} @_Z9k_inplacePdi(
// IR-NVPTX-SAME: ptr {{.*}}[[XARG:%[0-9]+]], i32
// IR-NVPTX: %[[S:[0-9]+]] = alloca [24 x i8], align 8
// IR-NVPTX-NOT: __poseidon_device_overlapping_arguments
// IR-NVPTX: %[[XI:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[XARG]], i64 %{{[0-9]+}}
// IR-NVPTX: %[[XJ:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XI]], i64 8
// IR-NVPTX: store i64 0, ptr %[[S]],
// IR-NVPTX: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[S]], i64 8
// IR-NVPTX: store double 1.000000e+00, ptr %[[Y]],
// IR-NVPTX: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[S]], i64 16
// IR-NVPTX: store double 1.000000e+00, ptr %[[Y1]],
// IR-NVPTX-NOT: __poseidon_device_overlapping_arguments
// IR-NVPTX: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0(ptr %[[XI]], ptr %[[S]], ptr {{(nonnull )?}}%[[XJ]], ptr %[[Y]])
// IR-NVPTX-NOT: __poseidon_device_overlapping_arguments
// IR-NVPTX: ret void
// BODY-NVPTX-LABEL: define {{.*}} @_Z9k_inplacePdi(
// BODY-NVPTX-NOT: __poseidon_device_overlapping_arguments
// BODY-NVPTX: {{^}}}

// The input selected between two local objects, against the output a.
// IR-NVPTX-LABEL: define {{.*}} @_Z8k_selectPKdPdi(
// IR-NVPTX-SAME: ptr {{.*}}, ptr {{.*}}, i32 {{.*}}[[N:%[0-9]+]])
// IR-NVPTX: %[[INS:[0-9]+]] = alloca [16 x i8], align 8
// IR-NVPTX: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// IR-NVPTX: %[[AS:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// IR-NVPTX: %[[B:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// IR-NVPTX: %[[C:[0-9]+]] = icmp {{(samesign )?}}sgt i32 [[N]], 0
// IR-NVPTX-NEXT: %[[IN:[0-9]+]] = select i1 %[[C]], ptr %[[A]], ptr %[[B]]
// IR-NVPTX: store double 1.000000e+00, ptr %[[AS]],
// IR-NVPTX: %[[AS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[AS]], i64 8
// IR-NVPTX: store double 1.000000e+00, ptr %[[AS1]],
// IR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[INS]], i8 0, i64 16, i1 false)
// IR-NVPTX: %[[AA:[0-9]+]] = ptrtoint ptr %[[A]] to i64
// IR-NVPTX-NEXT: %[[AE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[AA]], 16
// IR-NVPTX-NEXT: %[[IA:[0-9]+]] = ptrtoint ptr %[[IN]] to i64
// IR-NVPTX-NEXT: %[[IE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[IA]], 16
// IR-NVPTX-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-NVPTX-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[IE]], %[[AA]]
// IR-NVPTX-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[AE]], %[[IA]]
// IR-NVPTX: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-NVPTX: {{^}}[[STOP]]:
// IR-NVPTX-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[SELECT]] to ptr), i32 1, i32 2)
// IR-NVPTX-NEXT: unreachable
// IR-NVPTX: {{^}}[[GO]]:
// IR-NVPTX-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0(ptr %[[IN]], ptr %[[INS]], ptr %[[A]], ptr %[[AS]])
// IR-NVPTX: ret void

// The reference's 8-byte zero shadow; the run stops if it reaches b.
// IR-NVPTX-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(
// IR-NVPTX-SAME: ptr {{.*}}dereferenceable(8) [[REF:%[0-9]+]], ptr {{.*}}[[U:%[0-9]+]], i32 {{.*}}[[I:%[0-9]+]])
// IR-NVPTX: %[[FS:[0-9]+]] = alloca [8 x i8], align 8
// IR-NVPTX: %[[A:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[DA:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[B:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[DB:[0-9]+]] = alloca double, align 8
// IR-NVPTX: %[[IX:[0-9]+]] = sext i32 [[I]] to i64
// IR-NVPTX: %[[UI:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[U]], i64 %[[IX]]
// IR-NVPTX: %[[AV:[0-9]+]] = load double, ptr %[[UI]],
// IR-NVPTX: store double %[[AV]], ptr %[[A]],
// IR-NVPTX: store i64 0, ptr %[[DA]],
// IR-NVPTX: store double 1.000000e+00, ptr %[[DB]],
// IR-NVPTX: store i64 0, ptr %[[FS]],
// IR-NVPTX-NOT: store {{.*}}, ptr [[REF]]
// IR-NVPTX: %[[BA:[0-9]+]] = ptrtoint ptr %[[B]] to i64
// IR-NVPTX-NEXT: %[[BE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[BA]], 8
// IR-NVPTX-NEXT: %[[RA:[0-9]+]] = ptrtoint ptr [[REF]] to i64
// IR-NVPTX-NEXT: %[[RE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[RA]], 8
// IR-NVPTX-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-NVPTX-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[RE]], %[[BA]]
// IR-NVPTX-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[BE]], %[[RA]]
// IR-NVPTX: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-NVPTX: {{^}}[[STOP]]:
// IR-NVPTX-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[REFSTR]] to ptr), i32 1, i32 3)
// IR-NVPTX-NEXT: unreachable
// IR-NVPTX: {{^}}[[GO]]:
// IR-NVPTX-NEXT: call {{.*}} @diffe{{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0(ptr [[REF]], ptr %[[FS]], ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// IR-NVPTX-NOT: store {{.*}}, ptr [[REF]]
// IR-NVPTX: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// IR-NVPTX: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR-NVPTX: store double %[[R]], ptr %[[UI]],
// IR-NVPTX: ret void
// BODY-NVPTX-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(
// BODY-NVPTX-SAME: dereferenceable(8) [[REF:%[0-9]+]],
// BODY-NVPTX-NOT: store {{.*}}, ptr [[REF]],
// BODY-NVPTX-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[REF]],
// BODY-NVPTX: {{^}}}

// The range is [-8, 16) bytes; the shadow starts 8 bytes into an aligned 32.
// IR-NVPTX-LABEL: define {{.*}} @_Z7k_alignPKdPd(
// IR-NVPTX-SAME: ptr {{.*}}[[XARG:%[0-9]+]], ptr {{.*}}[[YARG:%[0-9]+]])
// IR-NVPTX: %[[PS:[0-9]+]] = alloca [32 x i8], align 16
// IR-NVPTX: %[[YS:[0-9]+]] = alloca [8 x i8], align 8
// IR-NVPTX: %[[X4:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[XARG]], i64 %{{[0-9]+}}
// IR-NVPTX: %[[PP:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X4]], i64 16
// IR-NVPTX: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[YARG]], i64 %{{[0-9]+}}
// IR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[PS]], i8 0, i64 32, i1 false)
// IR-NVPTX: store double 1.000000e+00, ptr %[[YS]],
// IR-NVPTX: %[[PA:[0-9]+]] = ptrtoint ptr %[[PP]] to i64
// IR-NVPTX-NEXT: %[[PLO:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[PA]], -8
// IR-NVPTX-NEXT: %[[PHI:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[PA]], 16
// IR-NVPTX-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-NVPTX-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 8
// IR-NVPTX-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-NVPTX-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ult i64 %[[PLO]], %[[YE]]
// IR-NVPTX-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[PHI]], %[[YA]]
// IR-NVPTX: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-NVPTX: {{^}}[[STOP]]:
// IR-NVPTX-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[ALIGN]] to ptr), i32 1, i32 2)
// IR-NVPTX-NEXT: unreachable
// IR-NVPTX: {{^}}[[GO]]:
// IR-NVPTX-NEXT: %[[P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[PS]], i64 16
// IR-NVPTX-NEXT: call {{.*}} @diffe{{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0(ptr {{(nonnull )?}}%[[PP]], ptr %[[P]], ptr %[[Y]], ptr %[[YS]])
// IR-NVPTX: ret void

// The same on AMDGPU, with private shadows and locals cast to generic.
// IR-AMDGPU: @[[INOUT:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [77 x i8] c"qf(double const*, double*) at call 0 in k_inout(double const*, double*, int)\00"
// IR-AMDGPU: @[[OUT:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [75 x i8] c"qf(double const*, double*) at call 0 in k_out(double const*, double*, int)\00"
// IR-AMDGPU: @[[SELECT:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [78 x i8] c"qf(double const*, double*) at call 0 in k_select(double const*, double*, int)\00"
// IR-AMDGPU: @[[REFSTR:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [92 x i8] c"body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int)\00"
// IR-AMDGPU: @[[ALIGN:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [72 x i8] c"lean(pair const*, double*) at call 0 in k_align(double const*, double*)\00"
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_inoutPKdPdi(
// IR-AMDGPU-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], ptr addrspace(1) {{.*}}%[[YARG:[0-9]+]], i32
// IR-AMDGPU-DAG: %[[XS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-AMDGPU-DAG: %[[YS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-AMDGPU-DAG: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[YARG]] to ptr
// IR-AMDGPU-DAG: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[YG]], i64 %[[I2:[0-9]+]]
// IR-AMDGPU-DAG: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XARG]] to ptr
// IR-AMDGPU-DAG: %[[X:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[XG]], i64 %[[I2]]
// IR-AMDGPU-DAG: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[XS]], i8 0, i64 16, i1 false)
// IR-AMDGPU-DAG: store double 1.000000e+00, ptr addrspace(5) %[[YS]], align 8
// IR-AMDGPU-DAG: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[YS]], i32 8
// IR-AMDGPU-DAG: store double 1.000000e+00, ptr addrspace(5) %[[YS1]], align 8
// IR-AMDGPU-NOT: store
// IR-AMDGPU: %[[XA:[0-9]+]] = ptrtoint ptr %[[X]] to i64
// IR-AMDGPU-NEXT: %[[XE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[XA]], 16
// IR-AMDGPU-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-AMDGPU-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 16
// IR-AMDGPU-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-AMDGPU-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[YE]], %[[XA]]
// IR-AMDGPU-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[XE]], %[[YA]]
// IR-AMDGPU: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-AMDGPU: {{^}}[[STOP]]:
// IR-AMDGPU-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[INOUT]] to ptr), i32 1, i32 2)
// IR-AMDGPU-NEXT: unreachable
// IR-AMDGPU: {{^}}[[GO]]:
// IR-AMDGPU-NEXT: %[[XSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[XS]] to ptr
// IR-AMDGPU-NEXT: %[[YSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[YS]] to ptr
// IR-AMDGPU-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0(ptr %[[X]], ptr %[[XSG]], ptr %[[Y]], ptr %[[YSG]])
// IR-AMDGPU-NOT: store
// IR-AMDGPU: ret void
// BODY-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_inoutPKdPdi(
// BODY-AMDGPU-NOT: store
// BODY-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[YS:[0-9]+]], align 8
// BODY-AMDGPU-NEXT: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[YS]], i32 8
// BODY-AMDGPU-NEXT: store double 1.000000e+00, ptr addrspace(5) %[[YS1]], align 8
// BODY-AMDGPU-NOT: store
// BODY-AMDGPU: {{^}}}
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z5k_outPKdPdi(
// IR-AMDGPU-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], ptr addrspace(1) {{.*}}%[[YARG:[0-9]+]], i32
// IR-AMDGPU: %[[YS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-AMDGPU: %[[A:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR-AMDGPU: %[[AS:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR-AMDGPU: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[YARG]] to ptr
// IR-AMDGPU: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-AMDGPU: %[[X0:[0-9]+]] = getelementptr inbounds{{.*}}, ptr addrspace(1) %[[XARG]], i64 %[[I2:[0-9]+]]
// IR-AMDGPU: store double %{{[0-9]+}}, ptr addrspace(5) %[[A]],
// IR-AMDGPU: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[YG]], i64 %[[I2]]
// IR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[AS]], i8 0, i64 16, i1 false)
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[YS]],
// IR-AMDGPU: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[YS]], i32 8
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[YS1]],
// IR-AMDGPU: %[[AA:[0-9]+]] = ptrtoint ptr %[[AG]] to i64
// IR-AMDGPU-NEXT: %[[AE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[AA]], 16
// IR-AMDGPU-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-AMDGPU-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 16
// IR-AMDGPU-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-AMDGPU-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[YE]], %[[AA]]
// IR-AMDGPU-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[AE]], %[[YA]]
// IR-AMDGPU: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-AMDGPU: {{^}}[[STOP]]:
// IR-AMDGPU-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[OUT]] to ptr), i32 1, i32 2)
// IR-AMDGPU-NEXT: unreachable
// IR-AMDGPU: {{^}}[[GO]]:
// IR-AMDGPU-NEXT: %[[ASG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[AS]] to ptr
// IR-AMDGPU-NEXT: %[[YSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[YS]] to ptr
// IR-AMDGPU-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0(ptr %[[AG]], ptr %[[ASG]], ptr %[[Y]], ptr %[[YSG]])
// IR-AMDGPU: ret void
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z9k_inplacePdi(
// IR-AMDGPU-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], i32
// IR-AMDGPU: %[[S:[0-9]+]] = alloca [24 x i8], align 8, addrspace(5)
// IR-AMDGPU-NOT: __poseidon_device_overlapping_arguments
// IR-AMDGPU: %[[XI1:[0-9]+]] = getelementptr inbounds{{.*}}, ptr addrspace(1) %[[XARG]], i64 %[[I2:[0-9]+]]
// IR-AMDGPU: %[[XI1G:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XI1]] to ptr
// IR-AMDGPU: %[[XJ:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XI1G]], i64 8
// IR-AMDGPU: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XARG]] to ptr
// IR-AMDGPU: %[[XI:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[XG]], i64 %[[I2]]
// IR-AMDGPU: store i64 0, ptr addrspace(5) %[[S]],
// IR-AMDGPU: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[S]], i32 8
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[Y]],
// IR-AMDGPU: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[S]], i32 16
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[Y1]],
// IR-AMDGPU: %[[SG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[S]] to ptr
// IR-AMDGPU: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[Y]] to ptr
// IR-AMDGPU-NOT: __poseidon_device_overlapping_arguments
// IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0(ptr %[[XI]], ptr %[[SG]], ptr {{(nonnull )?}}%[[XJ]], ptr %[[YG]])
// IR-AMDGPU-NOT: __poseidon_device_overlapping_arguments
// IR-AMDGPU: ret void
// BODY-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z9k_inplacePdi(
// BODY-AMDGPU-NOT: __poseidon_device_overlapping_arguments
// BODY-AMDGPU: {{^}}}
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z8k_selectPKdPdi(
// IR-AMDGPU-SAME: ptr addrspace(1) {{.*}}, ptr addrspace(1) {{.*}}, i32 {{.*}}[[N:%[0-9]+]])
// IR-AMDGPU: %[[INS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-AMDGPU: %[[A:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR-AMDGPU: %[[AS:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR-AMDGPU: %[[B:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR-AMDGPU: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-AMDGPU: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR-AMDGPU: %[[C:[0-9]+]] = icmp {{(samesign )?}}sgt i32 [[N]], 0
// IR-AMDGPU-NEXT: %[[IN:[0-9]+]] = select i1 %[[C]], ptr %[[AG]], ptr %[[BG]]
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[AS]],
// IR-AMDGPU: %[[AS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[AS]], i32 8
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[AS1]],
// IR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[INS]], i8 0, i64 16, i1 false)
// IR-AMDGPU: %[[AA:[0-9]+]] = ptrtoint ptr %[[AG]] to i64
// IR-AMDGPU-NEXT: %[[AE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[AA]], 16
// IR-AMDGPU-NEXT: %[[IA:[0-9]+]] = ptrtoint ptr %[[IN]] to i64
// IR-AMDGPU-NEXT: %[[IE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[IA]], 16
// IR-AMDGPU-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-AMDGPU-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[IE]], %[[AA]]
// IR-AMDGPU-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[AE]], %[[IA]]
// IR-AMDGPU: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-AMDGPU: {{^}}[[STOP]]:
// IR-AMDGPU-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[SELECT]] to ptr), i32 1, i32 2)
// IR-AMDGPU-NEXT: unreachable
// IR-AMDGPU: {{^}}[[GO]]:
// IR-AMDGPU-NEXT: %[[INSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[INS]] to ptr
// IR-AMDGPU-NEXT: %[[ASG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[AS]] to ptr
// IR-AMDGPU-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0(ptr %[[IN]], ptr %[[INSG]], ptr %[[AG]], ptr %[[ASG]])
// IR-AMDGPU: ret void
// IR-AMDGPU-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(
// IR-AMDGPU-SAME: ptr {{.*}}dereferenceable(8) [[REF:%[0-9]+]], ptr {{.*}}[[U:%[0-9]+]], i32 {{.*}}[[I:%[0-9]+]])
// IR-AMDGPU: %[[FS:[0-9]+]] = alloca [8 x i8], align 8, addrspace(5)
// IR-AMDGPU: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR-AMDGPU: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR-AMDGPU: %[[IX:[0-9]+]] = sext i32 [[I]] to i64
// IR-AMDGPU: %[[UI:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[U]], i64 %[[IX]]
// IR-AMDGPU: %[[AV:[0-9]+]] = load double, ptr %[[UI]],
// IR-AMDGPU: store double %[[AV]], ptr addrspace(5) %[[A]],
// IR-AMDGPU: store i64 0, ptr addrspace(5) %[[DA]],
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// IR-AMDGPU: store i64 0, ptr addrspace(5) %[[FS]],
// IR-AMDGPU-NOT: store {{.*}}, ptr [[REF]]
// IR-AMDGPU: %[[BA:[0-9]+]] = ptrtoint ptr %[[BG]] to i64
// IR-AMDGPU-NEXT: %[[BE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[BA]], 8
// IR-AMDGPU-NEXT: %[[RA:[0-9]+]] = ptrtoint ptr [[REF]] to i64
// IR-AMDGPU-NEXT: %[[RE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[RA]], 8
// IR-AMDGPU-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-AMDGPU-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[RE]], %[[BA]]
// IR-AMDGPU-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[BE]], %[[RA]]
// IR-AMDGPU: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-AMDGPU: {{^}}[[STOP]]:
// IR-AMDGPU-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[REFSTR]] to ptr), i32 1, i32 3)
// IR-AMDGPU-NEXT: unreachable
// IR-AMDGPU: {{^}}[[GO]]:
// IR-AMDGPU-NEXT: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-AMDGPU-NEXT: %[[FSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[FS]] to ptr
// IR-AMDGPU-NEXT: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// IR-AMDGPU-NEXT: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// IR-AMDGPU-NOT: store {{.*}}, ptr [[REF]]
// IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0(ptr [[REF]], ptr %[[FSG]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// IR-AMDGPU: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// IR-AMDGPU: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR-AMDGPU: store double %[[R]], ptr %[[UI]],
// IR-AMDGPU: ret void
// BODY-AMDGPU-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(
// BODY-AMDGPU-SAME: dereferenceable(8) [[REF:%[0-9]+]],
// BODY-AMDGPU-NOT: store {{.*}}, ptr [[REF]],
// BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[REF]],
// BODY-AMDGPU: {{^}}}
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_alignPKdPd(
// IR-AMDGPU-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], ptr addrspace(1) {{.*}}%[[YARG:[0-9]+]])
// IR-AMDGPU: %[[PS:[0-9]+]] = alloca [32 x i8], align 16, addrspace(5)
// IR-AMDGPU: %[[YS:[0-9]+]] = alloca [8 x i8], align 8, addrspace(5)
// IR-AMDGPU: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[YARG]] to ptr
// IR-AMDGPU: %[[X4:[0-9]+]] = getelementptr inbounds{{.*}}, ptr addrspace(1) %[[XARG]], i64
// IR-AMDGPU: %[[X4G:[0-9]+]] = addrspacecast ptr addrspace(1) %[[X4]] to ptr
// IR-AMDGPU: %[[PP:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X4G]], i64 16
// IR-AMDGPU: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[YG]], i64
// IR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[PS]], i8 0, i64 32, i1 false)
// IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[YS]],
// IR-AMDGPU: %[[PA:[0-9]+]] = ptrtoint ptr %[[PP]] to i64
// IR-AMDGPU-NEXT: %[[PLO:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[PA]], -8
// IR-AMDGPU-NEXT: %[[PHI:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[PA]], 16
// IR-AMDGPU-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-AMDGPU-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 8
// IR-AMDGPU-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-AMDGPU-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ult i64 %[[PLO]], %[[YE]]
// IR-AMDGPU-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[PHI]], %[[YA]]
// IR-AMDGPU: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR-AMDGPU: {{^}}[[STOP]]:
// IR-AMDGPU-NEXT: call void @__poseidon_device_overlapping_arguments(ptr addrspacecast (ptr addrspace(1) @[[ALIGN]] to ptr), i32 1, i32 2)
// IR-AMDGPU-NEXT: unreachable
// IR-AMDGPU: {{^}}[[GO]]:
// IR-AMDGPU-NEXT: %[[P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[PS]], i32 16
// IR-AMDGPU-NEXT: %[[PG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[P]] to ptr
// IR-AMDGPU-NEXT: %[[YSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[YS]] to ptr
// IR-AMDGPU-NEXT: call {{.*}} @diffe{{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0(ptr {{(nonnull )?}}%[[PP]], ptr %[[PG]], ptr %[[Y]], ptr %[[YSG]])
// IR-AMDGPU: ret void

// VAL-GEN-DAG: [poseidon] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in void call_site<qfunc>(qfunc const&, double const*, double*): 3 shadow object(s), output(s) 2
// VAL-GEN-DAG: [poseidon] void body<qfunc>(qfunc const*, double const*, double*) at call 0 in k_data(data_t, int): 3 shadow object(s), output(s) 2
// VAL-GEN-DAG: [poseidon] advance(state*) at call 0 in run(state): 1 shadow object(s), output(s) 0

// The functor's shadow is a zero closure, a's is zero, b's seeded.
// VAL-IR-NVPTX-LABEL: define {{.*}} @_Z13forall_kernelI7closureEvT_i(
// VAL-IR-NVPTX-SAME: ptr {{.*}}byval(%struct.closure) {{.*}}[[CLOSURE:%[0-9]+]], i32
// VAL-IR-NVPTX: %[[CS:[0-9]+]] = alloca %struct.closure, align 8
// VAL-IR-NVPTX: %[[A:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX: %[[DA:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX: %[[B:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX: %[[DB:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// VAL-IR-NVPTX: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// VAL-IR-NVPTX: store double %[[AV]], ptr %[[A]],
// VAL-IR-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// VAL-IR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[CS]], i8 0, i64 16, i1 false)
// VAL-IR-NVPTX: store i64 0, ptr %[[DA]],
// VAL-IR-NVPTX: store double 1.000000e+00, ptr %[[DB]],
// VAL-IR-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// VAL-IR-NVPTX: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0(ptr {{(nonnull )?}}[[CLOSURE]], ptr %[[CS]], ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// VAL-IR-NVPTX: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// VAL-IR-NVPTX: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// VAL-IR-NVPTX: store double %[[R]], ptr %[[UI]],
// VAL-IR-NVPTX: ret void
// VAL-BODY-NVPTX-LABEL: define {{.*}} @_Z13forall_kernelI7closureEvT_i(
// VAL-BODY-NVPTX-SAME: byval(%struct.closure) {{.*}}[[CLOSURE:%[0-9]+]], i32
// VAL-BODY-NVPTX-NOT: store {{.*}}, ptr [[CLOSURE]],
// VAL-BODY-NVPTX-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[CLOSURE]],
// VAL-BODY-NVPTX: {{^}}}

// The member 8 bytes into the parameter and into its shadow.
// VAL-IR-NVPTX-LABEL: define {{.*}} @_Z6k_data6data_ti(
// VAL-IR-NVPTX-SAME: ptr {{.*}}byval(%struct.data_t) {{.*}}"nvvm.grid_constant" [[DATA:%[0-9]+]], i32
// VAL-IR-NVPTX: %[[DS:[0-9]+]] = alloca %struct.data_t, align 8
// VAL-IR-NVPTX: %[[A:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX: %[[DA:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX: %[[B:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX: %[[DB:[0-9]+]] = alloca double, align 8
// VAL-IR-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// VAL-IR-NVPTX: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// VAL-IR-NVPTX: store double %[[AV]], ptr %[[A]],
// VAL-IR-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// VAL-IR-NVPTX: %[[QF:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr [[DATA]], i64 8
// VAL-IR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DS]], i8 0, i64 16, i1 false)
// VAL-IR-NVPTX: store i64 0, ptr %[[DA]],
// VAL-IR-NVPTX: store double 1.000000e+00, ptr %[[DB]],
// VAL-IR-NVPTX: %[[QFS:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DS]], i64 8
// VAL-IR-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// VAL-IR-NVPTX: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0(ptr {{(nonnull )?}}%[[QF]], ptr %[[QFS]], ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// VAL-IR-NVPTX: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// VAL-IR-NVPTX: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// VAL-IR-NVPTX: store double %[[R]], ptr %[[UI]],
// VAL-IR-NVPTX: ret void
// VAL-BODY-NVPTX-LABEL: define {{.*}} @_Z6k_data6data_ti(
// VAL-BODY-NVPTX-SAME: byval(%struct.data_t) {{.*}}[[DATA:%[0-9]+]], i32
// VAL-BODY-NVPTX-NOT: store {{.*}}, ptr [[DATA]],
// VAL-BODY-NVPTX-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[DATA]],
// VAL-BODY-NVPTX: {{^}}}

// Both elements of the shadow are seeded.
// VAL-IR-NVPTX-LABEL: define {{.*}} @_Z3run5state(
// VAL-IR-NVPTX-SAME: ptr {{.*}}byval(%struct.state) {{.*}}[[STATE:%[0-9]+]])
// VAL-IR-NVPTX: %[[SS:[0-9]+]] = alloca %struct.state
// VAL-IR-NVPTX-NOT: store
// VAL-IR-NVPTX: store double 1.000000e+00, ptr %[[SS]],
// VAL-IR-NVPTX: %[[SS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[SS]], i64 8
// VAL-IR-NVPTX: store double 1.000000e+00, ptr %[[SS1]],
// VAL-IR-NVPTX-NOT: store
// VAL-IR-NVPTX: call {{.*}} @diffe{{.*}}_Z7advanceP5state.poseidon._Z3run5state.0(ptr {{(nonnull )?}}[[STATE]], ptr %[[SS]])
// VAL-IR-NVPTX-NOT: store
// VAL-IR-NVPTX: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr [[STATE]], i64 8
// VAL-IR-NVPTX: %[[R:[0-9]+]] = load double, ptr %[[ST1]],
// VAL-IR-NVPTX: ret double %[[R]]
// VAL-BODY-NVPTX-LABEL: define {{.*}} @_Z3run5state(
// VAL-BODY-NVPTX-SAME: byval(%struct.state) {{.*}}[[STATE:%[0-9]+]])
// VAL-BODY-NVPTX-NOT: store {{.*}}, ptr [[STATE]],
// VAL-BODY-NVPTX-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[STATE]],
// VAL-BODY-NVPTX: {{^}}}

// On AMDGPU the frontend copies a kernel's byref aggregate, and stores a
// device function's register aggregate, to a local: the primal points into
// that copy; locals and shadows are private, cast to generic.
// VAL-IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z13forall_kernelI7closureEvT_i(
// VAL-IR-AMDGPU-SAME: ptr addrspace(4) {{.*}}byref(%struct.closure) {{.*}}%[[CLOSURE:[0-9]+]], i32
// VAL-IR-AMDGPU: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[CS:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// VAL-IR-AMDGPU: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) {{.*}}%[[COPY]], ptr addrspace(4) {{.*}}%[[CLOSURE]], i64 16, i1 false)
// VAL-IR-AMDGPU-DAG: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// VAL-IR-AMDGPU-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// VAL-IR-AMDGPU-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// VAL-IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-IR-AMDGPU: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// VAL-IR-AMDGPU: store double %[[AV]], ptr addrspace(5) %[[A]],
// VAL-IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-IR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[CS]], i8 0, i64 16, i1 false)
// VAL-IR-AMDGPU: store i64 0, ptr addrspace(5) %[[DA]],
// VAL-IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// VAL-IR-AMDGPU: %[[CSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[CS]] to ptr
// VAL-IR-AMDGPU: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// VAL-IR-AMDGPU: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// VAL-IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0(ptr %[[COPYG]], ptr %[[CSG]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// VAL-IR-AMDGPU: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// VAL-IR-AMDGPU: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// VAL-IR-AMDGPU: store double %[[R]], ptr %[[UI]],
// VAL-IR-AMDGPU: ret void
// VAL-BODY-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z13forall_kernelI7closureEvT_i(
// VAL-BODY-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.closure, align 8, addrspace(5)
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-BODY-AMDGPU: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr %[[COPYG]],
// VAL-BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[COPYG]],
// VAL-BODY-AMDGPU: {{^}}}

// The member 8 bytes into the local copy and into its shadow.
// VAL-IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z6k_data6data_ti(
// VAL-IR-AMDGPU-SAME: ptr addrspace(4) {{.*}}byref(%struct.data_t) {{.*}}%[[DATA:[0-9]+]], i32
// VAL-IR-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[DS:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// VAL-IR-AMDGPU: call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) {{.*}}%[[COPY]], ptr addrspace(4) {{.*}}%[[DATA]], i64 16, i1 false)
// VAL-IR-AMDGPU-DAG: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// VAL-IR-AMDGPU-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// VAL-IR-AMDGPU-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// VAL-IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-IR-AMDGPU: %[[AV:[0-9]+]] = load double, ptr %[[UI:[0-9]+]],
// VAL-IR-AMDGPU: store double %[[AV]], ptr addrspace(5) %[[A]],
// VAL-IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-IR-AMDGPU: %[[QF:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[COPYG]], i64 8
// VAL-IR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[DS]], i8 0, i64 16, i1 false)
// VAL-IR-AMDGPU: store i64 0, ptr addrspace(5) %[[DA]],
// VAL-IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// VAL-IR-AMDGPU: %[[DSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DS]] to ptr
// VAL-IR-AMDGPU: %[[QFS:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DSG]], i64 8
// VAL-IR-AMDGPU: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// VAL-IR-AMDGPU: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// VAL-IR-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0(ptr {{(nonnull )?}}%[[QF]], ptr {{(nonnull )?}}%[[QFS]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// VAL-IR-AMDGPU: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// VAL-IR-AMDGPU: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// VAL-IR-AMDGPU: store double %[[R]], ptr %[[UI]],
// VAL-IR-AMDGPU: ret void
// VAL-BODY-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z6k_data6data_ti(
// VAL-BODY-AMDGPU: %[[COPY:[0-9]+]] = alloca %struct.data_t, align 8, addrspace(5)
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-BODY-AMDGPU: %[[COPYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[COPY]] to ptr
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[COPY]],
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr %[[COPYG]],
// VAL-BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[COPYG]],
// VAL-BODY-AMDGPU: {{^}}}

// The struct arrives in registers, stored to a local; its shadow is seeded.
// VAL-IR-AMDGPU-LABEL: define {{.*}}double @_Z3run5state(
// VAL-IR-AMDGPU-SAME: [2 x double] %[[V:[0-9]+]])
// VAL-IR-AMDGPU: %[[ST:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[SS:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// VAL-IR-AMDGPU: %[[V0:[0-9]+]] = extractvalue [2 x double] %[[V]], 0
// VAL-IR-AMDGPU: store double %[[V0]], ptr addrspace(5) %[[ST]],
// VAL-IR-AMDGPU: %[[V1:[0-9]+]] = extractvalue [2 x double] %[[V]], 1
// VAL-IR-AMDGPU: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[ST]], i32 8
// VAL-IR-AMDGPU: store double %[[V1]], ptr addrspace(5) %[[ST1]],
// VAL-IR-AMDGPU: %[[STG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[ST]] to ptr
// VAL-IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[SS]],
// VAL-IR-AMDGPU: %[[SS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[SS]], i32 8
// VAL-IR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[SS1]],
// VAL-IR-AMDGPU-NOT: store
// VAL-IR-AMDGPU: %[[SSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[SS]] to ptr
// VAL-IR-AMDGPU-NOT: store
// VAL-IR-AMDGPU: call {{.*}} @diffe{{.*}}_Z7advanceP5state.poseidon._Z3run5state.0(ptr %[[STG]], ptr %[[SSG]])
// VAL-IR-AMDGPU-NOT: store
// VAL-IR-AMDGPU: %[[R:[0-9]+]] = load double, ptr addrspace(5) %[[ST1]],
// VAL-IR-AMDGPU: ret double %[[R]]
// VAL-BODY-AMDGPU-LABEL: define {{.*}}double @_Z3run5state(
// VAL-BODY-AMDGPU: %[[ST:[0-9]+]] = alloca %struct.state, align 8, addrspace(5)
// VAL-BODY-AMDGPU: %[[ST1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[ST]], i32 8
// VAL-BODY-AMDGPU: store double %{{[0-9]+}}, ptr addrspace(5) %[[ST1]],
// VAL-BODY-AMDGPU: %[[STG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[ST]] to ptr
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[ST]],
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr addrspace(5) %[[ST1]],
// VAL-BODY-AMDGPU-NOT: store {{.*}}, ptr %[[STG]],
// VAL-BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}%[[STG]],
// VAL-BODY-AMDGPU-NOT: call {{.*}}@llvm.mem{{.*}}(ptr addrspace(5) {{.*}}%[[ST]],
// VAL-BODY-AMDGPU: {{^}}}
// VAL-USE-NOT: cannot be a Poseidon site
// VAL-USE-DAG: per_call_device.cu:[[@LINE-758]]:35: warning: Poseidon: no profile at {{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z9call_siteI5qfuncEvRKT_PKdPd.0.fpprofile:
// VAL-USE-DAG: per_call_device.cu:[[@LINE-745]]:13: warning: Poseidon: no profile at {{.*}}_Z4bodyI5qfuncEvPKT_PKdPd.poseidon._Z6k_data6data_ti.0.fpprofile:
// VAL-USE-DAG: per_call_device.cu:[[@LINE-728]]:41: warning: Poseidon: no profile at {{.*}}_Z7advanceP5state.poseidon._Z3run5state.0.fpprofile:
// VAL-USE-NOT: cannot be a Poseidon site
// VAL-USE: {{^}}3 warnings generated when compiling for {{.+}}.{{$}}

// LOC-DAG: per_call_device.cu:[[@LINE-802]]:30: warning: Poseidon: no profile at {{.*}}_Z2sqd.poseidon._Z1kPd.0.fpprofile:
// LOC-DAG: per_call_device.cu:[[@LINE-803]]:30: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1.fpprofile:
// LOC-DAG: per_call_device.cu:[[@LINE-804]]:30: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2.fpprofile:
// LOC-DAG: per_call_device.cu:[[@LINE-805]]:30: warning: Poseidon: no profile at {{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3.fpprofile:
// LOC-NOT: cannot be a Poseidon site
// LOC: {{^}}4 warnings generated when compiling for {{.+}}.{{$}}
// LOC-IR-NVPTX-LABEL: define {{.*}}ptx_kernel void @_Z1kPd(
// LOC-IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// LOC-IR-NOT: call
// LOC-IR: ret void
// LOC-IR-NOT: __poseidon_fp_optimize
// LOC-IR-NOT: __poseidon_site
// LOC-GEN-DAG: [poseidon] scale(double const*, double*) at call 1 in k(double*): 2 shadow object(s), output(s) 1
// LOC-GEN-DAG: [poseidon] scale(double const*, double*) at call 2 in k(double*): 1 shadow object(s), output(s) 1
// LOC-GEN-DAG: [poseidon] weigh(holder const&, double const*, double*) at call 3 in k(double*): 2 shadow object(s), output(s) 2
// LOC-GENIR-NVPTX-LABEL: define {{.*}} @_Z1kPd(
// LOC-GENIR-NVPTX-SAME: ptr {{.*}}%[[XP:[0-9]+]])
// LOC-GENIR-NVPTX-DAG: %[[X:[0-9]+]] = alloca [32 x i8], align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[DA:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[B:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[DB:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[H:[0-9]+]] = alloca %struct.holder, align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[C:[0-9]+]] = alloca double, align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[DC:[0-9]+]] = alloca double, align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[D:[0-9]+]] = alloca double, align {{[0-9]+}}
// LOC-GENIR-NVPTX-DAG: %[[DD:[0-9]+]] = alloca double, align {{[0-9]+}}
// A call that passes no pointer has no shadow.
// LOC-GENIR-NVPTX: %[[X1P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 8
// LOC-GENIR-NVPTX: %[[X1:[0-9]+]] = load double, ptr %[[X1P]],
// LOC-GENIR-NVPTX: %[[SQ:[0-9]+]] = {{.*}}call {{.*}}double @diffe{{.*}}_Z2sqd.poseidon._Z1kPd.0(double %[[X1]])
// LOC-GENIR-NVPTX: store double %[[SQ]], ptr %[[XP]],
// a's shadow is zero; b is read after the call, seeded in both elements.
// LOC-GENIR-NVPTX: %[[X2P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 16
// LOC-GENIR-NVPTX: %[[X2:[0-9]+]] = load double, ptr %[[X2P]],
// LOC-GENIR-NVPTX: store double %[[X2]], ptr %[[A]],
// LOC-GENIR-NVPTX: %[[A1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[A]], i64 8
// LOC-GENIR-NVPTX: %[[X3P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 24
// LOC-GENIR-NVPTX: %[[X3:[0-9]+]] = load double, ptr %[[X3P]],
// LOC-GENIR-NVPTX: store double %[[X3]], ptr %[[A1]],
// LOC-GENIR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DA]], i8 0, i64 16, i1 false)
// LOC-GENIR-NVPTX: store double 1.000000e+00, ptr %[[DB]],
// LOC-GENIR-NVPTX: %[[DB1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DB]], i64 8
// LOC-GENIR-NVPTX: store double 1.000000e+00, ptr %[[DB1]],
// LOC-GENIR-NVPTX: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// LOC-GENIR-NVPTX: load double, ptr %[[B]],
// LOC-GENIR-NVPTX: %[[B1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[B]], i64 8
// LOC-GENIR-NVPTX: load double, ptr %[[B1]],
// x + 5 and x + 7 share one 32-byte shadow, the second 16 bytes into it.
// LOC-GENIR-NVPTX: %[[X5:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 40
// LOC-GENIR-NVPTX: %[[X7:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 56
// LOC-GENIR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[X]], i8 0, i64 16, i1 false)
// LOC-GENIR-NVPTX: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X]], i64 16
// LOC-GENIR-NVPTX: store double 1.000000e+00, ptr %[[Y]],
// LOC-GENIR-NVPTX: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X]], i64 24
// LOC-GENIR-NVPTX: store double 1.000000e+00, ptr %[[Y1]],
// LOC-GENIR-NVPTX: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2(ptr {{(nonnull )?}}%[[X5]], ptr %[[X]], ptr {{(nonnull )?}}%[[X7]], ptr %[[Y]])
// The holder has no shadow; d's is zero, c's seeded.
// LOC-GENIR-NVPTX: %[[X9:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 72
// LOC-GENIR-NVPTX: store ptr %[[X9]], ptr %[[H]],
// LOC-GENIR-NVPTX: %[[X10P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 80
// LOC-GENIR-NVPTX: %[[X10:[0-9]+]] = load double, ptr %[[X10P]],
// LOC-GENIR-NVPTX: store double %[[X10]], ptr %[[D]],
// LOC-GENIR-NVPTX: store i64 0, ptr %[[DD]],
// LOC-GENIR-NVPTX: store double 1.000000e+00, ptr %[[DC]],
// LOC-GENIR-NVPTX: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr %[[H]], ptr %[[D]], ptr %[[DD]], ptr %[[C]], ptr %[[DC]])
// LOC-GENIR-NVPTX: %[[CV:[0-9]+]] = load double, ptr %[[C]],
// LOC-GENIR-NVPTX: %[[X11P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 88
// LOC-GENIR-NVPTX: store double %[[CV]], ptr %[[X11P]],
// LOC-GENIR-NVPTX: ret void

// The same on AMDGPU, with private locals and shadows cast to generic.
// LOC-GENIR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// LOC-GENIR-AMDGPU-SAME: ptr addrspace(1) noundef %[[XP:[0-9]+]])
// LOC-GENIR-AMDGPU-DAG: %[[X:[0-9]+]] = alloca [32 x i8], align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[DA:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[B:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[DB:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[H:[0-9]+]] = alloca %struct.holder, align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[C:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[DC:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[D:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[DD:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// LOC-GENIR-AMDGPU-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// LOC-GENIR-AMDGPU-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// LOC-GENIR-AMDGPU-DAG: %[[HG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[H]] to ptr
// LOC-GENIR-AMDGPU-DAG: %[[CG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[C]] to ptr
// LOC-GENIR-AMDGPU-DAG: %[[DG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[D]] to ptr
// LOC-GENIR-AMDGPU-DAG: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XP]] to ptr
// LOC-GENIR-AMDGPU: %[[X1P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 8
// LOC-GENIR-AMDGPU: %[[X1:[0-9]+]] = load double, ptr addrspace(1) %[[X1P]],
// LOC-GENIR-AMDGPU: %[[SQ:[0-9]+]] = {{.*}}call {{.*}}double @diffe{{.*}}_Z2sqd.poseidon._Z1kPd.0(double %[[X1]])
// LOC-GENIR-AMDGPU: store double %[[SQ]], ptr addrspace(1) %[[XP]],
// LOC-GENIR-AMDGPU: %[[X2P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 16
// LOC-GENIR-AMDGPU: %[[X2:[0-9]+]] = load double, ptr addrspace(1) %[[X2P]],
// LOC-GENIR-AMDGPU: store double %[[X2]], ptr addrspace(5) %[[A]],
// LOC-GENIR-AMDGPU: %[[A1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[A]], i32 8
// LOC-GENIR-AMDGPU: %[[X3P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 24
// LOC-GENIR-AMDGPU: %[[X3:[0-9]+]] = load double, ptr addrspace(1) %[[X3P]],
// LOC-GENIR-AMDGPU: store double %[[X3]], ptr addrspace(5) %[[A1]],
// LOC-GENIR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[DA]], i8 0, i64 16, i1 false)
// LOC-GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// LOC-GENIR-AMDGPU: %[[DB1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[DB]], i32 8
// LOC-GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB1]],
// LOC-GENIR-AMDGPU: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// LOC-GENIR-AMDGPU: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// LOC-GENIR-AMDGPU: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// LOC-GENIR-AMDGPU: load double, ptr addrspace(5) %[[B]],
// LOC-GENIR-AMDGPU: %[[B1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[B]], i32 8
// LOC-GENIR-AMDGPU: load double, ptr addrspace(5) %[[B1]],
// LOC-GENIR-AMDGPU: %[[X5:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 40
// LOC-GENIR-AMDGPU: %[[X7:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 56
// LOC-GENIR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[X]], i8 0, i64 16, i1 false)
// LOC-GENIR-AMDGPU: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[X]], i32 16
// LOC-GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[Y]],
// LOC-GENIR-AMDGPU: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[X]], i32 24
// LOC-GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[Y1]],
// LOC-GENIR-AMDGPU: %[[SXG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[X]] to ptr
// LOC-GENIR-AMDGPU: %[[SYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[Y]] to ptr
// LOC-GENIR-AMDGPU: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2(ptr {{(nonnull )?}}%[[X5]], ptr %[[SXG]], ptr {{(nonnull )?}}%[[X7]], ptr %[[SYG]])
// LOC-GENIR-AMDGPU: %[[X9:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 72
// LOC-GENIR-AMDGPU: store ptr %[[X9]], ptr addrspace(5) %[[H]],
// LOC-GENIR-AMDGPU: %[[X10P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 80
// LOC-GENIR-AMDGPU: %[[X10:[0-9]+]] = load double, ptr addrspace(1) %[[X10P]],
// LOC-GENIR-AMDGPU: store double %[[X10]], ptr addrspace(5) %[[D]],
// LOC-GENIR-AMDGPU: store i64 0, ptr addrspace(5) %[[DD]],
// LOC-GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DC]],
// LOC-GENIR-AMDGPU: %[[DDG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DD]] to ptr
// LOC-GENIR-AMDGPU: %[[DCG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DC]] to ptr
// LOC-GENIR-AMDGPU: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr %[[HG]], ptr %[[DG]], ptr %[[DDG]], ptr %[[CG]], ptr %[[DCG]])
// LOC-GENIR-AMDGPU: %[[CV:[0-9]+]] = load double, ptr addrspace(5) %[[C]],
// LOC-GENIR-AMDGPU: %[[X11P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 88
// LOC-GENIR-AMDGPU: store double %[[CV]], ptr addrspace(1) %[[X11P]],
// LOC-GENIR-AMDGPU: ret void

// SEED-DAG: [poseidon] both(double const*, double*, double*) at call 0 in k(double*, int): 3 shadow object(s), output(s) 2, not read after the call: 1
// SEED-DAG: [poseidon] step(double*, double*) at call 1 in k(double*, int): 2 shadow object(s), output(s) 0 1
// The seeds of t, s and y, and no other.
// SEED-IR-LABEL: define {{.*}} @_Z1kPdi(
// SEED-IR-COUNT-3: store double 1.000000e+00
// SEED-IR-NOT: store double 1.000000e+00
// SEED-IR: {{^}}}
// SEED-SHADOW-LABEL: define {{.*}} @_Z1kPdi(
// SEED-SHADOW-SAME: ptr {{.*}}%[[X:[0-9]+]], i32
// SEED-SHADOW: %[[IN:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// SEED-SHADOW: %[[DIN:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// SEED-SHADOW: %[[P:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-SHADOW: %[[DP:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-SHADOW: %[[T:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-SHADOW: %[[DT:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-SHADOW: %[[S:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-SHADOW: %[[DS:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-SHADOW: %[[Y:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-SHADOW: %[[DY:[0-9]+]] = alloca double, align {{[0-9]+}}
// SEED-BOTH: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DIN]], i8 0, i64 16, i1 false)
// SEED-BOTH-NEXT: store i64 0, ptr %[[DP]],
// SEED-BOTH-NEXT: store double 1.000000e+00, ptr %[[DT]],
// SEED-BOTH-NEXT: call {{.*}} @diffe{{.*}}_Z4bothPKdPdS1_.poseidon._Z1kPdi.0(ptr %[[IN]], ptr %[[DIN]], ptr %[[P]], ptr %[[DP]], ptr %[[T]], ptr %[[DT]])
// SEED-BOTH: %[[TV:[0-9]+]] = load double, ptr %[[T]],
// SEED-BOTH: fmul {{.*}}double %[[TV]], 3.000000e+00
// SEED-BOTH: {{^}}}
// SEED-STEP: %[[SV:[0-9]+]] = load double, ptr %[[X]],
// SEED-STEP-NEXT: store double %[[SV]], ptr %[[S]],
// SEED-STEP: store double 1.000000e+00, ptr %[[DS]],
// SEED-STEP-NEXT: store double 1.000000e+00, ptr %[[DY]],
// SEED-STEP-NEXT: call {{.*}} @diffe{{.*}}_Z4stepPdS_.poseidon._Z1kPdi.1(ptr %[[S]], ptr %[[DS]], ptr %[[Y]], ptr %[[DY]])
// SEED-STEP-NOT: load {{.*}}, ptr %[[S]],
// SEED-STEP: %[[YV:[0-9]+]] = load double, ptr %[[Y]],
// SEED-STEP-NOT: load {{.*}}, ptr %[[S]],
// SEED-STEP: fmul {{.*}}double %[[YV]], 3.000000e+00
// SEED-STEP-NOT: load {{.*}}, ptr %[[S]],
// SEED-STEP: {{^}}}
// FUNCTOR-NOT: warning
// FUNCTOR: [poseidon] void apply<square>(square const*, double const*, double*) at call 0 in k(double*): 2 shadow object(s), output(s) 2
// FUNCTOR-NOT: warning
// The call passes a and b with their shadows, zero and seeded, and no functor.
// FUNCTOR-IR-LABEL: define {{.*}} @_Z1kPd(
// FUNCTOR-IR-SAME: ptr {{.*}}%[[X:[0-9]+]])
// FUNCTOR-IR: %[[A:[0-9]+]] = alloca double, align {{[0-9]+}}
// FUNCTOR-IR-NEXT: %[[DA:[0-9]+]] = alloca double, align {{[0-9]+}}
// FUNCTOR-IR-NEXT: %[[B:[0-9]+]] = alloca double, align {{[0-9]+}}
// FUNCTOR-IR-NEXT: %[[DB:[0-9]+]] = alloca double, align {{[0-9]+}}
// FUNCTOR-IR-NOT: alloca
// FUNCTOR-IR: %[[AV:[0-9]+]] = load double, ptr %[[X]],
// FUNCTOR-IR: store double %[[AV]], ptr %[[A]],
// FUNCTOR-IR: store i64 0, ptr %[[DA]],
// FUNCTOR-IR-NEXT: store double 1.000000e+00, ptr %[[DB]],
// FUNCTOR-IR-NEXT: call {{.*}} @diffe{{.*}}_Z5applyI6squareEvPKT_PKdPd.poseidon._Z1kPd.0(ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// FUNCTOR-IR: %[[BV:[0-9]+]] = load double, ptr %[[B]],
// FUNCTOR-IR: fmul {{.*}}double %[[BV]], 2.000000e+00
// FUNCTOR-IR: ret void
