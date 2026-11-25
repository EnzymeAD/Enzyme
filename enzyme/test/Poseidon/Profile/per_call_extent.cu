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

// A pointer into no object the pass knows the extent of (device memory, a
// reference parameter of the caller) gets a shadow when the call reaches a
// constant range of memory through it: a local object covering the range (in
// private memory on AMDGPU), zero, with 1.0 wherever the call writes. Pointers
// at a constant distance from one base share one shadow; pointers of distinct
// bases get their own, and the profiling run stops at a call where the memory
// such a shadow stands for overlaps another argument's and the call writes into
// one of the two. A pointer the call indexes with a value, or passes where the
// pass cannot follow it, makes the call no site, which is an error that names
// every such argument.

#include "poseidon/poseidon.h"

#define DEVICE __attribute__((device))
#define GLOBAL __attribute__((global))

#ifdef __NVPTX__
#define THREAD_X __nvvm_read_ptx_sreg_tid_x()
#define BLOCK_ID_X __nvvm_read_ptx_sreg_ctaid_x()
#else
#define THREAD_X __builtin_amdgcn_workitem_id_x()
#define BLOCK_ID_X __builtin_amdgcn_workgroup_id_x()
#endif

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

// In place: output and input in one array, overlapping by a constant
// distance. The two share one shadow, the output's 8 bytes into it, so the
// overlap is mirrored and nothing is checked.
GLOBAL void k_inplace(double *x, int n) {
  int i = THREAD_X;
  int j = 2 * i + 1;
  if (i < n)
    __poseidon_optimize(qf, x + 2 * i, x + j);
}

// A pointer into a local object through a select is a pointer of no object
// the pass knows: the run stops if it reaches what the call writes through
// another argument.
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

// An access that claims 16-byte alignment at the pointer and one 8 bytes
// before it: the shadow starts 8 bytes into a 16-byte aligned local object,
// so that the pointer's shadow is as aligned as the pointer.
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
// Refused: a loop over a length the call is given, a pointer passed to a
// function the pass cannot see, and both at once.
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

// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_inout(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_out(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_inplace(double*, int): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] qf(double const*, double*) at call 0 in k_select(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon] lean(pair const*, double*) at call 0 in k_align(double const*, double*): 2 shadow object(s), output(s) 1

// REFUSED-DAG: per_call_extent.cu:[[@LINE-20]]:13: error: Poseidon: sum(double const*, double*, int) at call 0 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-DAG: per_call_extent.cu:[[@LINE-21]]:13: error: Poseidon: leak(double const*, double*) at call 1 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call passes to sink(double const*), whose body the pass cannot see; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-DAG: per_call_extent.cu:[[@LINE-22]]:13: error: Poseidon: scan(double const*, double*, int) at call 2 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant; argument 2 is a pointer the call indexes with a value that is not a constant; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-DAG: per_call_extent.cu:[[@LINE-23]]:13: error: Poseidon: climb(double const*, double*) at call 3 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call passes to chain(double const*, int), which passes it on to itself at another offset; copy the argument into a local object, or pass its shadow in the call (enzyme_dup before the pointer, the shadow after it)
// REFUSED-GEN: {{^}}4 errors generated when compiling for {{.+}}.{{$}}
// REFUSED-USE: {{^}}4 errors generated when compiling for {{.+}}.{{$}}

// USE-NOT: {{qf|body|lean}}({{.*}}) at call 0 in {{.*}} cannot be a Poseidon site
// USE-DAG: per_call_extent.cu:[[@LINE-131]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0.fpprofile:
// USE-DAG: per_call_extent.cu:[[@LINE-125]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0.fpprofile:
// USE-DAG: per_call_extent.cu:[[@LINE-115]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0.fpprofile:
// USE-DAG: per_call_extent.cu:[[@LINE-106]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0.fpprofile:
// USE-DAG: per_call_extent.cu:[[@LINE-91]]:39: warning: Poseidon: no profile at {{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0.fpprofile:
// USE-DAG: per_call_extent.cu:[[@LINE-67]]:13: warning: Poseidon: no profile at {{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0.fpprofile:
// USE: {{^}}6 warnings generated when compiling for {{.+}}.{{$}}

// Each overlap check names its call.
// IR-NVPTX: @[[INOUT:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [77 x i8] c"qf(double const*, double*) at call 0 in k_inout(double const*, double*, int)\00"
// IR-NVPTX: @[[OUT:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [75 x i8] c"qf(double const*, double*) at call 0 in k_out(double const*, double*, int)\00"
// IR-NVPTX: @[[SELECT:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [78 x i8] c"qf(double const*, double*) at call 0 in k_select(double const*, double*, int)\00"
// IR-NVPTX: @[[REFSTR:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [92 x i8] c"body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int)\00"
// IR-NVPTX: @[[ALIGN:poseidon\.str\.?[0-9]*]] = private unnamed_addr addrspace(1) constant [72 x i8] c"lean(pair const*, double*) at call 0 in k_align(double const*, double*)\00"

// Shadows of 16 bytes: zero for x + 2i, 1.0 in both doubles for y + 2i. The
// two may overlap, y is written: the run stops if [x + 2i, x + 2i + 16) and
// [y + 2i, y + 2i + 16) meet. Nothing but the call's results is stored through
// the parameters.
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

// A local object against a pointer of no object the pass knows: checked as
// well.
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

// x + 2i and x + 2i + 1 share one shadow of 24 bytes, the output's 8 bytes into
// it: zero where only the input reaches, 1.0 where the output does.
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

// The reference's shadow: 8 bytes, zero, nothing stored through it. The call
// writes b, a local object: the run stops if the reference reaches it.
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

// The pointer's range runs from 8 bytes before it to 16 after; its shadow
// starts 8 bytes into a 16-byte aligned local object of 32 bytes.
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

// The same on AMDGPU, where the shadows and the locals are in private memory
// and reach the derivative and the overlap checks through casts to generic
// pointers.
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
