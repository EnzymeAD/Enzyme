// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: FileCheck --check-prefix=BODY %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s
// RUN: not %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/rc -DREFUSED %s -o /dev/null 2>&1 | FileCheck --check-prefixes=REFUSED,REFUSED-GEN %s
// RUN: not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/ru -DREFUSED %s -o /dev/null 2>&1 | FileCheck --check-prefixes=REFUSED,REFUSED-USE %s

// A pointer into no object the pass knows the extent of (device memory, a
// reference parameter of the caller) gets a shadow when the call reaches a
// constant range of memory through it: a local object in the stack's address
// space covering the range, zero, with 1.0 wherever the call writes. Pointers
// at a constant distance from one base share one shadow; pointers of distinct
// bases get their own, and the profiling run stops at a call where the memory
// such a shadow stands for overlaps another argument's and the call writes into
// one of the two. A pointer the call indexes with a value, or passes where the
// pass cannot follow it, makes the call no site, which is an error that names
// every such argument.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#define DEVICE __device__
#define GLOBAL __global__

DEVICE void qf(const double *x, double *y) {
  y[0] = x[0] * x[0] + x[1];
  y[1] = x[1] * x[1];
}

// Input and output in device memory: distinct bases.
GLOBAL void k_inout(const double *x, double *y, int n) {
  int i = threadIdx.x;
  if (i < n)
    __poseidon_optimize(qf, x + 2 * i, y + 2 * i);
}

// A local input, the output in device memory.
GLOBAL void k_out(const double *x, double *y, int n) {
  int i = threadIdx.x;
  if (i >= n)
    return;
  double a[2] = {x[2 * i], x[2 * i + 1]};
  __poseidon_optimize(qf, a, y + 2 * i);
}

// In place: output and input in one array, overlapping by a constant
// distance. The two share one shadow, the output's 8 bytes into it, so the
// overlap is mirrored and nothing is checked.
GLOBAL void k_inplace(double *x, int n) {
  int i = threadIdx.x;
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
  int i = blockIdx.x;
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
  int i = threadIdx.x;
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
  int i = threadIdx.x;
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

// REFUSED-DAG: per_call_extent_amdgpu.hip:[[@LINE-20]]:13: error: Poseidon: sum(double const*, double*, int) at call 0 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant. The pass makes the shadow of a pointer into a local object of the caller or an aggregate it takes by value, or else of the constant range of memory the call reaches through the pointer. Copy such an argument into a local object, or pass its shadow in the call: enzyme_dup before the pointer and the shadow after it.
// REFUSED-DAG: per_call_extent_amdgpu.hip:[[@LINE-21]]:13: error: Poseidon: leak(double const*, double*) at call 1 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call passes to sink(double const*), whose body the pass cannot see. The pass makes the shadow of a pointer into a local object of the caller or an aggregate it takes by value, or else of the constant range of memory the call reaches through the pointer. Copy such an argument into a local object, or pass its shadow in the call: enzyme_dup before the pointer and the shadow after it.
// REFUSED-DAG: per_call_extent_amdgpu.hip:[[@LINE-22]]:13: error: Poseidon: scan(double const*, double*, int) at call 2 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant; argument 2 is a pointer the call indexes with a value that is not a constant. The pass makes the shadow of a pointer into a local object of the caller or an aggregate it takes by value, or else of the constant range of memory the call reaches through the pointer. Copy such an argument into a local object, or pass its shadow in the call: enzyme_dup before the pointer and the shadow after it.
// REFUSED-DAG: per_call_extent_amdgpu.hip:[[@LINE-23]]:13: error: Poseidon: climb(double const*, double*) at call 3 in k_refused(double const*, double*, int) cannot be a Poseidon site: argument 1 is a pointer the call passes to chain(double const*, int), which passes it on to itself at another offset. The pass makes the shadow of a pointer into a local object of the caller or an aggregate it takes by value, or else of the constant range of memory the call reaches through the pointer. Copy such an argument into a local object, or pass its shadow in the call: enzyme_dup before the pointer and the shadow after it.
// REFUSED-GEN: {{^}}4 errors generated when compiling for {{.+}}.{{$}}
// REFUSED-USE: {{^}}6 warnings and 4 errors generated when compiling for {{.+}}.{{$}}

// USE-NOT: {{qf|body|lean}}({{.*}}) at call 0 in {{.*}} cannot be a Poseidon site
// USE-DAG: per_call_extent_amdgpu.hip:[[@LINE-131]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0.fpprofile:
// USE-DAG: per_call_extent_amdgpu.hip:[[@LINE-125]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0.fpprofile:
// USE-DAG: per_call_extent_amdgpu.hip:[[@LINE-115]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0.fpprofile:
// USE-DAG: per_call_extent_amdgpu.hip:[[@LINE-106]]:13: warning: Poseidon: no profile at {{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0.fpprofile:
// USE-DAG: per_call_extent_amdgpu.hip:[[@LINE-91]]:39: warning: Poseidon: no profile at {{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0.fpprofile:
// USE-DAG: per_call_extent_amdgpu.hip:[[@LINE-67]]:13: warning: Poseidon: no profile at {{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0.fpprofile:
// USE: {{^}}6 warnings generated when compiling for {{.+}}.{{$}}

// Each overlap check names its call.
// IR: @[[INOUT:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [77 x i8] c"qf(double const*, double*) at call 0 in k_inout(double const*, double*, int)\00"
// IR: @[[OUT:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [75 x i8] c"qf(double const*, double*) at call 0 in k_out(double const*, double*, int)\00"
// IR: @[[SELECT:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [78 x i8] c"qf(double const*, double*) at call 0 in k_select(double const*, double*, int)\00"
// IR: @[[REFSTR:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [92 x i8] c"body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int)\00"
// IR: @[[ALIGN:poseidon\.str(\.[0-9]+)?]] = private unnamed_addr addrspace(1) constant [72 x i8] c"lean(pair const*, double*) at call 0 in k_align(double const*, double*)\00"

// Shadows of 16 bytes: zero for x + 2i, 1.0 in both doubles for y + 2i. The
// two may overlap, y is written: the run stops if [x + 2i, x + 2i + 16) and
// [y + 2i, y + 2i + 16) meet. Nothing but the call's results is stored through
// the parameters.
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_inoutPKdPdi(
// IR-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], ptr addrspace(1) {{.*}}%[[YARG:[0-9]+]], i32
// IR-DAG: %[[XS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-DAG: %[[YS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-DAG: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[YARG]] to ptr
// IR-DAG: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[YG]], i64 %[[I2:[0-9]+]]
// IR-DAG: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XARG]] to ptr
// IR-DAG: %[[X:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[XG]], i64 %[[I2]]
// IR-DAG: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[XS]], i8 0, i64 16, i1 false)
// IR-DAG: store double 1.000000e+00, ptr addrspace(5) %[[YS]], align 8
// IR-DAG: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[YS]], i32 8
// IR-DAG: store double 1.000000e+00, ptr addrspace(5) %[[YS1]], align 8
// IR-NOT: store
// IR: %[[XA:[0-9]+]] = ptrtoint ptr %[[X]] to i64
// IR-NEXT: %[[XE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[XA]], 16
// IR-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 16
// IR-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[YE]], %[[XA]]
// IR-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[XE]], %[[YA]]
// IR: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR: {{^}}[[STOP]]:
// IR-NEXT: call void @poseidonOverlappingArgumentsCUDA(ptr addrspacecast (ptr addrspace(1) @[[INOUT]] to ptr), i32 1, i32 2)
// IR-NEXT: unreachable
// IR: {{^}}[[GO]]:
// IR-NEXT: %[[XSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[XS]] to ptr
// IR-NEXT: %[[YSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[YS]] to ptr
// IR-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0(ptr %[[X]], ptr %[[XSG]], ptr %[[Y]], ptr %[[YSG]])
// IR-NOT: store
// IR: ret void
// The only stores in the kernel are the two seeds.
// BODY-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_inoutPKdPdi(
// BODY-NOT: store
// BODY: store double 1.000000e+00, ptr addrspace(5) %[[YS:[0-9]+]], align 8
// BODY-NEXT: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[YS]], i32 8
// BODY-NEXT: store double 1.000000e+00, ptr addrspace(5) %[[YS1]], align 8
// BODY-NOT: store
// BODY: {{^}}}

// A local object against a pointer of no object the pass knows: checked as
// well.
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z5k_outPKdPdi(
// IR-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], ptr addrspace(1) {{.*}}%[[YARG:[0-9]+]], i32
// IR: %[[YS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR: %[[A:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR: %[[AS:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[YARG]] to ptr
// IR: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR: %[[X0:[0-9]+]] = getelementptr inbounds{{.*}}, ptr addrspace(1) %[[XARG]], i64 %[[I2:[0-9]+]]
// IR: store double %{{[0-9]+}}, ptr addrspace(5) %[[A]],
// IR: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[YG]], i64 %[[I2]]
// IR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[AS]], i8 0, i64 16, i1 false)
// IR: store double 1.000000e+00, ptr addrspace(5) %[[YS]],
// IR: %[[YS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[YS]], i32 8
// IR: store double 1.000000e+00, ptr addrspace(5) %[[YS1]],
// IR: %[[AA:[0-9]+]] = ptrtoint ptr %[[AG]] to i64
// IR-NEXT: %[[AE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[AA]], 16
// IR-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 16
// IR-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[YE]], %[[AA]]
// IR-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[AE]], %[[YA]]
// IR: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR: {{^}}[[STOP]]:
// IR-NEXT: call void @poseidonOverlappingArgumentsCUDA(ptr addrspacecast (ptr addrspace(1) @[[OUT]] to ptr), i32 1, i32 2)
// IR-NEXT: unreachable
// IR: {{^}}[[GO]]:
// IR-NEXT: %[[ASG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[AS]] to ptr
// IR-NEXT: %[[YSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[YS]] to ptr
// IR-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0(ptr %[[AG]], ptr %[[ASG]], ptr %[[Y]], ptr %[[YSG]])
// IR: ret void

// x + 2i and x + 2i + 1 share one shadow of 24 bytes, the output's 8 bytes into
// it: zero where only the input reaches, 1.0 where the output does.
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z9k_inplacePdi(
// IR-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], i32
// IR: %[[S:[0-9]+]] = alloca [24 x i8], align 8, addrspace(5)
// IR-NOT: poseidonOverlappingArgumentsCUDA
// IR: %[[XI1:[0-9]+]] = getelementptr inbounds{{.*}}, ptr addrspace(1) %[[XARG]], i64 %[[I2:[0-9]+]]
// IR: %[[XI1G:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XI1]] to ptr
// IR: %[[XJ:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XI1G]], i64 8
// IR: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XARG]] to ptr
// IR: %[[XI:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[XG]], i64 %[[I2]]
// IR: store i64 0, ptr addrspace(5) %[[S]],
// IR: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[S]], i32 8
// IR: store double 1.000000e+00, ptr addrspace(5) %[[Y]],
// IR: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[S]], i32 16
// IR: store double 1.000000e+00, ptr addrspace(5) %[[Y1]],
// IR: %[[SG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[S]] to ptr
// IR: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[Y]] to ptr
// IR-NOT: poseidonOverlappingArgumentsCUDA
// IR: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0(ptr %[[XI]], ptr %[[SG]], ptr {{(nonnull )?}}%[[XJ]], ptr %[[YG]])
// IR-NOT: poseidonOverlappingArgumentsCUDA
// IR: ret void
// BODY-LABEL: define {{.*}}amdgpu_kernel void @_Z9k_inplacePdi(
// BODY-NOT: poseidonOverlappingArgumentsCUDA
// BODY: {{^}}}

// The input selected between two local objects, against the output a.
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z8k_selectPKdPdi(
// IR-SAME: ptr addrspace(1) {{.*}}, ptr addrspace(1) {{.*}}, i32 {{.*}}[[N:%[0-9]+]])
// IR: %[[INS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR: %[[A:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR: %[[AS:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR: %[[B:[0-9]+]] = alloca [2 x double], align 16, addrspace(5)
// IR: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR: %[[C:[0-9]+]] = icmp {{(samesign )?}}sgt i32 [[N]], 0
// IR-NEXT: %[[IN:[0-9]+]] = select i1 %[[C]], ptr %[[AG]], ptr %[[BG]]
// IR: store double 1.000000e+00, ptr addrspace(5) %[[AS]],
// IR: %[[AS1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[AS]], i32 8
// IR: store double 1.000000e+00, ptr addrspace(5) %[[AS1]],
// IR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[INS]], i8 0, i64 16, i1 false)
// IR: %[[AA:[0-9]+]] = ptrtoint ptr %[[AG]] to i64
// IR-NEXT: %[[AE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[AA]], 16
// IR-NEXT: %[[IA:[0-9]+]] = ptrtoint ptr %[[IN]] to i64
// IR-NEXT: %[[IE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[IA]], 16
// IR-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[IE]], %[[AA]]
// IR-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[AE]], %[[IA]]
// IR: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR: {{^}}[[STOP]]:
// IR-NEXT: call void @poseidonOverlappingArgumentsCUDA(ptr addrspacecast (ptr addrspace(1) @[[SELECT]] to ptr), i32 1, i32 2)
// IR-NEXT: unreachable
// IR: {{^}}[[GO]]:
// IR-NEXT: %[[INSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[INS]] to ptr
// IR-NEXT: %[[ASG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[AS]] to ptr
// IR-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0(ptr %[[IN]], ptr %[[INSG]], ptr %[[AG]], ptr %[[ASG]])
// IR: ret void

// The reference's shadow: 8 bytes, zero, nothing stored through it. The call
// writes b, a local object: the run stops if the reference reaches it.
// IR-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(
// IR-SAME: ptr {{.*}}dereferenceable(8) [[REF:%[0-9]+]], ptr {{.*}}[[U:%[0-9]+]], i32 {{.*}}[[I:%[0-9]+]])
// IR: %[[FS:[0-9]+]] = alloca [8 x i8], align 8, addrspace(5)
// IR: %[[A:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[DA:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[B:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[DB:[0-9]+]] = alloca double, align 8, addrspace(5)
// IR: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// IR: %[[IX:[0-9]+]] = sext i32 [[I]] to i64
// IR: %[[UI:[0-9]+]] = getelementptr inbounds{{.*}}, ptr [[U]], i64 %[[IX]]
// IR: %[[AV:[0-9]+]] = load double, ptr %[[UI]],
// IR: store double %[[AV]], ptr addrspace(5) %[[A]],
// IR: store i64 0, ptr addrspace(5) %[[DA]],
// IR: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// IR: store i64 0, ptr addrspace(5) %[[FS]],
// IR-NOT: store {{.*}}, ptr [[REF]]
// IR: %[[BA:[0-9]+]] = ptrtoint ptr %[[BG]] to i64
// IR-NEXT: %[[BE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[BA]], 8
// IR-NEXT: %[[RA:[0-9]+]] = ptrtoint ptr [[REF]] to i64
// IR-NEXT: %[[RE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[RA]], 8
// IR-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[RE]], %[[BA]]
// IR-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[BE]], %[[RA]]
// IR: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR: {{^}}[[STOP]]:
// IR-NEXT: call void @poseidonOverlappingArgumentsCUDA(ptr addrspacecast (ptr addrspace(1) @[[REFSTR]] to ptr), i32 1, i32 3)
// IR-NEXT: unreachable
// IR: {{^}}[[GO]]:
// IR-NEXT: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// IR-NEXT: %[[FSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[FS]] to ptr
// IR-NEXT: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// IR-NEXT: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// IR-NOT: store {{.*}}, ptr [[REF]]
// IR: call {{.*}} @diffe{{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0(ptr [[REF]], ptr %[[FSG]], ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// IR: %[[BV:[0-9]+]] = load double, ptr addrspace(5) %[[B]],
// IR: %[[R:[0-9]+]] = fmul {{.*}}double %[[BV]], 2.000000e+00
// IR: store double %[[R]], ptr %[[UI]],
// IR: ret void
// BODY-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(
// BODY-SAME: dereferenceable(8) [[REF:%[0-9]+]],
// BODY-NOT: store {{.*}}, ptr [[REF]],
// BODY-NOT: call {{.*}}@llvm.mem{{.*}}(ptr {{.*}}[[REF]],
// BODY: {{^}}}

// The pointer's range runs from 8 bytes before it to 16 after; its shadow
// starts 8 bytes into a 16-byte aligned local object of 32 bytes.
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_alignPKdPd(
// IR-SAME: ptr addrspace(1) {{.*}}%[[XARG:[0-9]+]], ptr addrspace(1) {{.*}}%[[YARG:[0-9]+]])
// IR: %[[PS:[0-9]+]] = alloca [32 x i8], align 16, addrspace(5)
// IR: %[[YS:[0-9]+]] = alloca [8 x i8], align 8, addrspace(5)
// IR: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[YARG]] to ptr
// IR: %[[X4:[0-9]+]] = getelementptr inbounds{{.*}}, ptr addrspace(1) %[[XARG]], i64
// IR: %[[X4G:[0-9]+]] = addrspacecast ptr addrspace(1) %[[X4]] to ptr
// IR: %[[PP:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X4G]], i64 16
// IR: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}}, ptr %[[YG]], i64
// IR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[PS]], i8 0, i64 32, i1 false)
// IR: store double 1.000000e+00, ptr addrspace(5) %[[YS]],
// IR: %[[PA:[0-9]+]] = ptrtoint ptr %[[PP]] to i64
// IR-NEXT: %[[PLO:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[PA]], -8
// IR-NEXT: %[[PHI:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[PA]], 16
// IR-NEXT: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR-NEXT: %[[YE:[0-9]+]] = add {{(nuw )?(nsw )?}}i64 %[[YA]], 8
// IR-DAG: %[[OV:[0-9]+]] = and i1 %[[#OVL:]], %[[#OVR:]]
// IR-DAG: %[[#min(OVL,OVR)]] = icmp {{(samesign )?}}ult i64 %[[PLO]], %[[YE]]
// IR-DAG: %[[#max(OVL,OVR)]] = icmp {{(samesign )?}}ugt i64 %[[PHI]], %[[YA]]
// IR: br i1 %[[OV]], label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR: {{^}}[[STOP]]:
// IR-NEXT: call void @poseidonOverlappingArgumentsCUDA(ptr addrspacecast (ptr addrspace(1) @[[ALIGN]] to ptr), i32 1, i32 2)
// IR-NEXT: unreachable
// IR: {{^}}[[GO]]:
// IR-NEXT: %[[P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[PS]], i32 16
// IR-NEXT: %[[PG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[P]] to ptr
// IR-NEXT: %[[YSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[YS]] to ptr
// IR-NEXT: call {{.*}} @diffe{{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0(ptr {{(nonnull )?}}%[[PP]], ptr %[[PG]], ptr %[[Y]], ptr %[[YSG]])
// IR: ret void
