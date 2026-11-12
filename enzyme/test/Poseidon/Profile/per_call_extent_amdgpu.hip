// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen.ll 2>&1 | FileCheck --check-prefixes=GEN,REFUSED %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o /dev/null 2>&1 | FileCheck --check-prefixes=USE,REFUSED %s

// per_call_extent_nvptx.cu for HIP: the shadow of a pointer into device
// memory, or of a reference parameter, is a local object in the stack's
// address space that covers what the call reaches through the pointer.

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

GLOBAL void k_refused(const double *x, double *y, int n) {
  int i = threadIdx.x;
  double s;
  __poseidon_optimize(sum, x + 8 * i, &s, n);
  y[i] = s;
  __poseidon_optimize(leak, x + 8 * i, &s);
  y[i + 1] = s;
  __poseidon_optimize(scan, x + 8 * i, y + 8 * i, n);
}

// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_inout(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_out(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_inplace(double*, int): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_select(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon-profgen] lean(pair const*, double*) at call 0 in k_align(double const*, double*): 2 shadow object(s), output(s) 1

// REFUSED-DAG: warning: {{.*}}Poseidon: sum(double const*, double*, int) at call 0 in k_refused(double const*, double*, int) is not a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant. The pass makes
// REFUSED-DAG: warning: {{.*}}Poseidon: leak(double const*, double*) at call 1 in k_refused(double const*, double*, int) is not a Poseidon site: argument 1 is a pointer the call passes to sink(double const*), whose body the pass cannot see. The pass makes
// REFUSED-DAG: warning: {{.*}}Poseidon: scan(double const*, double*, int) at call 2 in k_refused(double const*, double*, int) is not a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant; argument 2 is a pointer the call indexes with a value that is not a constant. The pass makes

// USE-NOT: {{qf|body|lean}}({{.*}}) at call 0 in {{.*}} is not a Poseidon site
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z9k_inplacePdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged

// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_inoutPKdPdi(
// IR-DAG: %[[XS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-DAG: %[[YS:[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// IR-DAG: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[XS]], i8 0, i64 16,
// IR-DAG: store double 1.000000e+00, ptr addrspace(5) %[[YS]], align 8
// IR-DAG: %[[YS1:[0-9]+]] = getelementptr inbounds nuw i8, ptr addrspace(5) %[[YS]], i32 8
// IR-DAG: store double 1.000000e+00, ptr addrspace(5) %[[YS1]], align 8
// IR: br i1 %{{[0-9]+}}, label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR: [[STOP]]:
// IR-NEXT: call void @poseidonOverlappingArgumentsCUDA(ptr addrspacecast (ptr addrspace(1) @{{.*}} to ptr), i32 1, i32 2)
// IR-NEXT: unreachable
// IR: [[GO]]:
// IR-NEXT: %[[XSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[XS]] to ptr
// IR-NEXT: %[[YSG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[YS]] to ptr
// IR-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0(ptr %{{[0-9]+}}, ptr %[[XSG]], ptr %{{[0-9]+}}, ptr %[[YSG]])
// IR-NOT: store
// IR: ret void

// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z5k_outPKdPdi(
// IR: call void @poseidonOverlappingArgumentsCUDA({{.*}}, i32 1, i32 2)
// IR: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0(
// IR: ret void

// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z9k_inplacePdi(
// IR: alloca [24 x i8], align 8, addrspace(5)
// IR-NOT: poseidonOverlappingArgumentsCUDA
// IR: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0(
// IR: ret void

// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z8k_selectPKdPdi(
// IR: call void @poseidonOverlappingArgumentsCUDA({{.*}}, i32 1, i32 2)
// IR: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0(
// IR: ret void

// IR-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(ptr {{.*}}dereferenceable(8) %0,
// IR: alloca [8 x i8], align 8, addrspace(5)
// IR-NOT: store {{.*}}, ptr %0
// IR: call {{.*}} @diffe{{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0(ptr %0,
// IR: ret void

// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z7k_alignPKdPd(
// IR: %[[PS:[0-9]+]] = alloca [32 x i8], align 16, addrspace(5)
// IR: %[[P:[0-9]+]] = getelementptr inbounds nuw i8, ptr addrspace(5) %[[PS]], i32 16
// IR: %[[PG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[P]] to ptr
// IR: call {{.*}} @diffe{{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0(ptr {{.*}}, ptr %[[PG]],
// IR: ret void
