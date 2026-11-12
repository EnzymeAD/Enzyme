// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen.ll 2>&1 | FileCheck --check-prefixes=GEN,REFUSED %s
// RUN: FileCheck --check-prefix=IR %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o /dev/null 2>&1 | FileCheck --check-prefixes=USE,REFUSED %s

// A pointer into no object the pass knows the extent of (into device memory,
// or a reference parameter of the caller) has a shadow when the call reaches
// a constant range of memory through it: a local object that covers the
// range, zero, with 1.0 wherever the call writes, since the memory outlives
// the call. Two such pointers at a constant distance from one base share one
// shadow; pointers of distinct bases get shadows of their own, and the
// profiling run stops at a call where the memory such a shadow stands for
// overlaps another argument's and the call writes into one of the two. A
// pointer the call indexes with a value, or passes where the pass cannot
// follow it, leaves the call as written, and the warning names every such
// argument.

#include "poseidon/poseidon.h"

#define DEVICE __attribute__((device))
#define GLOBAL __attribute__((global))

DEVICE void qf(const double *x, double *y) {
  y[0] = x[0] * x[0] + x[1];
  y[1] = x[1] * x[1];
}

// Input and output in device memory: distinct bases.
GLOBAL void k_inout(const double *x, double *y, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    __poseidon_optimize(qf, x + 2 * i, y + 2 * i);
}

// A local input, the output in device memory.
GLOBAL void k_out(const double *x, double *y, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i >= n)
    return;
  double a[2] = {x[2 * i], x[2 * i + 1]};
  __poseidon_optimize(qf, a, y + 2 * i);
}

// In place: output and input in one array, overlapping by a constant
// distance. The two share one shadow, the output's 8 bytes into it, so the
// overlap is mirrored and nothing is checked.
GLOBAL void k_inplace(double *x, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
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
  int i = __nvvm_read_ptx_sreg_ctaid_x();
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
  int i = __nvvm_read_ptx_sreg_tid_x();
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

// A helper that passes the pointer on to itself one element further.
DEVICE __attribute__((noinline)) double chain(const double *x, int n) {
  return n > 0 ? x[0] + chain(x + 1, n - 1) : 0.0;
}
DEVICE void climb(const double *x, double *y) { *y = 2.0 * chain(x, 3); }

GLOBAL void k_refused(const double *x, double *y, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  double s;
  __poseidon_optimize(sum, x + 8 * i, &s, n);
  y[i] = s;
  __poseidon_optimize(leak, x + 8 * i, &s);
  y[i + 1] = s;
  __poseidon_optimize(scan, x + 8 * i, y + 8 * i, n);
  __poseidon_optimize(climb, x + 8 * i, &s);
  y[i + 2] = s;
}

// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_inout(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_out(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_inplace(double*, int): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] qf(double const*, double*) at call 0 in k_select(double const*, double*, int): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] body(qfunc const*, double const*, double*) at call 0 in run_ref(qfunc const&, double*, int): 3 shadow object(s), output(s) 2
// GEN-DAG: [poseidon-profgen] lean(pair const*, double*) at call 0 in k_align(double const*, double*): 2 shadow object(s), output(s) 1

// REFUSED-DAG: warning: {{.*}}Poseidon: sum(double const*, double*, int) at call 0 in k_refused(double const*, double*, int) is not a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant. The pass makes the shadow of a pointer into a local object of the caller or an aggregate it takes by value, or else of the constant range of memory the call reaches through the pointer. Copy such an argument into a local object, or write the call with __poseidon_fp_optimize and shadows of your own. The call runs as written.
// REFUSED-DAG: warning: {{.*}}Poseidon: leak(double const*, double*) at call 1 in k_refused(double const*, double*, int) is not a Poseidon site: argument 1 is a pointer the call passes to sink(double const*), whose body the pass cannot see. The pass makes
// REFUSED-DAG: warning: {{.*}}Poseidon: scan(double const*, double*, int) at call 2 in k_refused(double const*, double*, int) is not a Poseidon site: argument 1 is a pointer the call indexes with a value that is not a constant; argument 2 is a pointer the call indexes with a value that is not a constant. The pass makes
// REFUSED-DAG: warning: {{.*}}Poseidon: climb(double const*, double*) at call 3 in k_refused(double const*, double*, int) is not a Poseidon site: argument 1 is a pointer the call passes to chain(double const*, int), which passes it on to itself at another offset. The pass makes

// USE-NOT: {{qf|body|lean}}({{.*}}) at call 0 in {{.*}} is not a Poseidon site
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z9k_inplacePdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// USE-DAG: [poseidon] _Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged

// Shadows of 16 bytes: zero for x + 2i, 1.0 in both doubles for y + 2i. The
// two may overlap, y is written: the run stops if they do. Nothing but the
// call's results is stored through the parameters.
// IR-LABEL: define {{.*}} @_Z7k_inoutPKdPdi(ptr {{.*}}%0, ptr {{.*}}%1, i32
// IR-DAG: %[[XS:[0-9]+]] = alloca [16 x i8], align 8
// IR-DAG: %[[YS:[0-9]+]] = alloca [16 x i8], align 8
// IR-DAG: %[[Y:[0-9]+]] = getelementptr inbounds nuw [8 x i8], ptr %1,
// IR-DAG: %[[X:[0-9]+]] = getelementptr inbounds nuw [8 x i8], ptr %0,
// IR-DAG: call void @llvm.memset.p0.i64(ptr {{.*}}%[[XS]], i8 0, i64 16,
// IR-DAG: store double 1.000000e+00, ptr %[[YS]], align 8
// IR-DAG: %[[YS1:[0-9]+]] = getelementptr inbounds nuw i8, ptr %[[YS]], i64 8
// IR-DAG: store double 1.000000e+00, ptr %[[YS1]], align 8
// IR-DAG: %[[XA:[0-9]+]] = ptrtoint ptr %[[X]] to i64
// IR-DAG: %[[YA:[0-9]+]] = ptrtoint ptr %[[Y]] to i64
// IR: br i1 %{{[0-9]+}}, label %[[STOP:[0-9]+]], label %[[GO:[0-9]+]]
// IR: [[STOP]]:
// IR-NEXT: call void @poseidonOverlappingArgumentsCUDA(ptr addrspacecast (ptr addrspace(1) @{{.*}} to ptr), i32 1, i32 2)
// IR-NEXT: unreachable
// IR: [[GO]]:
// IR-NEXT: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z7k_inoutPKdPdi.0(ptr %[[X]], ptr %[[XS]], ptr %[[Y]], ptr %[[YS]])
// IR-NOT: store
// IR: ret void

// A local object against a pointer of no object the pass knows: checked as
// well.
// IR-LABEL: define {{.*}} @_Z5k_outPKdPdi(
// IR: call void @poseidonOverlappingArgumentsCUDA({{.*}}, i32 1, i32 2)
// IR: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z5k_outPKdPdi.0(
// IR: ret void

// IR-LABEL: define {{.*}} @_Z9k_inplacePdi(
// IR: %[[S:[0-9]+]] = alloca [24 x i8], align 8
// IR-NOT: poseidonOverlappingArgumentsCUDA
// IR: %[[Y:[0-9]+]] = getelementptr inbounds nuw i8, ptr %[[S]], i64 8
// IR-NOT: poseidonOverlappingArgumentsCUDA
// IR: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z9k_inplacePdi.0(ptr %{{[0-9]+}}, ptr %[[S]], ptr {{.*}}, ptr %[[Y]])
// IR: ret void

// IR-LABEL: define {{.*}} @_Z8k_selectPKdPdi(
// IR: %[[IN:[0-9]+]] = select i1 %{{[0-9]+}}, ptr %[[A:[0-9]+]], ptr %{{[0-9]+}}
// IR: call void @poseidonOverlappingArgumentsCUDA({{.*}}, i32 1, i32 2)
// IR: call {{.*}} @diffe{{.*}}_Z2qfPKdPd.poseidon._Z8k_selectPKdPdi.0(ptr %[[IN]], ptr %{{[0-9]+}}, ptr %[[A]],
// IR: ret void

// The reference's shadow: 8 bytes, zero, nothing stored through it.
// IR-LABEL: define {{.*}} @_Z7run_refRK5qfuncPdi(ptr {{.*}}dereferenceable(8) %0,
// IR: %[[FS:[0-9]+]] = alloca [8 x i8], align 8
// IR-NOT: store {{.*}}, ptr %0
// IR: call {{.*}} @diffe{{.*}}_Z4bodyPK5qfuncPKdPd.poseidon._Z7run_refRK5qfuncPdi.0(ptr %0, ptr %[[FS]],
// IR: ret void

// IR-LABEL: define {{.*}} @_Z7k_alignPKdPd(
// IR: %[[PS:[0-9]+]] = alloca [32 x i8], align 16
// IR: %[[P:[0-9]+]] = getelementptr inbounds nuw i8, ptr %[[PS]], i64 16
// IR: call {{.*}} @diffe{{.*}}_Z4leanPK4pairPd.poseidon._Z7k_alignPKdPd.0(ptr {{.*}}, ptr %[[P]],
// IR: ret void
