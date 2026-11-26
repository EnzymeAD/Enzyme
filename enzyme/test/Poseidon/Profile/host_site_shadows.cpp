// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c -poseidon-print %s -o %t/gen 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: llvm-nm %t/gen | FileCheck --check-prefix=SYM %s
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=OUT %s
// RUN: FileCheck --check-prefix=PROF %s < %t/prof/_Z5scalePKdPdi.fpprofile
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c2 -DUNKNOWN %s -o %t/unknown
// RUN: env POSEIDON_PROFILE_DIR=%t/prof2 not --crash %t/unknown 2>&1 | FileCheck --check-prefix=UNKNOWN %s
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c3 -DMARKER_ONLY %s -o %t/marker
// RUN: llvm-nm %t/marker | FileCheck --check-prefix=MARKER %s
// RUN: env POSEIDON_PROFILE_DIR=%t/prof3 %t/marker
// RUN: ls %t/prof3/_Z5scalePKdPdi.fpprofile
// RUN: %poseidon_clangxx -O2 -poseidon-profile-generate -poseidon-cache=%t/c4 -DFLUX %s -o %t/flux
// RUN: env POSEIDON_PROFILE_DIR=%t/prof4 %t/flux | FileCheck --check-prefix=FLUX-OUT %s
// RUN: FileCheck --check-prefix=FLUX %s < %t/prof4/_Z4fluxPKdPdi.fpprofile

// Host site shadows are sized from the allocation each pointer points into:
// every heap allocator, static arrays, stack arrays (also through a call that
// is not inlined), interior pointers; each element written counts once, with
// the seed of its index in its allocation (outputSeed). A pointer into memory
// the profiler did not see allocated stops the run. A build with marker sites
// only (MARKER_ONLY) links the allocator, which then only passes allocations
// through. A flux added to one output and subtracted from the next (FLUX), which
// the sum of the outputs does not depend on, has a sensitivity under the seed.

#include <cstdlib>

#ifdef MARKER_ONLY
template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

__attribute__((noinline)) void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i] * x[i] + x[i];
}

int main() {
  const int n = 64;
  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(n * sizeof(double));
  double *dx = (double *)calloc(n, sizeof(double));
  double *dy = (double *)malloc(n * sizeof(double));
  for (int i = 0; i < n; ++i) {
    x[i] = 0.01 * i;
    dy[i] = 1.0;
  }
  __poseidon_fp_optimize<void>((void *)scale, enzyme_dup, x, dx, enzyme_dup, y, dy, n);
  return y[3] > 1.0;
}
#elif defined(FLUX)
#include "poseidon/poseidon.h"
#include <cstdio>

POSEIDON_OPTIMIZE void flux(const double *x, double *y, int m) {
  for (int f = 0; f < m; ++f) {
    double g = x[f] * 0.75 + 0.125;
    y[2 * f] += g;
    y[2 * f + 1] -= g;
  }
}

int main() {
  const int m = 32;
  double *x = (double *)malloc(m * sizeof(double));
  double *y = (double *)calloc(2 * m, sizeof(double));
  for (int f = 0; f < m; ++f)
    x[f] = 0.5 + 0.01 * f;
  flux(x, y, m);
  printf("%g %g\n", y[0], y[63]);
}
#else
#include "poseidon/poseidon.h"
#include <cstdio>
#include <new>
#include <sys/mman.h>
#include <vector>

constexpr int n = 64;

POSEIDON_OPTIMIZE void scale(const double *x, double *y, int m) {
  for (int i = 0; i < m; ++i)
    y[i] = x[i] * x[i] + x[i];
}

__attribute__((noinline)) void through(const double *x, double *y, int m) { scale(x, y, m); }

static double fileX[n], fileY[n];

int main() {
  static double localY[n];
  double stackX[n], stackY[n], stackZ[n];
  double *mx = (double *)malloc(n * sizeof(double));
  double *my = (double *)malloc(n * sizeof(double));
  double *cy = (double *)calloc(n, sizeof(double));
  double *ry = (double *)realloc(malloc(8), n * sizeof(double));
  double *ny = new double[n];
  double *one = new double;
  double *ay = new (std::align_val_t(64)) double[n];
  double *aa = (double *)aligned_alloc(64, n * sizeof(double));
  void *pm = nullptr;
  if (posix_memalign(&pm, 64, n * sizeof(double)))
    return 1;
  std::vector<double> vy(n);
  for (int i = 0; i < n; ++i)
    mx[i] = stackX[i] = fileX[i] = 0.01 * (i + 1);

  scale(mx, my, n);
  scale(mx, cy, n);
  scale(mx, ry, n);
  scale(mx, ny, n);
  scale(mx, one, 1);
  scale(mx, ay, n);
  scale(mx, aa, n);
  scale(mx, (double *)pm, n);
  scale(mx, vy.data(), n);
  scale(fileX, fileY, n);
  scale(fileX, localY, n);
  scale(stackX, stackY, n);
  through(stackX, stackZ, n);
  double *iy = (double *)malloc(n * sizeof(double));
  scale(mx + 8, iy + 8, n - 8);
#ifdef UNKNOWN
  double *mm = (double *)mmap(nullptr, n * sizeof(double), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  scale(mx, mm, n);
#endif
  printf("%g %g %g %g %g %g\n", my[3], vy[3], fileY[3], localY[3], stackY[3] + stackZ[3], iy[9]);
}
#endif

// GEN: [poseidon] scale(double const*, double*, int): site 0, 2 shadow buffer(s), output(s) 1
// SYM: {{[VvDdRr]}} __poseidon_host_sites_present
// OUT: 0.0416 0.0416 0.0416 0.0416 0.0832 0.11
// One record (y = x * x + x, x in [0.01, 0.64]) over 12 calls of 64 elements,
// one of 1 and one of 56 from the ninth element on; each element's gradient is
// its seed.
// PROF: MinRes = 1.00999999999999996e-02
// PROF-NEXT: MaxRes = 1.04960000000000009e+00
// PROF-NEXT: SumSens = 4.075769764{{[0-9]+}}e+02
// PROF-NEXT: SumGrad = 6.605011246470{{[0-9]+}}e+01
// PROF-NEXT: SumAbsGrad = 8.661119977248{{[0-9]+}}e+02
// PROF-NEXT: SumSqGrad = 9.690972830498{{[0-9]+}}e+02
// PROF-NEXT: SumSqSens = 2.992841263329{{[0-9]+}}e+02
// PROF-NEXT: Exec = 825
// PROF-NEXT: NumOperands = 3
// PROF-NEXT: Operand[0] = [1.00000000000000002e-02, 6.40000000000000013e-01, 1.00000000000000002e-02]
// PROF-NEXT: Operand[1] = [1.00000000000000002e-02, 6.40000000000000013e-01, 1.00000000000000002e-02]
// PROF-NEXT: Operand[2] = [1.00000000000000002e-02, 6.40000000000000013e-01, 1.00000000000000002e-02]
// UNKNOWN: [poseidon] argument 2 ('y') of scale(double const*, double*, int) points to 0x{{[0-9a-f]+}}, which the profiler did not see allocated.
// UNKNOWN-SAME: Allocate this array on the heap, or give its shadow in an explicit list of __poseidon_optimize: enzyme_dup before the pointer and the shadow after it.
// MARKER: {{^ +}}w __poseidon_host_sites_present
// MARKER: T malloc
// FLUX-OUT: {{^}}0.5 -0.7325{{$}}
// The flux, then the two updates.
// FLUX: SiteId = 0
// FLUX: {{^0$}}
// FLUX-NEXT: MinRes = 5.00000000000000000e-01
// FLUX-NEXT: MaxRes = 7.32500000000000040e-01
// FLUX-NEXT: SumSens = 1.9035803788{{[0-9]+}}e+01
// FLUX-NEXT: SumGrad = -1.4719200107{{[0-9]+}}e+00
// FLUX-NEXT: SumAbsGrad = 3.0554717623{{[0-9]+}}e+01
// FLUX-NEXT: SumSqGrad = 5.5377054003{{[0-9]+}}e+01
// FLUX-NEXT: SumSqSens = 2.1997792344{{[0-9]+}}e+01
// FLUX-NEXT: Exec = 32
// FLUX: {{^1$}}
// FLUX-NEXT: MinRes = 5.00000000000000000e-01
// FLUX-NEXT: MaxRes = 7.32500000000000040e-01
// FLUX-NEXT: SumSens = 2.1531320261{{[0-9]+}}e+01
// FLUX-NEXT: SumGrad = 1.8651746642{{[0-9]+}}e+00
// FLUX-NEXT: SumAbsGrad = 3.4814760435{{[0-9]+}}e+01
// FLUX-NEXT: SumSqGrad = 4.0261344721{{[0-9]+}}e+01
// FLUX-NEXT: SumSqSens = 1.5695070868{{[0-9]+}}e+01
// FLUX-NEXT: Exec = 32
// FLUX: {{^2$}}
// FLUX-NEXT: MinRes = -7.32500000000000040e-01
// FLUX-NEXT: MaxRes = -5.00000000000000000e-01
// FLUX-NEXT: SumSens = 1.9909248655{{[0-9]+}}e+01
// FLUX-NEXT: SumGrad = 3.3370946750{{[0-9]+}}e+00
// FLUX-NEXT: SumAbsGrad = 3.2326091102{{[0-9]+}}e+01
// FLUX-NEXT: SumSqGrad = 3.4799021718{{[0-9]+}}e+01
// FLUX-NEXT: SumSqSens = 1.3342142892{{[0-9]+}}e+01
// FLUX-NEXT: Exec = 32
// FLUX-NOT: {{^3$}}
