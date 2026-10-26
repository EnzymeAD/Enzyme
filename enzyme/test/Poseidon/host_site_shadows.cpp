// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen 2> %t/gen.err
// RUN: FileCheck --check-prefix=GEN %s < %t/gen.err
// RUN: llvm-nm %t/gen | FileCheck --check-prefix=SYM %s
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen | FileCheck --check-prefix=OUT %s
// RUN: FileCheck --check-prefix=PROF %s < %t/prof/preprocess__Z5scalePKdPdi_poseidon_body.fpprofile
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c2 -DUNKNOWN %s -o %t/unknown 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof2 not --crash %t/unknown 2>&1 | FileCheck --check-prefix=UNKNOWN %s
// REQUIRES: poseidon, enzyme

// The shadows of a POSEIDON_OPTIMIZE function on the host, sized from the
// allocation each pointer argument points into: every heap allocator, a
// file-scope and a function-local static array, stack arrays passed directly
// and through a function that is not inlined, and interior pointers. Each call
// writes a buffer no other call writes, so each output carries the seed once
// and the gradient on the site's last operation counts every element written.
// A pointer into memory the profiler did not see allocated stops the run,
// naming the function and the argument.

#include "poseidon/poseidon.h"
#include <cstdio>
#include <cstdlib>
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

// GEN: [poseidon-profgen] scale(double const*, double*, int): site 0, 2 shadow buffer(s), output(s) 1

// SYM: {{[VvDdRr]}} __poseidon_host_sites_present

// OUT: 0.0416 0.0416 0.0416 0.0416 0.0832 0.11

// 12 calls of 64 elements, one of 1 and one of 56.
// PROF: SumGrad = 8.25000000000000000e+02
// PROF: Exec = 825

// UNKNOWN: Poseidon FPProfiler: argument 2 ('y') of scale(double const*, double*, int) points to 0x{{[0-9a-f]+}}, which the profiler did not see allocated.
