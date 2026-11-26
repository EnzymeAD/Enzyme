// REQUIRES: enzyme, cuda-device
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 -ffp-contract=on \
// RUN:     %loadClangPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm --poseidon-cache=%t/cache \
// RUN:     -I%poseidon_profile_inc %s -x none %poseidon_profile_cuda_lib %poseidon_profile_lib \
// RUN:     -L%cuda_lib_dir -lcudart -lstdc++ -lm -o %t/exe
// RUN: env POSEIDON_PROFILE_DIR=%t/small %t/exe 1024 | FileCheck --check-prefix=PRIMAL-SMALL %s
// RUN: env POSEIDON_PROFILE_DIR=%t/large %t/exe 4096 | FileCheck --check-prefix=PRIMAL-LARGE %s
// RUN: FileCheck --check-prefix=SMALL %s < %t/small/_Z6gatheri5TableiPKiPd.fpprofile
// RUN: FileCheck --check-prefix=LARGE %s < %t/large/_Z6gatheri5TableiPKiPd.fpprofile

// A kernel site's recorded traffic per array, at two problem sizes: the extent
// each launch read and wrote, summed over two launches; a gathered array counts
// its extent once, a table of four doubles its 32 bytes per launch.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

struct Table {
  const double *x;
  const double *c;
};

POSEIDON_OPTIMIZE __global__ void gather(int n, Table t, int m, const int *idx,
                                         double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = t.c[i % 4] * t.x[idx[i]] + y[i] * t.c[(i + 1) % 4];
}

int main(int argc, char **argv) {
  const int n = atoi(argv[1]), m = n / 4;
  int *hi = (int *)malloc(sizeof(int) * n);
  double *hx = (double *)malloc(sizeof(double) * m);
  for (int i = 0; i < n; ++i)
    hi[i] = (i * 7) % m;
  for (int i = 0; i < m; ++i)
    hx[i] = 1.0 + i * 1e-3;
  double hc[4] = {0.5, 0.25, 0.125, 1.0};
  int *idx;
  double *x, *c, *y;
  cudaMalloc(&idx, sizeof(int) * n);
  cudaMalloc(&x, sizeof(double) * m);
  cudaMalloc(&c, sizeof(hc));
  cudaMalloc(&y, sizeof(double) * n);
  cudaMemcpy(idx, hi, sizeof(int) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(x, hx, sizeof(double) * m, cudaMemcpyHostToDevice);
  cudaMemcpy(c, hc, sizeof(hc), cudaMemcpyHostToDevice);
  cudaMemset(y, 0, sizeof(double) * n);
  for (int r = 0; r < 2; ++r)
    gather<<<(n + 127) / 128, 128>>>(n, Table{x, c}, m, idx, y);
  double *hy = (double *)malloc(sizeof(double) * n), s = 0.0;
  cudaMemcpy(hy, y, sizeof(double) * n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i)
    s += hy[i];
  printf("sum %.6f\n", s);
  return 0;
}

// PRIMAL-SMALL: {{^}}sum 766.408000{{$}}
// PRIMAL-LARGE: {{^}}sum 4110.112000{{$}}

// SMALL: CanonicalHash = {{[0-9a-f]+}}
// SMALL-NEXT: TrafficArray = [[C:[0-9]]] 1 8
// SMALL-NEXT: TrafficArray = [[I:[0-9]]] 3 -1
// SMALL-NEXT: TrafficArray = [[X:[0-9]]] 1 0
// SMALL-NEXT: TrafficArray = [[Y:[0-9]]] 4 -1
// SMALL-NEXT: Traffic = [[C]] 8 4096 0 64 0
// SMALL-NEXT: Traffic = [[I]] 4 2048 0 8192 0
// SMALL-NEXT: Traffic = [[X]] 8 2048 0 4096 0
// SMALL-NEXT: Traffic = [[Y]] 8 2048 2048 16384 16384
// SMALL-NOT: Traffic

// LARGE: CanonicalHash = {{[0-9a-f]+}}
// LARGE-NEXT: TrafficArray = [[C:[0-9]]] 1 8
// LARGE-NEXT: TrafficArray = [[I:[0-9]]] 3 -1
// LARGE-NEXT: TrafficArray = [[X:[0-9]]] 1 0
// LARGE-NEXT: TrafficArray = [[Y:[0-9]]] 4 -1
// LARGE-NEXT: Traffic = [[C]] 8 16384 0 64 0
// LARGE-NEXT: Traffic = [[I]] 4 8192 0 32768 0
// LARGE-NEXT: Traffic = [[X]] 8 8192 0 16384 0
// LARGE-NEXT: Traffic = [[Y]] 8 8192 8192 65536 65536
// LARGE-NOT: Traffic
