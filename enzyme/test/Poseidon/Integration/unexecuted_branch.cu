// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -g -ffp-contract=on -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen 0 | FileCheck --check-prefix=FP64 %s
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O3 -g -ffp-contract=on -poseidon-profile-use=%t/prof -poseidon-loose-coverage -poseidon-tau=1e-6 -poseidon-cost-model=%gpu_cost_model -poseidon-enable-herbie=0 -poseidon-print -poseidon-cache=%t/c %s -o %t/use 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err
// RUN: %t/use 0 | FileCheck --check-prefix=OUT %s
// RUN: %t/use 1 | FileCheck --check-prefix=OUT %s

// A setup kernel in the shape of a finite-element geometric factor: the
// determinant and adjugate feed a scalar-coefficient branch and a
// matrix-coefficient branch, and the profiling run takes the scalar one only.
// The operations of the other branch have no profile; under
// -poseidon-loose-coverage they are left as written, the executed operations
// are priced and rewritten, and both branches still compute their result.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void setup(const double *J, const double *M,
                                        int matrix, double *D, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double a = J[2 * i], b = J[2 * i + 1];
  double det = a * a - b * b + 3.0;
  double adj = det * a + b;
  if (matrix) {
    double m = M[i];
    D[i] = adj * m + det * m * m;
  } else {
    D[i] = adj * 0.5 + det;
  }
}

int main(int argc, char **argv) {
  const int matrix = argc > 1 ? std::atoi(argv[1]) : 0;
  const int n = 1024;
  double hJ[2 * n], hM[n], hD[n];
  for (int i = 0; i < n; ++i) {
    hJ[2 * i] = 1.0 + i / double(n);
    hJ[2 * i + 1] = 0.25 + 0.5 * i / double(n);
    hM[i] = 2.0 - i / double(n);
  }
  double *J, *M, *D;
  cudaMalloc(&J, sizeof hJ);
  cudaMalloc(&M, sizeof hM);
  cudaMalloc(&D, sizeof hD);
  cudaMemcpy(J, hJ, sizeof hJ, cudaMemcpyHostToDevice);
  cudaMemcpy(M, hM, sizeof hM, cudaMemcpyHostToDevice);
  setup<<<n / 128, 128>>>(J, M, matrix, D, n);
  cudaMemcpy(hD, D, sizeof hD, cudaMemcpyDeviceToHost);
  double ref = 0, err = 0;
  for (int i = 0; i < n; ++i) {
    double a = hJ[2 * i], b = hJ[2 * i + 1], m = hM[i];
    double det = a * a - b * b + 3.0, adj = det * a + b;
    double r = matrix ? adj * m + det * m * m : adj * 0.5 + det;
    ref += r * r;
    err += (hD[i] - r) * (hD[i] - r);
  }
  double rel = std::sqrt(err / ref);
  printf("%s\n", rel < 1e-14 ? "as FP64" : rel <= 1e-6 ? "within 1e-6" : "too far");
  return 0;
}

// FP64: as FP64

// USE: [poseidon] {{.*}}setup{{.*}}: 3 floating-point operation(s) the profiling run never executed are left as written (unexecuted_branch.cu:34): no profile gives their ranges, cost or error
// USE: [poseidon] pin: _Z5setupPKdS0_iPdi:PT0=
// OUT: within 1e-6
