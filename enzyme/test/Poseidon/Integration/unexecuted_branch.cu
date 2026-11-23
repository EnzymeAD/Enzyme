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

// USE: [poseidon] Optimizing [[BODY:preprocess__Z5setupPKdS0_iPdi_poseidon_body]] with relative error tolerance: 1.000000e-06
// USE: [poseidon] [[BODY]]: 3 floating-point operation(s) the profiling run never executed are left as written (unexecuted_branch.cu:34): no profile gives their ranges, cost or error, so the subgraphs of the executed operations end at them and the error a rewrite of those passes on to them is not modelled{{$}}
// USE: [poseidon] [[BODY]]: tau=1.000000e-06 S={{[^ ]+}} A0={{[^ ]+}} selected cost={{[^ ]+}} accCost={{[^ ]+}} rel={{[^ ]+}}{{$}}
// USE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
// USE: [poseidon] pin: _Z5setupPKdS0_iPdi:PT0=All FP64(0%) + FP32(100%){{$}}

// The determinant, the adjugate and the scalar branch are computed in FP32;
// the matrix branch takes them widened and computes in FP64 as written.
// USE: define internal void @[[BODY]](ptr {{[^,]*}}[[J:%[0-9]+]], ptr {{[^,]*}}[[M:%[0-9]+]], i32 {{[^,]*}}[[MAT:%[0-9]+]], ptr {{[^,]*}}[[D:%[0-9]+]], i32 {{[^,]*}}%{{[0-9]+}})
// USE: [[PA:%[0-9]+]] = getelementptr {{.*}}ptr [[J]], i64 {{%[0-9]+}}
// USE-NEXT: [[A:%[0-9]+]] = load double, ptr [[PA]]
// USE-NEXT: [[PB:%[0-9]+]] = getelementptr {{.*}}ptr [[PA]], i64 8
// USE-NEXT: [[B:%[0-9]+]] = load double, ptr [[PB]]
// USE-NEXT: [[BF:%[0-9]+]] = fptrunc double [[B]] to float
// USE-NEXT: [[NB:%[0-9]+]] = fneg float [[BF]]
// USE-NEXT: [[BB:%[0-9]+]] = fmul float [[BF]], [[NB]]
// USE-NEXT: [[AF:%[0-9]+]] = fptrunc double [[A]] to float
// USE-NEXT: [[AA:%[0-9]+]] = {{(tail )?}}call float @llvm.fmuladd.f32(float [[AF]], float [[AF]], float [[BB]])
// USE-NEXT: [[DET:%[0-9]+]] = fadd float [[AA]], 3.000000e+00
// USE-NEXT: [[ADJ:%[0-9]+]] = {{(tail )?}}call float @llvm.fmuladd.f32(float [[DET]], float [[AF]], float [[BF]])
// USE-NEXT: [[SCALAR:%[0-9]+]] = icmp eq i32 [[MAT]], 0
// USE: br i1 [[SCALAR]], label %[[SB:[0-9]+]], label %[[MB:[0-9]+]]
// USE: {{^}}[[MB]]:
// USE-NEXT: [[ADJD:%[0-9]+]] = fpext float [[ADJ]] to double
// USE-NEXT: [[DETD:%[0-9]+]] = fpext float [[DET]] to double
// USE-NEXT: [[PM:%[0-9]+]] = getelementptr {{.*}}ptr [[M]], i64 [[I:%[0-9]+]]
// USE-NEXT: [[MV:%[0-9]+]] = load double, ptr [[PM]]
// USE-NEXT: [[DM:%[0-9]+]] = fmul double [[MV]], [[DETD]]
// USE-NEXT: [[DMM:%[0-9]+]] = fmul double [[MV]], [[DM]]
// USE-NEXT: [[RM:%[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double [[ADJD]], double [[MV]], double [[DMM]])
// USE: {{^}}[[SB]]:
// USE-NEXT: [[RS:%[0-9]+]] = {{(tail )?}}call float @llvm.fmuladd.f32(float [[ADJ]], float 5.000000e-01, float [[DET]])
// USE-NEXT: [[RSD:%[0-9]+]] = fpext float [[RS]] to double
// USE: [[R:%[0-9]+]] = phi double [ [[RM]], %[[MB]] ], [ [[RSD]], %[[SB]] ]
// USE-NEXT: [[PD:%[0-9]+]] = getelementptr {{.*}}ptr [[D]], i64 [[I]]
// USE-NEXT: store double [[R]], ptr [[PD]]
// USE: [poseidon] {{.*}}unexecuted_branch.cu: 1 function(s) a rewrite reached; every other function as written
// OUT: within 1e-6
