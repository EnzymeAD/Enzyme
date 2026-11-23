// Precision candidates of a kernel whose loops are strided by thread index up
// to a bound the kernel is given, priced and applied end to end. The guard of
// the inner loop is invariant in the outer one. Each candidate is priced on a
// clone cleaned up by the compile's pipeline for the GPU, which does not
// unswitch on a condition that may diverge, so the clone keeps its one copy of
// the body; every candidate is priced, none at zero, and at 1e-6 the cheapest
// candidate meeting the target, FP32, is applied. The emitted site, cleaned up
// without the target, is unswitched: it tests the guard once, before the outer
// loop, computes in FP32, and runs within FP32's error of the binary64 result.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on \
// RUN:     -poseidon-profile-generate -poseidon-cache=%t/pcache %s -o %t/exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profile %t/exe 0 1e-15 \
// RUN:   | FileCheck --check-prefix=EXEC %s
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on \
// RUN:     -poseidon-profile-use=%t/profile -poseidon-cache=%t/c \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-tau=1e-6 %s -o %t/opt.exe 2>&1 \
// RUN:   | FileCheck --check-prefix=SOLVE --implicit-check-not=cand=0.000000e+00 %s
// RUN: %t/opt.exe 1e-9 1e-5 | FileCheck --check-prefix=EXEC %s

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <poseidon/poseidon.h>

POSEIDON_OPTIMIZE __global__ void setup_kernel(int Q, const double *W,
                                               const double *J, double *D) {
  const int e = blockIdx.x;
  for (int qy = threadIdx.y; qy < Q; qy += blockDim.y)
    for (int qx = threadIdx.x; qx < Q; qx += blockDim.x) {
      const int q = qx + Q * qy;
      const double *j = J + 4 * (q + Q * Q * e);
      const double det = j[0] * j[3] - j[1] * j[2];
      D[q + Q * Q * e] = W[q] * (j[0] * j[0] + j[1] * j[1]) / det;
    }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <min error> <max error>\n", argv[0]);
    return 2;
  }
  const double lo = std::atof(argv[1]), hi = std::atof(argv[2]);
  constexpr int Q = 4, NE = 64, NQ = Q * Q;
  static double hW[NQ], hJ[4 * NQ * NE], hD[NQ * NE];
  for (int q = 0; q < NQ; ++q)
    hW[q] = 0.1 + 0.01 * q;
  for (int i = 0; i < NQ * NE; ++i)
    for (int k = 0; k < 4; ++k)
      hJ[4 * i + k] = (k % 3 == 0 ? 1.0 : 0.0) + 0.001 * ((7 * i + k) % 13);
  double *W, *J, *D;
  if (cudaMalloc(&W, sizeof(hW)) != cudaSuccess ||
      cudaMalloc(&J, sizeof(hJ)) != cudaSuccess ||
      cudaMalloc(&D, sizeof(hD)) != cudaSuccess)
    return 1;
  cudaMemcpy(W, hW, sizeof(hW), cudaMemcpyHostToDevice);
  cudaMemcpy(J, hJ, sizeof(hJ), cudaMemcpyHostToDevice);
  setup_kernel<<<NE, dim3(Q, Q)>>>(Q, W, J, D);
  if (cudaMemcpy(hD, D, sizeof(hD), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;
  double maxRel = 0.0;
  for (int e = 0; e < NE; ++e)
    for (int q = 0; q < NQ; ++q) {
      const double *j = hJ + 4 * (q + NQ * e);
      double ref = hW[q] * (j[0] * j[0] + j[1] * j[1]) /
                   (j[0] * j[3] - j[1] * j[2]);
      maxRel = std::fmax(maxRel, std::fabs(hD[q + NQ * e] - ref) /
                                     std::fabs(ref));
    }
  printf("max relative error vs binary64 %.3e\n", maxRel);
  bool ok = maxRel >= lo && maxRel <= hi;
  printf(ok ? "SETUP-PASS\n" : "SETUP-FAIL\n");
  return ok ? 0 : 1;
}

// EXEC: max relative error vs binary64
// EXEC-NEXT: SETUP-PASS

// SOLVE: [poseidon] Optimizing [[BODY:preprocess__Z12setup_kerneliPKdS0_Pd_poseidon_body]] with relative error tolerance: 1.000000e-06
// SOLVE-NOT: cand=0.000000e+00
// SOLVE: CS candidate 'All FP64(0%) + FP32(100%)': initial=6.658000e+03 cand=3.980000e+02 exec=1024
// SOLVE-NOT: cand=0.000000e+00
// SOLVE: [poseidon] [[BODY]]: tau=1.000000e-06 S={{[^ ]+}} A0={{[^ ]+}} selected cost={{[^ ]+}} accCost={{[^ ]+}} rel={{[^ ]+}}{{$}}
// SOLVE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
// SOLVE: [poseidon] pin: _Z12setup_kerneliPKdS0_Pd:PT0=All FP64(0%) + FP32(100%){{$}}

// The emitted site: D = W (j0 j0 + j1 j1) / (j0 j3 - j1 j2) in FP32, under a
// guard tested once before the loops.
// SOLVE: define internal void @[[BODY]](i32 {{[^,]*}}[[Q:%[0-9]+]], ptr {{[^,]*}}[[W:%[0-9]+]], ptr {{[^,]*}}[[J:%[0-9]+]], ptr {{[^,]*}}[[D:%[0-9]+]])
// SOLVE: [[TX:%[0-9]+]] = {{.*}}call {{.*}}i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// SOLVE: [[GUARD:%[0-9]+]] = icmp {{.*}}ult i32 [[TX]], [[Q]]
// SOLVE: br i1 [[GUARD]], label %[[OUTER:[0-9]+]], label %[[EXIT:[0-9]+]]
// SOLVE: {{^}}[[OUTER]]:
// SOLVE-NOT: br i1 [[GUARD]],
// SOLVE: [[J0P:%[0-9]+]] = getelementptr {{.*}}ptr [[J]], i64 {{%[0-9]+}}
// SOLVE-DAG: [[J0:%[0-9]+]] = load double, ptr [[J0P]]
// SOLVE-DAG: [[J1P:%[0-9]+]] = getelementptr {{.*}}ptr [[J0P]], i64 8
// SOLVE-DAG: [[J1:%[0-9]+]] = load double, ptr [[J1P]]
// SOLVE-DAG: [[J2P:%[0-9]+]] = getelementptr {{.*}}ptr [[J0P]], i64 16
// SOLVE-DAG: [[J2:%[0-9]+]] = load double, ptr [[J2P]]
// SOLVE-DAG: [[J3P:%[0-9]+]] = getelementptr {{.*}}ptr [[J0P]], i64 24
// SOLVE-DAG: [[J3:%[0-9]+]] = load double, ptr [[J3P]]
// SOLVE-DAG: [[J0F:%[0-9]+]] = fptrunc double [[J0]] to float
// SOLVE-DAG: [[J1F:%[0-9]+]] = fptrunc double [[J1]] to float
// SOLVE-DAG: [[J2F:%[0-9]+]] = fptrunc double [[J2]] to float
// SOLVE-DAG: [[J3F:%[0-9]+]] = fptrunc double [[J3]] to float
// SOLVE-DAG: [[NJ2:%[0-9]+]] = fneg float [[J2F]]
// SOLVE-DAG: [[P12:%[0-9]+]] = fmul float [[J1F]], [[NJ2]]
// SOLVE-DAG: [[DET:%[0-9]+]] = {{(tail )?}}call float @llvm.fmuladd.f32(float [[J0F]], float [[J3F]], float [[P12]])
// SOLVE: [[WP:%[0-9]+]] = getelementptr {{.*}}ptr [[W]], i64 {{%[0-9]+}}
// SOLVE-NEXT: [[WV:%[0-9]+]] = load double, ptr [[WP]]
// SOLVE-DAG: [[SQ1:%[0-9]+]] = fmul float [[J1F]], [[J1F]]
// SOLVE-DAG: [[SQ:%[0-9]+]] = {{(tail )?}}call float @llvm.fmuladd.f32(float [[J0F]], float [[J0F]], float [[SQ1]])
// SOLVE-DAG: [[WF:%[0-9]+]] = fptrunc double [[WV]] to float
// SOLVE: [[NUM:%[0-9]+]] = fmul float [[SQ]], [[WF]]
// SOLVE-NEXT: [[QT:%[0-9]+]] = fdiv float [[NUM]], [[DET]]
// SOLVE-NEXT: [[R:%[0-9]+]] = fpext float [[QT]] to double
// SOLVE: [[DP:%[0-9]+]] = getelementptr {{.*}}ptr [[D]], i64 {{%[0-9]+}}
// SOLVE-NEXT: store double [[R]], ptr [[DP]]
// SOLVE-NOT: br i1 [[GUARD]],
// SOLVE: {{^}}[[EXIT]]:
// SOLVE-NEXT: ret void
