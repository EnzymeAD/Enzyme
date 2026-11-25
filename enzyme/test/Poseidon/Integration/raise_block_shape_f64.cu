// The 8x8x8 product of raise_scalar_matmul_f64.cu, profiled in an 8x8 block and
// raised onto the tensor cores, then launched in the profiled block and in an
// 8x4 one. The raise is laid out for the profiled block, so the 8x4 launch
// runs the loop as written: its four rows are the FP64 product, and the rows
// it does not reach are left as they were. The 8x8 launch runs the f16 tile.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 \
// RUN:     %loadClangPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm -poseidon-cache=%t/cg \
// RUN:     -I%poseidon_profile_inc %s -x none %poseidon_profile_cuda_lib %poseidon_profile_lib \
// RUN:     -L%cuda_lib_dir -lcudart -lstdc++ -lm -o %t/gen.exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profiles %t/gen.exe 8 | FileCheck --check-prefix=PROFILED %s
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 %loadClangPoseidon \
// RUN:     -mllvm -poseidon-profile-use=%t/profiles -mllvm -poseidon-cache=%t/c \
// RUN:     -mllvm -poseidon-raise-wmma -mllvm -poseidon-ozaki-host-dispatch=false \
// RUN:     -mllvm -poseidon-enable-herbie=false -mllvm -poseidon-enable-pt=false \
// RUN:     -mllvm -poseidon-cost-model=%gpu_cost_model -mllvm -poseidon-print \
// RUN:     %s -L%cuda_lib_dir -lcudart -o %t/opt.exe 2>&1 | FileCheck --check-prefix=RAISE %s
// RUN: %t/opt.exe 8 | FileCheck --check-prefix=PROFILED %s
// RUN: %t/opt.exe 4 | FileCheck --check-prefix=OTHER %s

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;

extern "C" __device__ __attribute__((noinline)) void
block_matmul_body_f64(const double *A, const double *B, double *D) {
  unsigned tx = threadIdx.x;
  unsigned ty = threadIdx.y;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < 8; ++k)
    acc += A[ty * 8 + k] * B[k * 8 + tx];
  D[ty * 8 + tx] = acc;
}

__global__ void block_matmul_kernel(const double *A, const double *B,
                                    double *D) {
  __poseidon_fp_optimize<void>((void *)block_matmul_body_f64, enzyme_const, A,
                               enzyme_const, B, enzyme_const, D);
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Launches the kernel in an 8 x `rows` block and prints the largest relative
// error of the rows it reaches against the FP64 product, and whether the rows
// it does not reach kept their value.
int main(int argc, char **argv) {
  constexpr int M = 8, N = 8, K = 8;
  const int rows = argc > 1 ? std::atoi(argv[1]) : M;
  double hA[M * K], hB[K * N], hD[M * N];
  for (int i = 0; i < M * K; ++i)
    hA[i] = 1.0 + 0.25 * (i % 5);
  for (int i = 0; i < K * N; ++i)
    hB[i] = 0.5 + 0.125 * (i % 7);
  for (int i = 0; i < M * N; ++i)
    hD[i] = -1.0;

  double *dA = nullptr, *dB = nullptr, *dD = nullptr;
  CK(cudaMalloc(&dA, sizeof(hA)));
  CK(cudaMalloc(&dB, sizeof(hB)));
  CK(cudaMalloc(&dD, sizeof(hD)));
  CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dD, hD, sizeof(hD), cudaMemcpyHostToDevice));
  block_matmul_kernel<<<1, dim3(N, rows)>>>(dA, dB, dD);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

  double maxRel = 0.0;
  bool untouched = true;
  for (int ty = 0; ty < M; ++ty)
    for (int tx = 0; tx < N; ++tx) {
      if (ty >= rows) {
        untouched &= hD[ty * N + tx] == -1.0;
        continue;
      }
      double expected = 0.0;
      for (int k = 0; k < K; ++k)
        expected += hA[ty * K + k] * hB[k * N + tx];
      maxRel = std::fmax(maxRel,
                         std::fabs(hD[ty * N + tx] - expected) / expected);
    }
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dD));
  printf("block 8x%d: %s, rows beyond %s\n", rows,
         maxRel <= 1e-12   ? "FP64"
         : maxRel <= 1e-2  ? "F16"
                           : "WRONG",
         untouched ? "untouched" : "WRITTEN");
  return 0;
}

// RAISE: Applying solution for matmul[0] -> wmma m16n16k16 f16/f16 (#0)

// PROFILED: block 8x8: {{FP64|F16}}, rows beyond untouched
// OTHER: block 8x4: FP64, rows beyond untouched
