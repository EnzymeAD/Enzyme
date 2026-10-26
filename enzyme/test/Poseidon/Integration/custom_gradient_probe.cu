// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 \
// RUN:     %clangLoadPoseidonEnzyme %s \
// RUN:     -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lm -o %t.exe
// RUN: %t.exe | FileCheck %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

// An identity probe carrying an Enzyme custom derivative, differentiated
// inside an NVPTX kernel. Poseidon's profile generation places such probes on
// every profiled value and reads the accumulated adjoint out of the reverse
// pass, so this is the contract that has to hold on the device.

#include <cstdio>
#include <cuda_runtime.h>

__device__ double d_adj[5];
__device__ double d_val[5];

extern "C" __device__ __attribute__((noinline)) double
poseidon_probe(double v, long slot) {
  // Opaque identity. Poseidon builds the probe immediately before Enzyme runs,
  // so a plain `return v` survives there; in this source-level test the
  // optimizer would infer `returned` and fold the call away before AD.
  asm volatile("" : "+d"(v) : "l"(slot));
  return v;
}

// Returns the primal value: the probed value is used downstream, so Enzyme
// needs the augmented forward to produce it. No tape.
extern "C" __device__ __attribute__((noinline)) double
poseidon_probe_aug(double v, long slot) {
  return v;
}

// Enzyme massages the constant `slot` into a duplicated argument, so the
// reverse takes a shadow for it, then the adjoint of the return. Returns the
// adjoint of v.
extern "C" __device__ __attribute__((noinline)) double
poseidon_probe_rev(double v, long slot, long shadow_slot, double d_ret) {
  atomicAdd(&d_adj[slot], d_ret);
  // The primal reaching the reverse is the datum the profiler's sensitivity
  // record is built from, so check it as well as the adjoint.
  atomicAdd(&d_val[slot], v);
  return d_ret;
}

// A __device__ global has an undef initializer in the host module, which the
// registration lowering rejects, so the array exists only in the device
// compilation.
#ifdef __CUDA_ARCH__
__device__ void *__enzyme_register_gradient_poseidon_probe[3] = {
    (void *)poseidon_probe, (void *)poseidon_probe_aug,
    (void *)poseidon_probe_rev};
#endif

extern "C" __device__ __attribute__((noinline)) void
body(const double *x, double *out) {
  double acc = 1.0;
#pragma unroll 1
  for (int i = 0; i < 4; ++i)
    acc = poseidon_probe(acc * x[i], i);
  *out = poseidon_probe(acc * 2.0, 4);
}

template <typename... T> __device__ void __enzyme_autodiff(void *, T...);
__device__ int enzyme_dup;

__global__ void diff_body(const double *x, double *dx, double *out,
                          double *dout) {
  __enzyme_autodiff((void *)body, enzyme_dup, x, dx, enzyme_dup, out, dout);
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  double hx[4] = {1.0, 2.0, 3.0, 4.0};
  double *x, *dx, *out, *dout;
  CK(cudaMalloc(&x, sizeof(hx)));
  CK(cudaMalloc(&dx, sizeof(hx)));
  CK(cudaMalloc(&out, sizeof(double)));
  CK(cudaMalloc(&dout, sizeof(double)));
  CK(cudaMemcpy(x, hx, sizeof(hx), cudaMemcpyHostToDevice));
  CK(cudaMemset(dx, 0, sizeof(hx)));
  CK(cudaMemset(out, 0, sizeof(double)));
  double seed = 1.0;
  CK(cudaMemcpy(dout, &seed, sizeof(double), cudaMemcpyHostToDevice));
  double zeros[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  CK(cudaMemcpyToSymbol(d_adj, zeros, sizeof(zeros)));
  CK(cudaMemcpyToSymbol(d_val, zeros, sizeof(zeros)));

  diff_body<<<1, 1>>>(x, dx, out, dout);
  CK(cudaDeviceSynchronize());

  double hadj[5], hdx[4], hval[5];
  CK(cudaMemcpyFromSymbol(hadj, d_adj, sizeof(hadj)));
  CK(cudaMemcpyFromSymbol(hval, d_val, sizeof(hval)));
  CK(cudaMemcpy(hdx, dx, sizeof(hdx), cudaMemcpyDeviceToHost));

  // out = 2*x0*x1*x2*x3; the adjoint of the value entering probe slot i is
  // 2 * prod_{j>i} x_j, and dx_i = adjoint_i * prod_{j<i} x_j.
  for (int i = 0; i < 5; ++i)
    printf("adj[%d] = %.1f\n", i, hadj[i]);
  for (int i = 0; i < 5; ++i)
    printf("val[%d] = %.1f\n", i, hval[i]);
  for (int i = 0; i < 4; ++i)
    printf("dx[%d] = %.1f\n", i, hdx[i]);
  return 0;
}

// CHECK: adj[0] = 48.0
// CHECK-NEXT: adj[1] = 24.0
// CHECK-NEXT: adj[2] = 8.0
// CHECK-NEXT: adj[3] = 2.0
// CHECK-NEXT: adj[4] = 1.0
// CHECK-NEXT: val[0] = 1.0
// CHECK-NEXT: val[1] = 2.0
// CHECK-NEXT: val[2] = 6.0
// CHECK-NEXT: val[3] = 24.0
// CHECK-NEXT: val[4] = 48.0
// CHECK-NEXT: dx[0] = 48.0
// CHECK-NEXT: dx[1] = 24.0
// CHECK-NEXT: dx[2] = 16.0
// CHECK-NEXT: dx[3] = 12.0
