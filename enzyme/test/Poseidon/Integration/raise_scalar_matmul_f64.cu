// A scalar 8x8x8 FP64 loop with enzyme_const pointers (no gradient), so with no
// target the solve raises it to the cheapest f16 WMMA tile, staged in shared
// memory, with no FP arithmetic of its own left.
// REQUIRES: enzyme, cuda-device
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -fcuda-rdc -O2 %loadClangPoseidonEnzyme -mllvm --poseidon-profile-generate -mllvm -poseidon-cache=%t/cg -I%poseidon_profile_inc -DPOSEIDON_DEVICE_MATH %s -x none %poseidon_profile_cuda_lib %poseidon_profile_lib -L%cuda_lib_dir -lcudart -lstdc++ -lm -o %t/exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profiles %t/exe | FileCheck --check-prefix=PRIMAL %s
// RUN: FileCheck --check-prefix=PROF %s < %t/profiles/scalar_matmul_body_f64.fpprofile
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -I%poseidon_profile_inc -DPOSEIDON_DEVICE_MATH -S -emit-llvm %s -o %t/dev.ll
// RUN: %opt %t/dev.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%t/profiles -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-cache= -poseidon-enable-matmul -poseidon-print -poseidon-matmul-library-calls=false -poseidon-cost-model=%gpu_cost_model -S 2>&1 | tee %t/use.out | FileCheck %s
// RUN: FileCheck --check-prefix=NOFP %s < %t/use.out

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;

extern "C" __device__ __attribute__((noinline)) void
scalar_matmul_body_f64(const double *A, const double *B, double *D) {
  unsigned tx = threadIdx.x;
  unsigned ty = threadIdx.y;
  double acc = 0.0;
#pragma unroll 1
  for (int k = 0; k < 8; ++k)
    acc += A[ty * 8 + k] * B[k * 8 + tx];
  D[ty * 8 + tx] = acc;
}

__global__ void scalar_matmul_kernel(const double *A, const double *B,
                                     double *D) {
  __poseidon_fp_optimize<void>((void *)scalar_matmul_body_f64, enzyme_const, A,
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

int main(void) {
  constexpr int M = 8, N = 8, K = 8;
  double hA[M * K], hB[K * N], hD[M * N];
  for (int i = 0; i < M * K; ++i)
    hA[i] = 1.0 + 0.25 * (i % 5);
  for (int i = 0; i < K * N; ++i)
    hB[i] = 0.5 + 0.125 * (i % 7);

  double *dA = nullptr, *dB = nullptr, *dD = nullptr;
  CK(cudaMalloc(&dA, sizeof(hA)));
  CK(cudaMalloc(&dB, sizeof(hB)));
  CK(cudaMalloc(&dD, sizeof(hD)));
  CK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  scalar_matmul_kernel<<<1, dim3(N, M)>>>(dA, dB, dD);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

  int nbad = 0;
  for (int ty = 0; ty < M; ++ty) {
    for (int tx = 0; tx < N; ++tx) {
      double expected = 0.0;
      for (int k = 0; k < K; ++k)
        expected += hA[ty * K + k] * hB[k * N + tx];
      if (std::fabs(hD[ty * N + tx] - expected) > 1e-12) {
        if (nbad < 4)
          fprintf(stderr, "D[%d,%d] = %.17g (expected %.17g)\n", ty, tx,
                  hD[ty * N + tx], expected);
        ++nbad;
      }
    }
  }
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dD));
  if (nbad) {
    fprintf(stderr, "MATMUL-FAIL: %d mismatches\n", nbad);
    return 1;
  }
  printf("MATMUL-PASS\n");
  return 0;
}

// PRIMAL: MATMUL-PASS

// PROF: MaxBlockDims = 8 8 1
// PROF-NEXT: MaxGridDims = 1 1 1
// PROF-NEXT: LaunchCount = 1
// PROF-NEXT: SiteId = 0
// PROF-NEXT: ReductionTrip = 0 8
// PROF-NEXT: CanonicalHash = fea337c617ef94c9
// PROF-NEXT: {{^0$}}
// PROF: SumGrad = 0.00000000000000000e+00
// PROF-NEXT: SumAbsGrad = 0.00000000000000000e+00
// PROF-NEXT: Exec = 512
// PROF-NOT: {{^[0-9]+$}}

// CHECK: [poseidon] Optimizing [[BODY:preprocess_scalar_matmul_body_f64]] with relative error tolerance: 0.000000e+00
// CHECK: [poseidon] Found 1 matrix product(s) in preprocess_scalar_matmul_body_f64
// CHECK-NEXT: Matmul[0]: 8x8x8 type=f64 origin=ConstantShape
// CHECK-NOT: Matmul[1]:
// CHECK: Starting accuracy DP solver with cost budget: 0.000000e+00 of the site's profiled cost
// CHECK: Applying solution for matmul[0] -> wmma m16n16k16 f16/f16 (#0)
// CHECK: [poseidon] pin: scalar_matmul_body_f64:M0=wmma m16n16k16 f16/f16{{$}}

// The kernel calls the raised body.
// CHECK-LABEL: ; ModuleID =
// CHECK-LABEL: define dso_local ptx_kernel void @_Z20scalar_matmul_kernelPKdS0_Pd(
// CHECK-SAME: ptr {{[^,]*}}[[A:%[0-9]+]], ptr {{[^,]*}}[[B:%[0-9]+]], ptr {{[^,]*}}[[D:%[0-9]+]])
// CHECK-NEXT: call void @[[BODY]](ptr [[A]], ptr [[B]], ptr [[D]])
// CHECK-NEXT: ret void

// The raise in the profiled 8x8 block, the loop elsewhere; one warp does the
// tiles, every thread widens its element into D.
// CHECK-LABEL: define dso_local void @preprocess_scalar_matmul_body_f64(
// CHECK-SAME: ptr {{[^,]*}}[[PA:%[0-9]+]], ptr {{[^,]*}}[[PB:%[0-9]+]], ptr {{[^,]*}}[[PD:%[0-9]+]])
// CHECK: [[NX:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK-NEXT: [[X:%[0-9]+]] = icmp eq i32 [[NX]], 8
// CHECK-NEXT: [[NY:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
// CHECK-NEXT: [[Y:%[0-9]+]] = icmp eq i32 [[NY]], 8
// CHECK-NEXT: [[XY:%[0-9]+]] = and i1 [[X]], [[Y]]
// CHECK-NEXT: [[NZ:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
// CHECK-NEXT: [[Z:%[0-9]+]] = icmp eq i32 [[NZ]], 1
// CHECK-NEXT: [[SAME:%[0-9]+]] = and i1 [[Z]], [[XY]]
// CHECK-NEXT: br i1 [[SAME]], label %raise.entry, label %{{.+}}
// CHECK: [[RES:%.+]] = phi double [ [[SUM:%[0-9]+]], %{{[^ ]+}} ], [ [[DD:%[0-9]+]], %wmma_store_gate.merge ]
// CHECK: [[OP:%[0-9]+]] = getelementptr {{.*}}ptr [[PD]], i64 {{%[0-9]+}}
// CHECK-NEXT: store double [[RES]], ptr [[OP]]
// CHECK-NEXT: ret void
// CHECK: {{^}}raise.entry:
// CHECK: [[FA:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) [[TA:@__poseidon_raise_preprocess_scalar_matmul_body_f64_a_f16_16x16]] to ptr), i32 16)
// CHECK-NEXT: [[FB:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) [[TB:@__poseidon_raise_preprocess_scalar_matmul_body_f64_b_f16_16x16]] to ptr), i32 16)
// CHECK-NEXT: [[A0:%[0-9]+]] = extractvalue {{.*}} [[FA]], 0
// CHECK: [[B0:%[0-9]+]] = extractvalue {{.*}} [[FB]], 0
// CHECK: [[C:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16(<2 x half> [[A0]], {{.*}}, <2 x half> [[B0]], {{.*}}, <2 x half> zeroinitializer, <2 x half> zeroinitializer, <2 x half> zeroinitializer, <2 x half> zeroinitializer)
// CHECK-NEXT: [[C0:%[0-9]+]] = extractvalue {{.*}} [[C]], 0
// CHECK: [[M0:%[^ ]+]] = phi <2 x half> [ [[C0]], {{%[^ ]+}} ], [ zeroinitializer, {{%[^ ]+}} ]
// CHECK: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) [[TD:@__poseidon_raise_preprocess_scalar_matmul_body_f64_d_f16_16x16]] to ptr), <2 x half> [[M0]], {{.*}}, i32 16)
// CHECK: [[DP:%[0-9]+]] = getelementptr {{.*}}ptr addrspacecast (ptr addrspace(3) [[TD]] to ptr), i64 {{%[0-9]+}}
// CHECK-NEXT: [[DH:%[0-9]+]] = load half, ptr [[DP]]
// CHECK-NEXT: [[DD]] = fpext half [[DH]] to double
// CHECK: [[SUM]] = tail call contract double @llvm.fmuladd.f64(
// CHECK: {{^}$}}

// The loop's multiply-add is the only floating-point arithmetic left.
// NOFP-LABEL: ; ModuleID =
// NOFP-LABEL: define dso_local void @preprocess_scalar_matmul_body_f64(
// NOFP-NOT: {{fadd|fsub|fmul|fma|fdiv}}
// NOFP: tail call contract double @llvm.fmuladd.f64(
// NOFP-NOT: {{fadd|fsub|fmul|fma|fdiv}}
// NOFP: {{^}$}}
