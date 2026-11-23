// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t && echo fatbin > %t/k.fatbin
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm %loadClangPoseidonEnzyme %s -o %t/plain-dev.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen-dev.ll
// RUN: grep -v -e '^%[0-9]* = type' -e '^$' %t/plain-dev.ll > %t/plain-dev.cmp
// RUN: grep -v -e '^%[0-9]* = type' -e '^%%struct.PoseidonCUDAProfileTable = type' -e '^$' -e '__poseidon_profile_table' %t/gen-dev.ll > %t/gen-dev.cmp
// RUN: diff %t/plain-dev.cmp %t/gen-dev.cmp
// RUN: FileCheck --check-prefix=DEV %s < %t/gen-dev.ll
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O2 -S -emit-llvm %loadClangPoseidonEnzyme %s -o %t/plain-host.ll
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O2 -S -emit-llvm %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen-host.ll
// RUN: not diff %t/plain-host.ll %t/gen-host.ll > %t/host.diff
// RUN: FileCheck --check-prefix=HOST %s < %t/host.diff
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O0 -S -emit-llvm %loadClangPoseidonEnzyme -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen-host-bin.ll
// RUN: FileCheck --check-prefix=REG %s < %t/gen-host-bin.ll

// Profile generation compiles a unit without a site as a plain build does,
// but for what the program's other units need of it: main calls the
// profiler's registration, and the device module defines the profile table,
// which the host half registers whether or not the device half has a site.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

__device__ double helper(double v) {
  double s = 0;
  for (int k = 0; k < 4; ++k)
    s += v * k + s * 0.5;
  return s;
}

__global__ void mass(const double *B, const double *D, double *Y) {
  double t[8][8];
  for (int q = 0; q < 8; ++q)
    for (int d = 0; d < 8; ++d) {
      double s = 0;
      for (int k = 0; k < 8; ++k)
        s += B[q * 8 + k] * D[k * 8 + d];
      t[q][d] = helper(s);
    }
  for (int q = 0; q < 8; ++q)
    for (int d = 0; d < 8; ++d)
      Y[q * 8 + d] = t[q][d] * t[d][q];
}

int main() { return 0; }

// DEV: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(4) @__poseidon_profile_table to ptr)], section "llvm.metadata"
// DEV-NEXT: @__poseidon_profile_table = weak_odr dso_local addrspace(4) externally_initialized constant %struct.PoseidonCUDAProfileTable zeroinitializer, align 8

// HOST: > {{.*}}call void @__poseidon_profile_init_cuda()
// HOST: > declare void @__poseidon_profile_init_cuda()
// HOST-NOT: <

// REG: @__poseidon_profile_table = internal global [[[N:[0-9]+]] x i8] zeroinitializer, align 8
// REG-NEXT: @[[NAME:[0-9]+]] = private unnamed_addr constant [25 x i8] c"__poseidon_profile_table\00"
// REG-NEXT: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cuda_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }]
// REG-LABEL: define dso_local noundef i32 @main()
// REG-NEXT: call void @__poseidon_profile_init_cuda()
// REG: define internal {{.*}}void @__cuda_register_globals(ptr %[[H:[0-9]+]])
// REG: call void @__cudaRegisterVar(ptr %[[H]], ptr @__poseidon_profile_table, ptr @[[NAME]], ptr @[[NAME]], i32 0, i64 [[N]], i32 1, i32 0)
// REG-NEXT: ret void
// REG-LABEL: define internal void @__poseidon_profile_register_module_table()
// REG-NEXT: call void @__poseidon_profile_register_module(ptr @__poseidon_profile_table)
// REG-NEXT: ret void
