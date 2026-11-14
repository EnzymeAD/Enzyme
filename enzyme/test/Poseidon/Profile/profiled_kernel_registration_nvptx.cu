// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir -p %t && echo fatbin > %t/k.fatbin
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=u %s -o %t/dev.ll
// RUN: cat %t/c/launches_*.json | FileCheck --check-prefix=MANIFEST %s
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O0 -S -emit-llvm -I%poseidon_profile_inc -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=u %s -o %t/host.ll
// RUN: FileCheck --check-prefix=HOST %s < %t/host.ll
// RUN: grep -o 'c"SiteId[^"]*"' %t/dev.ll | sort > %t/dev.headers
// RUN: grep -o 'c"SiteId[^"]*"' %t/host.ll | sort > %t/host.headers
// RUN: count 2 < %t/dev.headers
// RUN: diff %t/dev.headers %t/host.headers
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O0 -S -emit-llvm -I%poseidon_profile_inc -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=other %s -o %t/other.ll
// RUN: FileCheck --check-prefix=NONE --implicit-check-not=poseidonRegisterProfileStatic %s < %t/other.ll
// RUN: %clang -x cuda --cuda-host-only -nocudainc -nocudalib -O0 -S -emit-llvm -I%poseidon_profile_inc -Xclang -fcuda-include-gpubinary -Xclang %t/k.fatbin %loadClangPoseidonEnzyme -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=u %s -o %t/plain.ll
// RUN: FileCheck --check-prefix=PLAIN --implicit-check-not=poseidonRegisterProfileStatic %s < %t/plain.ll

// The device profiler writes a kernel's profile only once the kernel has run,
// so the device half hands each profiled kernel's header to the host half
// through the launch manifest, and the host half registers it with the
// profiling runtime before main, as a host site registers its own. A host
// compile with no manifest for its unit, or not generating a profile,
// registers nothing.

#include "poseidon/poseidon.h"

#define __global__ __attribute__((global))

struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

POSEIDON_OPTIMIZE __global__ void scale(double *x, double a) {
  x[0] = a * x[0] + 1.0;
}

POSEIDON_OPTIMIZE __global__ void norm2(const double *a, double *y) {
  double s = 0.0;
  for (int k = 0; k < 16; ++k)
    s += a[k] * a[k];
  y[0] = s;
}

// MANIFEST: "profile headers": {
// MANIFEST-DAG: "_Z5norm2PKdPd": {
// MANIFEST-DAG: "profile": "preprocess__Z5norm2PKdPd_poseidon_body"
// MANIFEST-DAG: "_Z5scalePdd": {
// MANIFEST-DAG: "profile": "preprocess__Z5scalePdd_poseidon_body"
// MANIFEST-DAG: "header": "SiteId = {{[0-9]+}}\nCanonicalHash = {{[0-9a-f]+}}\nProfileFormat = 2\n"

// HOST: @llvm.global_ctors = {{.*}}{ i32 0, ptr @poseidon_register_static_preprocess__Z5norm2PKdPd_poseidon_body, ptr null }, { i32, ptr, ptr } { i32 0, ptr @poseidon_register_static_preprocess__Z5scalePdd_poseidon_body, ptr null }]
// HOST-LABEL: define internal void @poseidon_register_static_preprocess__Z5norm2PKdPd_poseidon_body()
// HOST-NEXT: call void @poseidonRegisterProfileStatic(ptr @poseidon_site_preprocess__Z5norm2PKdPd_poseidon_body, ptr @fpprofile_static_preprocess__Z5norm2PKdPd_poseidon_body)
// HOST-LABEL: define internal void @poseidon_register_static_preprocess__Z5scalePdd_poseidon_body()
// HOST-NEXT: call void @poseidonRegisterProfileStatic(ptr @poseidon_site_preprocess__Z5scalePdd_poseidon_body, ptr @fpprofile_static_preprocess__Z5scalePdd_poseidon_body)

// NONE: @__poseidon_profile_table

// PLAIN: define
