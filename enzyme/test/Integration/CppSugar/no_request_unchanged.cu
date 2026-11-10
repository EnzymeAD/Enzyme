// A module that asks nothing of Enzyme compiles as it does without the plugin:
// on the host, and in CUDA device code, where the plugin keeps every function
// of clang's CUDA math header alive in case a derivative calls it. The header
// is stood in for, so no CUDA installation is needed.

// REQUIRES: nvptx-registered-target
// RUN: rm -rf %t && split-file %s %t
// RUN: %clang -O2 -S -emit-llvm %t/host.c -o %t/host.ll
// RUN: %clang -O2 -S -emit-llvm %t/host.c -o %t/host.enzyme.ll %loadClangEnzyme
// RUN: diff %t/host.ll %t/host.enzyme.ll
// RUN: %clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 -O2 -S -include %t/__clang_cuda_math.h %t/device.cu -o %t/device.ptx
// RUN: %clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 -O2 -S -include %t/__clang_cuda_math.h %t/device.cu -o %t/device.enzyme.ptx %loadClangEnzyme
// RUN: diff %t/device.ptx %t/device.enzyme.ptx

//--- host.c
#include <string.h>

int parse(int argc, char **argv) {
  int flags = 0;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-i") == 0)
      flags |= 1;
    else if (strcmp(argv[i], "-s") == 0)
      flags |= 2;
    else if (strcmp(argv[i], "-r") == 0)
      flags |= 4;
  }
  return flags;
}

//--- __clang_cuda_math.h
extern "C" __attribute__((device)) double __nv_sqrt(double);
extern "C" __attribute__((device)) double __nv_cos(double);
static __attribute__((device)) double sqrt(double x) { return __nv_sqrt(x); }
static __attribute__((device)) double cos(double x) { return __nv_cos(x); }

//--- device.cu
__attribute__((global)) void norm1p(double *o, const double *a, int n) {
  int i = __nvvm_read_ptx_sreg_tid_x();
  if (i < n)
    o[i] = sqrt(a[i] * a[i] + 1.0);
}
