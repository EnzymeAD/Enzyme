// The GEMM dispatch runtimes compiled as HIP, on hipBLAS, against
// hipblasDgemm (Inputs/dispatch_rt.cpp): every Ozaki-II case returns the bits
// the CUDA runtime returns on an RTX 5090, and every direct mode but TF32,
// which HIP lacks, stays within its a priori bound.
//
// REQUIRES: hip-runtime
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 \
// RUN:     -I%poseidon_runtime_inc %S/../Inputs/dispatch_rt.cpp -x none \
// RUN:     %poseidon_rt_hip_lib -L%rocm_path/lib -lhipblas -lamdhip64 \
// RUN:     -lstdc++ -lm -o %t.exe
// RUN: %t.exe ozaki | FileCheck --check-prefix=OZAKI %S/../Inputs/dispatch_rt.cpp
// RUN: %t.exe direct | FileCheck --check-prefix=DIRECT %S/../Inputs/dispatch_rt.cpp
