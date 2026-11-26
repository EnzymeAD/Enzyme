// The library call runtimes compiled as HIP, on hipBLAS, against
// hipblasDgemm (Inputs/gemm_runtime.cpp): every Ozaki-II case returns the bits
// the CUDA runtime returns on an RTX 5090, and every direct mode but TF32,
// which HIP lacks, stays within its a priori bound; an empty contraction
// scales C by beta, calls on two streams return the bits they return alone,
// and a negative extent stops the program.
//
// REQUIRES: hip-device
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 \
// RUN:     -I%poseidon_runtime_inc %S/../Inputs/gemm_runtime.cpp -x none \
// RUN:     %poseidon_gemm_hip_lib -L%rocm_path/lib -lhipblas -lamdhip64 \
// RUN:     -lstdc++ -lm -o %t.exe
// RUN: %t.exe ozaki | FileCheck --check-prefix=OZAKI %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe direct | FileCheck --check-prefix=DIRECT %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe empty | FileCheck --check-prefix=EMPTY %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe streams | FileCheck --check-prefix=STREAMS %S/../Inputs/gemm_runtime.cpp
// RUN: not --crash %t.exe negative 2>&1 | FileCheck --check-prefix=NEGATIVE %S/../Inputs/gemm_runtime.cpp
