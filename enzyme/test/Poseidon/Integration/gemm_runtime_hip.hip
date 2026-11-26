// The library call runtimes on hipBLAS (Inputs/gemm_runtime.cpp); HIP has no
// TF32 mode.
// REQUIRES: hip-device
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 -I%poseidon_runtime_inc %S/../Inputs/gemm_runtime.cpp -x none %poseidon_gemm_hip_lib -L%rocm_path/lib -lhipblas -lamdhip64 -lstdc++ -lm -o %t.exe
// RUN: %t.exe ozaki | FileCheck --check-prefix=OZAKI %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe direct | FileCheck --check-prefix=DIRECT %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe empty | FileCheck --check-prefix=EMPTY %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe streams | FileCheck --check-prefix=STREAMS %S/../Inputs/gemm_runtime.cpp
// RUN: not --crash %t.exe negative 2>&1 | FileCheck --check-prefix=NEGATIVE %S/../Inputs/gemm_runtime.cpp
