// The library call runtimes on cuBLAS (Inputs/gemm_runtime.cpp), FP32 and TF32
// also bit exact against the library's products.
// REQUIRES: cuda-device
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 -I%poseidon_runtime_inc %S/../Inputs/gemm_runtime.cpp -x none %poseidon_gemm_cuda_lib -L%cuda_lib_dir -lcublas -lcudart -lstdc++ -lm -o %t.exe
// RUN: %t.exe ozaki | FileCheck --check-prefix=OZAKI %S/../Inputs/gemm_runtime.cpp
// RUN: %poseidon_calibrate_library_cuda operand-min 256 1 f16_f32 bf16_f32 > %t.floors
// RUN: env -u CUMPSGEMM_COMPUTE_MODE %t.exe direct %t.floors | FileCheck --check-prefixes=SGEMM,DIRECT %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe empty | FileCheck --check-prefixes=EMPTY,EMPTY-CUDA %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe streams | FileCheck --check-prefix=STREAMS %S/../Inputs/gemm_runtime.cpp
// RUN: not --crash %t.exe negative 2>&1 | FileCheck --check-prefix=NEGATIVE %S/../Inputs/gemm_runtime.cpp
