// The GEMM dispatch runtimes on cuBLAS against cublasDgemm (Inputs/
// dispatch_rt.cpp): every Ozaki-II case returns the bits the HIP build is held
// to, every direct mode but TF32 stays within its a priori bound, the FP32
// mode is cublasSgemm bit for bit and the TF32 mode the library's TF32 product
// of operands rounded as the in-kernel raise rounds them, also with every
// extent above 1024, where a cuMpSGEMM bound ahead of cuBLAS (a build with
// TCEC links it, after cuBLAS) would take the call over; an empty contraction
// scales C by beta, calls on two streams return the bits they return alone,
// and a negative extent stops the program.
//
// REQUIRES: cuda-runtime
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 -I%poseidon_runtime_inc \
// RUN:     %S/../Inputs/gemm_runtime.cpp -x none %poseidon_gemm_cuda_lib \
// RUN:     -L%cuda_lib_dir -lcublas -lcudart -lstdc++ -lm -o %t.exe
// RUN: %t.exe ozaki | FileCheck --check-prefix=OZAKI %S/../Inputs/gemm_runtime.cpp
// RUN: env -u CUMPSGEMM_COMPUTE_MODE %t.exe direct \
// RUN:   | FileCheck --check-prefixes=SGEMM,DIRECT %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe empty | FileCheck --check-prefixes=EMPTY,EMPTY-CUDA %S/../Inputs/gemm_runtime.cpp
// RUN: %t.exe streams | FileCheck --check-prefix=STREAMS %S/../Inputs/gemm_runtime.cpp
// RUN: not --crash %t.exe negative 2>&1 | FileCheck --check-prefix=NEGATIVE %S/../Inputs/gemm_runtime.cpp
