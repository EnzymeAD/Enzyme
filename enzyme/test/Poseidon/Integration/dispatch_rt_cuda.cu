// The GEMM dispatch runtimes on cuBLAS against cublasDgemm (Inputs/
// dispatch_rt.cpp): every Ozaki-II case returns the bits the HIP build is held
// to, every direct mode but TF32 stays within its a priori bound, and the FP32
// mode is cublasSgemm bit for bit.
//
// REQUIRES: cuda-runtime
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 -I%poseidon_runtime_inc \
// RUN:     %S/../Inputs/dispatch_rt.cpp -x none %poseidon_rt_lib \
// RUN:     -L%cuda_lib_dir -lcublas -lcudart -lstdc++ -lm -o %t.exe
// RUN: %t.exe ozaki | FileCheck --check-prefix=OZAKI %S/../Inputs/dispatch_rt.cpp
// RUN: %t.exe direct | FileCheck --check-prefixes=SGEMM,DIRECT %S/../Inputs/dispatch_rt.cpp
