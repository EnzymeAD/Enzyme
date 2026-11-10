// RUN: rm -rf %t && mkdir -p %t
// RUN: %cmake %cmake_generator -S %S/Inputs/package -B %t/gen -DCMAKE_CXX_COMPILER=%clangxx -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=instrument > /dev/null
// RUN: %cmake --build %t/gen -v > %t/gen.build.log
// RUN: FileCheck --check-prefix=GEN %s < %t/gen.build.log
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen/host | FileCheck --check-prefix=OUT %s
// RUN: %cmake %cmake_generator -S %S/Inputs/package -B %t/use -DCMAKE_CXX_COMPILER=%clangxx -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=optimize -DPOSEIDON_PROFILE=%t/prof "-DPOSEIDON_FLAGS=-poseidon-tau=1e-6 -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0" > /dev/null
// RUN: %cmake --build %t/use -- -t commands host > %t/use.commands
// RUN: FileCheck --check-prefix=USE %s < %t/use.commands
// REQUIRES: poseidon, enzyme, amdgpu-registered-target, rocm

// package_find.cpp with the HIP runtimes chosen (POSEIDON_GPU_RUNTIME=HIP):
// every Poseidon flag reaches the compiler through -Xclang, since HIP's driver
// hands -mllvm to the device linker as well, and the actions link the HIP
// builds of the profiler and of the GEMM dispatch (the optimized build's
// commands only: it links the ROCm installation's hipBLAS).

// GEN: -Xclang -mllvm -Xclang -poseidon-profile-generate
// GEN: libposeidon_profile_host.a {{.*}}libposeidon_profile_hip.a

// OUT: wave 348.384

// USE: -Xclang -mllvm -Xclang -poseidon-profile-use={{[^ ]+}}/prof -Xclang -mllvm -Xclang -poseidon-tau=1e-6
// USE: libposeidon_rt_hip.a {{.*}}libhipblas.so {{.*}}libamdhip64.so
