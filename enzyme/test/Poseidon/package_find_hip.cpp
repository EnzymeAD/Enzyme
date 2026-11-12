// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %cmake %cmake_generator -S %S/Inputs/package -B %t/gen -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=instrument > /dev/null
// RUN: %cmake --build %t/gen -v > %t/gen.build.log
// RUN: FileCheck --check-prefix=GEN %s < %t/gen.build.log
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen/host | FileCheck --check-prefix=OUT %s
// RUN: %cmake %cmake_generator -S %S/Inputs/package -B %t/use -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=optimize -DPOSEIDON_PROFILE=%t/prof "-DPOSEIDON_FLAGS=-poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0" > /dev/null
// RUN: %cmake --build %t/use -- -t commands host > %t/use.commands
// RUN: FileCheck --check-prefix=USE %s < %t/use.commands
// RUN: %cmake --install %enzyme_binary_dir/Enzyme --prefix %t/inst > /dev/null
// RUN: %cmake --install %poseidon_package_dir --prefix %t/inst > /dev/null
// RUN: not grep -rI '%rocm_path' %t/inst/lib/cmake/Poseidon
// RUN: mkdir -p %t/rocm/lib && touch %t/rocm/lib/libhipblas.so %t/rocm/lib/libamdhip64.so
// RUN: env ROCM_PATH=%t/rocm %cmake %cmake_generator -S %S/Inputs/package -B %t/iuse -DCMAKE_CXX_COMPILER=%clang++ -DCMAKE_PREFIX_PATH=%t/inst -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=optimize -DPOSEIDON_PROFILE=%t/prof > /dev/null
// RUN: %cmake --build %t/iuse -- -t commands host | FileCheck --check-prefix=IUSE -DPREFIX=%t %s

// package_find.cpp with the HIP runtimes chosen (POSEIDON_GPU_RUNTIME=HIP):
// every Poseidon flag reaches the compiler through -Xclang, since HIP's driver
// hands -mllvm to the device linker as well, profile generation tells Enzyme
// that a HIP compile is an LTO pre-link compile, and the actions link the HIP
// builds of the profiler and of the GEMM dispatch (the optimized build's
// commands only: it links the ROCm installation's hipBLAS).

// GEN: -Xclang -mllvm -Xclang -poseidon-profile-generate
// GEN-SAME: -Xclang -mllvm -Xclang -enzyme-lto-prelink=1
// GEN: libposeidon_profile_host.a {{.*}}libposeidon_profile_hip.a

// OUT: wave 348.384

// USE: -Xclang -mllvm -Xclang -poseidon-profile-use={{[^ ]+}}/prof -Xclang -mllvm -Xclang -poseidon-tau=1e-6
// USE: libposeidon_rt_hip.a {{.*}}libhipblas.so {{.*}}libamdhip64.so

// Installed, the package names no file of the ROCm it was built against: a
// project links hipBLAS and the HIP runtime of the ROCm installation of its
// own machine (ROCM_PATH, else /opt/rocm).
// IUSE: [[PREFIX]]/inst/lib/libposeidon_rt_hip.a [[PREFIX]]/rocm/lib/libhipblas.so [[PREFIX]]/rocm/lib/libamdhip64.so
