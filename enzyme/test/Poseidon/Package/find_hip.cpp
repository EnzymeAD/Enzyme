// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/gen -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=instrument > /dev/null
// RUN: %cmake --build %t/gen -v > %t/gen.build.log
// RUN: FileCheck --check-prefix=GEN %s < %t/gen.build.log
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen/host | FileCheck --check-prefix=OUT %s
// RUN: %cmake %cmake_generator -S %S/../Inputs/package -B %t/use -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=optimize -DPOSEIDON_PROFILE=%t/prof "-DPOSEIDON_FLAGS=-poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-algebraic=0" > /dev/null
// RUN: %cmake --build %t/use -- -t commands host > %t/use.commands
// RUN: FileCheck --check-prefix=USE %s < %t/use.commands
// RUN: %cmake --install %enzyme_binary_dir/Enzyme --prefix %t/inst > /dev/null
// RUN: %cmake --install %poseidon_package_dir --prefix %t/inst > /dev/null
// RUN: not grep -rI '%rocm_path' %t/inst/%install_libdir/cmake/Poseidon
// RUN: mkdir -p %t/rocm/lib && touch %t/rocm/lib/libhipblas.so %t/rocm/lib/libamdhip64.so
// RUN: env ROCM_PATH=%t/rocm %cmake %cmake_generator -S %S/../Inputs/package -B %t/iuse -DCMAKE_CXX_COMPILER=%clang++ -DCMAKE_PREFIX_PATH=%t/inst -DPOSEIDON_GPU_RUNTIME=HIP -DPOSEIDON_MODE=optimize -DPOSEIDON_PROFILE=%t/prof > /dev/null
// RUN: %cmake --build %t/iuse -- -t commands host | FileCheck --check-prefix=IUSE -DLIBDIR=%install_libdir -DPREFIX=%t %s

// find.cpp with POSEIDON_GPU_RUNTIME=HIP: flags go through -Xclang (HIP hands
// -mllvm to the device linker too), profile generation passes
// -enzyme-lto-prelink, and the actions link the HIP runtimes (hipBLAS of the
// project's ROCm, ROCM_PATH else /opt/rocm, once installed).

// GEN: -Xclang -mllvm -Xclang -poseidon-profile-generate
// GEN-SAME: -Xclang -mllvm -Xclang -enzyme-lto-prelink=1
// GEN: libposeidon_profile_shadows.a {{.*}}libposeidon_profile_hip.a
// OUT: wave 348.384
// USE: -Xclang -mllvm -Xclang -poseidon-profile-use={{[^ ]+}}/prof -Xclang -mllvm -Xclang -poseidon-tau=1e-6
// USE: libposeidon_gemm_hip.a {{.*}}libhipblas.so {{.*}}libamdhip64.so
// IUSE: [[PREFIX]]/inst/[[LIBDIR]]/libposeidon_gemm_hip.a [[PREFIX]]/rocm/lib/libhipblas.so [[PREFIX]]/rocm/lib/libamdhip64.so
