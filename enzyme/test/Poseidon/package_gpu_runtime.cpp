// RUN: rm -rf %t && mkdir -p %t/hip
// RUN: sed 's/^set(POSEIDON_HAS_HIP .*)$/set(POSEIDON_HAS_HIP ON)/' %poseidon_package_dir/PoseidonConfig.cmake > %t/hip/PoseidonConfig.cmake
// RUN: cp %poseidon_package_dir/PoseidonConfigVersion.cmake %poseidon_package_dir/PoseidonTargets.cmake %t/hip/
// RUN: %cmake %cmake_generator -S %S/Inputs/package_options -B %t/hip.build -DPoseidon_DIR=%t/hip -DPOSEIDON_GPU_RUNTIME=HIP > %t/hip.log
// RUN: FileCheck --check-prefix=HIP %s < %t/hip.log
// RUN: %cmake %cmake_generator -S %S/Inputs/package_options -B %t/cuda.build -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=CUDA > %t/cuda.log
// RUN: FileCheck --check-prefix=CUDA --implicit-check-not=enzyme-lto-prelink %s < %t/cuda.log
// REQUIRES: poseidon

// The action targets of this build tree's package for either GPU runtime,
// without the runtime's toolchain: the HIP arm reads the package as a build
// with the HIP runtimes would configure it. With HIP every Poseidon option
// goes through -Xclang, and the action that loads Enzyme, profile generation,
// also tells Enzyme that a HIP compile is an LTO pre-link compile, as
// poseidon-clang does; with CUDA no action gets that option.

// HIP: -- HIP Plain: -fpass-plugin={{[^;]+}};-ffp-contract=on
// HIP-NEXT: -- HIP ProfileGenerate: -fpass-plugin={{[^;]+}};-fpass-plugin={{[^;]+}};SHELL:-Xclang -mllvm -Xclang -poseidon-profile-generate;SHELL:-Xclang -mllvm -Xclang -enzyme-lto-prelink=1;-ffp-contract=on
// HIP-NEXT: -- HIP ProfileUse: -fpass-plugin={{[^;]+}};-ffp-contract=on

// CUDA: -- CUDA Plain: -fpass-plugin={{[^;]+}};-ffp-contract=on
// CUDA-NEXT: -- CUDA ProfileGenerate: -fpass-plugin={{[^;]+}};-fpass-plugin={{[^;]+}};SHELL:-mllvm -poseidon-profile-generate;-ffp-contract=on
// CUDA-NEXT: -- CUDA ProfileUse: -fpass-plugin={{[^;]+}};-ffp-contract=on
