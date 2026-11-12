// RUN: rm -rf %t && mkdir -p %t/hip
// RUN: sed 's/^set(POSEIDON_HAS_HIP .*)$/set(POSEIDON_HAS_HIP ON)/' %poseidon_package_dir/PoseidonConfig.cmake > %t/hip/PoseidonConfig.cmake
// RUN: cp %poseidon_package_dir/PoseidonConfigVersion.cmake %poseidon_package_dir/PoseidonTargets.cmake %t/hip/
// RUN: %cmake %cmake_generator -S %S/../Inputs/package_options -B %t/hip.build -DPoseidon_DIR=%t/hip -DPOSEIDON_GPU_RUNTIME=HIP > %t/hip.log
// RUN: FileCheck --check-prefix=HIP %s < %t/hip.log
// RUN: %cmake %cmake_generator -S %S/../Inputs/package_options -B %t/cuda.build -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=CUDA > %t/cuda.log
// RUN: FileCheck --check-prefix=CUDA --implicit-check-not=enzyme-lto-prelink %s < %t/cuda.log

// The action targets of this build tree's package for either GPU runtime,
// without the runtime's toolchain: the HIP arm reads the package as a build
// with the HIP runtimes would configure it. Every Poseidon option goes through
// -Xclang, and with HIP the action that loads Enzyme, profile generation, also
// tells Enzyme that a HIP compile is an LTO pre-link compile, as
// poseidon-clang does. The package exports that option, as
// Poseidon::EnzymeOptions and POSEIDON_ENZYME_OPTIONS, for a project that
// loads Enzyme itself under Plain or ProfileUse, which do not carry it. With
// CUDA no target and no variable holds it. Only the CUDA runtimes need the
// CUDA toolkit; a project on the HIP runtimes is not made to find one. The
// AMD GPU processor of the HIP builds is the project's to see.

// HIP: -- HIP Plain: -fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// HIP-NEXT: -- HIP ProfileGenerate: -fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}SHELL:-Xclang -mllvm -Xclang -poseidon-profile-generate;SHELL:-Xclang -mllvm -Xclang -enzyme-lto-prelink=1;-ffp-contract=on
// HIP-NEXT: -- HIP ProfileUse: -fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// HIP-NEXT: -- HIP EnzymeOptions: SHELL:-Xclang -mllvm -Xclang -enzyme-lto-prelink=1{{$}}
// HIP-NEXT: -- HIP POSEIDON_ENZYME_OPTIONS: [SHELL:-Xclang -mllvm -Xclang -enzyme-lto-prelink=1]
// HIP-NEXT: -- HIP CUDAToolkit: []
// HIP-NEXT: -- HIP POSEIDON_AMDGPU_ARCH: [gfx{{[0-9a-f]+}}]

// CUDA: -- CUDA Plain: -fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// CUDA-NEXT: -- CUDA ProfileGenerate: -fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}SHELL:-Xclang -mllvm -Xclang -poseidon-profile-generate;-ffp-contract=on
// CUDA-NEXT: -- CUDA ProfileUse: -fpass-plugin={{[^;]+}};{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// CUDA-NEXT: -- CUDA EnzymeOptions: options-NOTFOUND
// CUDA-NEXT: -- CUDA POSEIDON_ENZYME_OPTIONS: []
// CUDA-NEXT: -- CUDA CUDAToolkit:
// CUDA-NEXT: -- CUDA POSEIDON_AMDGPU_ARCH: [{{(gfx[0-9a-f]+)?}}]
