// RUN: rm -rf %t && split-file %s %t && mkdir -p %t/hip
// RUN: sed 's/^set(POSEIDON_HAS_HIP .*)$/set(POSEIDON_HAS_HIP ON)/' %poseidon_package_dir/PoseidonConfig.cmake > %t/hip/PoseidonConfig.cmake
// RUN: cp %poseidon_package_dir/PoseidonConfigVersion.cmake %poseidon_package_dir/PoseidonTargets.cmake %t/hip/
// RUN: %cmake %cmake_generator -S %t/options -B %t/hip.build -DPoseidon_DIR=%t/hip -DPOSEIDON_GPU_RUNTIME=HIP > %t/hip.log
// RUN: FileCheck --check-prefix=HIP -DARCH=%amdgpu_arch %s < %t/hip.log
// RUN: %cmake %cmake_generator -S %t/options -B %t/cuda.build -DPoseidon_DIR=%poseidon_package_dir -DPOSEIDON_GPU_RUNTIME=CUDA > %t/cuda.log
// RUN: FileCheck --check-prefixes=CUDA,%if cuda-toolkit %{CUDATK%} %else %{NOCUDATK%} \
// RUN:   --implicit-check-not=enzyme-lto-prelink %s < %t/cuda.log

// The action targets of the package for either GPU runtime, without its
// toolchain (for HIP, a copy configured as a HIP build would be). Options go
// through -Xclang; with HIP, profile generation also passes
// -enzyme-lto-prelink=1, exported as Poseidon::EnzymeOptions and
// POSEIDON_ENZYME_OPTIONS. Only a CUDA build looks for the CUDA toolkit; the
// HIP package names the AMD GPU processor of its builds.

// HIP: -- HIP Plain: -fpass-plugin=$<TARGET_FILE:Poseidon::Poseidon-{{[0-9]+}}>;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// HIP-NEXT: -- HIP ProfileGenerate: -fpass-plugin=$<TARGET_FILE:Poseidon::Poseidon-{{[0-9]+}}>;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-fpass-plugin={{[^;]+}}/ClangEnzyme-{{[0-9]+}}.so;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}SHELL:-Xclang -mllvm -Xclang -poseidon-profile-generate;SHELL:-Xclang -mllvm -Xclang -enzyme-lto-prelink=1;-ffp-contract=on
// HIP-NEXT: -- HIP ProfileUse: -fpass-plugin=$<TARGET_FILE:Poseidon::Poseidon-{{[0-9]+}}>;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// HIP-NEXT: -- HIP EnzymeOptions: SHELL:-Xclang -mllvm -Xclang -enzyme-lto-prelink=1{{$}}
// HIP-NEXT: -- HIP POSEIDON_ENZYME_OPTIONS: [SHELL:-Xclang -mllvm -Xclang -enzyme-lto-prelink=1]{{$}}
// HIP-NEXT: -- HIP CUDAToolkit: []{{$}}
// HIP-NEXT: -- HIP POSEIDON_AMDGPU_ARCH: [[[ARCH]]]{{$}}
// CUDA: -- CUDA Plain: -fpass-plugin=$<TARGET_FILE:Poseidon::Poseidon-{{[0-9]+}}>;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// CUDA-NEXT: -- CUDA ProfileGenerate: -fpass-plugin=$<TARGET_FILE:Poseidon::Poseidon-{{[0-9]+}}>;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-fpass-plugin={{[^;]+}}/ClangEnzyme-{{[0-9]+}}.so;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}SHELL:-Xclang -mllvm -Xclang -poseidon-profile-generate;-ffp-contract=on
// CUDA-NEXT: -- CUDA ProfileUse: -fpass-plugin=$<TARGET_FILE:Poseidon::Poseidon-{{[0-9]+}}>;{{(SHELL:-Xclang -load -Xclang [^;]+;)?}}-ffp-contract=on
// CUDA-NEXT: -- CUDA EnzymeOptions: options-NOTFOUND
// CUDA-NEXT: -- CUDA POSEIDON_ENZYME_OPTIONS: []{{$}}
// CUDATK-NEXT: -- CUDA CUDAToolkit: [TRUE]{{$}}
// NOCUDATK-NEXT: -- CUDA CUDAToolkit: []{{$}}
// CUDA-NEXT: -- CUDA POSEIDON_AMDGPU_ARCH: [{{(gfx[0-9a-f]+)?}}]{{$}}

//--- options/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(PoseidonPackageOptions NONE)

find_package(Poseidon CONFIG REQUIRED)
foreach(action Plain ProfileGenerate ProfileUse EnzymeOptions)
  get_target_property(options Poseidon::${action} INTERFACE_COMPILE_OPTIONS)
  message(STATUS "${POSEIDON_GPU_RUNTIME} ${action}: ${options}")
endforeach()
message(
  STATUS
    "${POSEIDON_GPU_RUNTIME} POSEIDON_ENZYME_OPTIONS: "
    "[${POSEIDON_ENZYME_OPTIONS}]")
message(STATUS "${POSEIDON_GPU_RUNTIME} CUDAToolkit: [${CUDAToolkit_FOUND}]")
message(
  STATUS
    "${POSEIDON_GPU_RUNTIME} POSEIDON_AMDGPU_ARCH: " "[${POSEIDON_AMDGPU_ARCH}]")
