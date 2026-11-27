// ozaki_moduli_counts.cu compiled as HIP, against the HIP Ozaki-II runtime on
// hipBLAS: every modulus count gives the simulation's bits there too.
// REQUIRES: hip-device
// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -frounding-math -I%S/../../../Poseidon -c %S/../../../Poseidon/Techniques/Matmul/Simulation.cpp -o %t/sim.o
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 -ffp-contract=off -I%S/../../../Poseidon/Techniques/Matmul -I%poseidon_runtime_inc %S/ozaki_moduli_counts.cu -x none %t/sim.o %poseidon_gemm_hip_lib -L%rocm_path/lib -lhipblas -lamdhip64 -lstdc++ -lm -o %t/exe
// RUN: %t/exe | FileCheck --match-full-lines %S/ozaki_moduli_counts.cu
