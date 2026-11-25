// ozaki_moduli_counts.cu compiled as HIP, against the HIP Ozaki-II runtime on
// hipBLAS: every modulus count gives the simulation's bits there too.
//
// REQUIRES: hip-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -frounding-math \
// RUN:     -I%S/../../../Poseidon -c \
// RUN:     %S/../../../Poseidon/Techniques/Matmul/Simulation.cpp \
// RUN:     -o %t/sim.o
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 \
// RUN:     -ffp-contract=off -I%S/../../../Poseidon/Techniques/Matmul \
// RUN:     -I%poseidon_runtime_inc %S/ozaki_moduli_counts.cu -x none %t/sim.o \
// RUN:     %poseidon_gemm_hip_lib -L%rocm_path/lib -lhipblas -lamdhip64 \
// RUN:     -lstdc++ -lm -o %t/exe
// RUN: %t/exe | FileCheck --match-full-lines %S/ozaki_moduli_counts.cu
