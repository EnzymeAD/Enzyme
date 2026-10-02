// The three user actions through poseidon-clang++, on the kernel
// raise_scalar_matmul_f64.cu profiles and raises: no plugin paths, no runtime
// sources, no -mllvm.
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate %S/raise_scalar_matmul_f64.cu -o %t.exe
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
//
// RUN: rm -rf %t.cache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.cache \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-enable-pt=0 -poseidon-raise-wmma \
// RUN:     %S/raise_scalar_matmul_f64.cu -o %t.opt.exe 2>&1 | FileCheck %s
// RUN: %t.opt.exe | FileCheck --check-prefix=PRIMAL %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

// PRIMAL: MATMUL-PASS

// CHECK: [poseidon] Found 1 AbstractMatmul(s) for preprocess_scalar_matmul_body_f64
// CHECK-NEXT: Matmul[0]: 8x8x8 a=f64 b=f64 acc=f64 d=f64 origin=ScalarLoopReduction
// CHECK-NOT: Matmul[1]:
// CHECK: Applying solution for matmul[0] -> wmma m16n16k16 f16/f16
